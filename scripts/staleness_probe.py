import argparse
import collections
import json
import math
import os
import statistics
import sys
import time

HOME = os.environ.get(
    "CORRECTOR_HOME",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="pi0", choices=["pi0", "pi05"])
ap.add_argument("--openpi", default=f"{HOME}/openpi")
ap.add_argument("--sentry", default=f"{HOME}/src")
ap.add_argument("--assets", default=f"{HOME}/openpi_assets")
ap.add_argument("--tok", default=f"{HOME}/assets/paligemma_tokenizer.model")
ap.add_argument("--suite", default="libero_spatial")
ap.add_argument("--tasks", type=int, default=4)
ap.add_argument("--trials", type=int, default=3)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--replan", type=int, default=10,
                help="replan interval of the MAIN rollout, which only has to "
                     "keep the trajectory healthy; probes live on open-loop "
                     "branches and may reach far past it")
ap.add_argument("--probes", default="",
                help="elapsed steps k to measure at; empty picks a grid from H")
ap.add_argument("--ms", default="",
                help="horizons the aggregate distances average over; empty "
                     "picks from H")
ap.add_argument("--m-main", type=int, default=0,
                help="the horizon the terminal report and the crossing k* use")
ap.add_argument("--taus", default="0.3,0.5,0.7",
                help="model-times the stale chunk is re-noised to.  0.5 is the "
                     "setting the loop ships; 0.3 stays stale and 0.7 collapses "
                     "to the conditional mean, so the two ends are the window.")
ap.add_argument("--tau-main", type=float, default=0.5)
ap.add_argument("--refs", type=int, default=8,
                help="fresh plans forming the reference distribution at each "
                     "probe.  Witness 0 reuses the plan's own noise, so it "
                     "differs from the stale chunk by the OBSERVATION alone.")
ap.add_argument("--compile", choices=["default", "autotune", "off"],
                default="default")
ap.add_argument("--out", default="")
args = ap.parse_args()

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, f"{args.openpi}/src")
sys.path.insert(0, f"{args.openpi}/packages/openpi-client/src")
sys.path.insert(0, args.sentry)

import numpy as np
import torch

torch.set_num_threads(4)

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools

from sentry.core.types import Observation
from sentry.envs.libero_data import (
    DELTA_ACTION_DIMS as DELTA_DIMS, LiberoPrompt,
    state_spec_from_openpi_norm_stats,
)
from sentry.envs.libero_spec import find_norm_stats, from_openpi_norm_stats
from sentry.models.openpi_adapter import OpenPiBackend, load_pi0_pytorch

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
             "libero_10": 520, "libero_90": 400}
DUMMY = [0.0] * 6 + [-1.0]
WAIT = 10
RES = 256
POSE = 6
CONV, RAW = ((f"{args.assets}/pi0_libero_pytorch", f"{args.assets}/pi0_libero")
             if args.model == "pi0" else
             (f"{args.assets}/pi05_libero_pytorch", f"{args.assets}/pi05_libero"))
PI05 = args.model == "pi05"
DELTA = 0 if PI05 else DELTA_DIMS
QUANT = PI05

model = load_pi0_pytorch(CONV, device="cuda")
be = OpenPiBackend(model, device="cuda", M=10, attach_adapters=False)
model.eval()
M, H, D = be.M, be.H, be.d_a
FULL_V, FULL_B = be.L_V, be.L_B

PROBES = ([int(x) for x in args.probes.split(",")] if args.probes else
          ([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30] if H >= 50
           else [k for k in range(1, H - 1)]))
PROBES = sorted(k for k in PROBES if 0 < k < H)
MS = sorted(int(x) for x in args.ms.split(",")) if args.ms else \
    ([5, 10, 20] if H >= 50 else [2, 3, 5])
M_MAIN = args.m_main or (10 if H >= 50 else 3)
TAUS = [float(x) for x in args.taus.split(",")]
assert M_MAIN in MS, f"--m-main {M_MAIN} must be one of --ms {MS}"
assert args.tau_main in TAUS, "--tau-main must be one of --taus"
OUT = args.out or f"{HOME}/ckpt/staleness/{args.model}_{args.suite}_s{args.seed}.json"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

print(f"{args.model}: chunk H={H}, action dim {D}, {FULL_V} encoder / "
      f"{FULL_B} backbone layers, plan = {M} Euler steps")
print(f"probes k={PROBES}  horizons m={MS} (main {M_MAIN})  "
      f"taus={TAUS} (main {args.tau_main})  refs={args.refs}")

ns = find_norm_stats(RAW)
spec = from_openpi_norm_stats(ns, d_a=D, use_quantiles=QUANT)
sm, ss = state_spec_from_openpi_norm_stats(ns, d_state=8, use_quantiles=QUANT)
prompt = LiberoPrompt(args.tok, max_len=be.max_token_len)


def quat2axisangle(quat):
    q = np.asarray(quat, dtype=np.float64).copy()
    q[3] = min(1.0, max(-1.0, q[3]))
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def raw_state(d):
    return np.concatenate([d["robot0_eef_pos"],
                           quat2axisangle(d["robot0_eef_quat"]),
                           d["robot0_gripper_qpos"]]).astype(np.float32)


def to_observation(raw, tokens, t):
    img = np.ascontiguousarray(raw["agentview_image"][::-1, ::-1])
    wri = np.ascontiguousarray(raw["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
    wri = image_tools.convert_to_uint8(image_tools.resize_with_pad(wri, 224, 224))
    frames = np.stack([img, wri]).astype(np.float32) / 127.5 - 1.0
    images = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
    st = torch.zeros(D)
    st[:8] = (torch.from_numpy(raw_state(raw)) - sm) / ss
    return Observation(images=images, language=tokens, state=st, t=t)


@torch.no_grad()
def integrate(x0, tau0, n, obs):
    x = x0.clone().float().to(be.device)
    tau = tau0.clone().float().to(be.device)
    step = (tau / max(n, 1)).view(-1, 1, 1)
    for _ in range(n):
        v = be.velocity(A_tau=x, tau=tau, obs=obs, E_V=FULL_V, E_B=FULL_B,
                        adapters=False)
        x = x - step * v.float()
        tau = tau - step.view(-1)
    return x


def dist(X, Y, m):
    return float(torch.linalg.vector_norm(
        X[:m, :POSE] - Y[:m, :POSE], ord=2, dim=-1).mean())


def dist_pos(X, Y):
    return torch.linalg.vector_norm(X[:, :POSE] - Y[:, :POSE], ord=2, dim=-1)


def hold_pad(A, H):
    if A.shape[0] >= H:
        return A[:H].clone()
    return torch.cat([A, A[-1:].expand(H - A.shape[0], A.shape[1])], dim=0)


class LiberoEnv:

    def __init__(self, env, tokens, t0_raw):
        self._env, self._tokens, self._raw = env, tokens, t0_raw
        self._t = 0
        self._done = False
        self._anchor = raw_state(t0_raw)

    def mark_plan_anchor(self):
        self._anchor = raw_state(self._raw)

    def observe(self):
        return to_observation(self._raw, self._tokens, self._t)

    def step(self, action):
        a = action.detach().cpu() * spec.scale + spec.mean
        if DELTA:
            a[:DELTA] += torch.from_numpy(self._anchor[:DELTA])
        self._raw, _, done, _ = self._env.step(a[:7].tolist())
        self._t += 1
        self._done = self._done or bool(done)

    def frame_correction(self):
        if not DELTA:
            return None
        now = raw_state(self._raw)
        d = self._anchor[:DELTA] - now[:DELTA]
        return torch.from_numpy(d).float() / spec.scale[:DELTA].cpu()

    def save(self):
        return (self._env.get_sim_state(), self._raw, self._t, self._done,
                self._anchor.copy())

    def restore(self, snap):
        sim, raw, t, done, anchor = snap
        self._env.regenerate_obs_from_state(sim)
        self._raw, self._t, self._done = raw, t, done
        self._anchor = anchor

    @property
    def terminated(self):
        return self._done


REC = collections.defaultdict(list)
POS = collections.defaultdict(list)
VARIANTS = ["cold"] + [f"warm{t:g}" for t in TAUS]


@torch.no_grad()
def measure(env, A_plan, noise_plan, k, gen):
    obs_now = env.observe()
    fc = env.frame_correction()

    A_stale = hold_pad(A_plan[k:], H)
    if fc is not None:
        A_stale[:, :DELTA] += fc

    nz = torch.randn(args.refs - 1, H, D, generator=gen)
    x0 = torch.cat([noise_plan.cpu().unsqueeze(0), nz])
    R = integrate(x0, torch.ones(args.refs), M, obs_now).cpu()
    cent = R.mean(dim=0)

    eps = torch.randn(H, D, generator=gen)
    starts = [eps] + [t * eps + (1.0 - t) * A_stale for t in TAUS]
    tau0s = [1.0] + list(TAUS)
    out = integrate(torch.stack(starts), torch.tensor(tau0s), 1, obs_now).cpu()
    cand = {"identity": A_stale}
    for i, nm in enumerate(VARIANTS):
        cand[nm] = out[i]

    live = H - k
    for m in [m for m in MS if m <= live]:
        pair = [dist(R[i], R[j], m)
                for i in range(args.refs) for j in range(i + 1, args.refs)]
        REC[f"k{k}/m{m}/ref/spread"].append(statistics.median(pair))
        REC[f"k{k}/m{m}/ref/dnn"].append(statistics.median(
            [min(dist(R[p], R[q], m) for q in range(args.refs) if q != p)
             for p in range(args.refs)]))
        REC[f"k{k}/m{m}/ref/dcent"].append(statistics.median(
            [dist(R[p], cent, m) for p in range(args.refs)]))
        for tag, X in cand.items():
            REC[f"k{k}/m{m}/{tag}/dnn"].append(
                min(dist(X, R[p], m) for p in range(args.refs)))
            REC[f"k{k}/m{m}/{tag}/dcent"].append(dist(X, cent, m))

    m_pos = max([m for m in MS if m <= live] or [live])
    ref_pp = torch.stack([dist_pos(R[i], R[j])
                          for i in range(args.refs)
                          for j in range(i + 1, args.refs)])
    nn_ref, nn_ref_agg = [], []
    for p in range(args.refs):
        others = [q for q in range(args.refs) if q != p]
        dp = torch.stack([dist_pos(R[p], R[q]) for q in others])
        nn_ref.append(dp.min(dim=0).values)
        q_star = min(others, key=lambda q: dist(R[p], R[q], m_pos))
        nn_ref_agg.append(dist_pos(R[p], R[q_star]))
    POS[f"k{k}/ref/dnn"].append(
        torch.stack(nn_ref).median(dim=0).values[:live].tolist())
    POS[f"k{k}/ref/dnn_agg"].append(
        torch.stack(nn_ref_agg).median(dim=0).values[:live].tolist())
    POS[f"k{k}/ref/spread"].append(ref_pp.median(dim=0).values[:live].tolist())
    for tag, X in cand.items():
        dp = torch.stack([dist_pos(X, R[p]) for p in range(args.refs)])
        POS[f"k{k}/{tag}/dnn"].append(dp.min(dim=0).values[:live].tolist())
        p_star = min(range(args.refs), key=lambda p: dist(X, R[p], m_pos))
        POS[f"k{k}/{tag}/dnn_agg"].append(dp[p_star][:live].tolist())


suite = benchmark.get_benchmark_dict()[args.suite]()
n_tasks = suite.n_tasks if args.tasks == 0 else min(args.tasks, suite.n_tasks)
T_max = MAX_STEPS[args.suite]


def init_states_for(suite, task_id):
    task = suite.get_task(task_id)
    return torch.load(os.path.join(get_libero_path("init_states"),
                                   task.problem_folder, task.init_states_file),
                      weights_only=False)


_t0 = suite.get_task(0)
_bddl = os.path.join(get_libero_path("bddl_files"),
                     _t0.problem_folder, _t0.bddl_file)
_e = OffScreenRenderEnv(bddl_file_name=_bddl, camera_heights=RES, camera_widths=RES)
_e.seed(args.seed)
_e.reset()
_raw = _e.set_init_state(init_states_for(suite, 0)[0])
for _ in range(WAIT):
    _raw, _, _, _ = _e.step(DUMMY)
_obs = to_observation(_raw, prompt(_t0.language), 0)
_noise = torch.randn(H, D, generator=torch.Generator().manual_seed(3))
_ref = be.plan(_obs, noise=_noise).float()
_mine = integrate(_noise.unsqueeze(0), torch.ones(1), M, _obs)[0]
_gap = float(torch.linalg.vector_norm(_mine.cpu() - _ref.cpu(), ord=2, dim=-1).max())
print(f"integrator vs openpi sample_actions: max per-action gap {_gap:.2e}")
if _gap > 1e-2:
    sys.exit("hand-rolled integrator does not reproduce the sampler -- refusing "
             "to report distances from a solver that is not the policy")

COMPILE_MODE = None
if args.compile != "off":
    _eager = model.denoise_step
    _MODES = {"default": ("default", "max-autotune-no-cudagraphs", "max-autotune"),
              "autotune": ("max-autotune", "max-autotune-no-cudagraphs", "default")}
    for _mode in _MODES[args.compile]:
        try:
            torch._dynamo.reset()
            model.denoise_step = torch.compile(_eager, mode=_mode, dynamic=False)
            with torch.no_grad():
                _d = float(torch.linalg.vector_norm(
                    be.plan(_obs, noise=_noise).float() - _ref, ord=2, dim=-1).max())
            if _d > 0.05:
                print(f"  compile[{_mode}]: chunk moved {_d:.4f} -- rejected")
                continue
            COMPILE_MODE = _mode
            print(f"  compile[{_mode}]: chunk preserved ({_d:.2e})")
            break
        except Exception as e:
            print(f"  compile[{_mode}]: {type(e).__name__}: {str(e)[:70]}")
    if COMPILE_MODE is None:
        model.denoise_step = _eager
        print("  running eager")
    else:
        with torch.no_grad():
            integrate(torch.randn(1, H, D), torch.ones(1), M, _obs)
            integrate(torch.randn(args.refs, H, D), torch.ones(args.refs), M, _obs)
            integrate(torch.randn(1 + len(TAUS), H, D),
                      torch.full((1 + len(TAUS),), 0.5), 1, _obs)
_e.close()

t_start = time.time()
n_probe = n_ep = 0
for task_id in range(n_tasks):
    task = suite.get_task(task_id)
    tokens = prompt(task.language)
    bddl = os.path.join(get_libero_path("bddl_files"),
                        task.problem_folder, task.bddl_file)
    inits = init_states_for(suite, task_id)

    for trial in range(args.trials):
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=RES,
                                 camera_widths=RES)
        env.seed(args.seed)
        env.reset()
        raw = env.set_init_state(inits[trial % len(inits)])
        done = False
        for _ in range(WAIT):
            raw, _, done, _ = env.step(DUMMY)
        w = LiberoEnv(env, tokens, raw)
        w._done = done
        gen = torch.Generator().manual_seed(
            args.seed * 7919 + task_id * 131 + trial)

        steps = 0
        while not w.terminated and steps < T_max:
            w.mark_plan_anchor()
            noise_p = torch.randn(H, D, generator=gen).to(be.device)
            A_plan = integrate(noise_p.unsqueeze(0), torch.ones(1),
                               M, w.observe())[0].cpu()

            snap = w.save()
            for i in range(max(PROBES) + 1):
                if i in PROBES and not w.terminated:
                    measure(w, A_plan, noise_p, i, gen)
                    n_probe += 1
                if w.terminated:
                    break
                w.step(A_plan[i])
            w.restore(snap)

            for i in range(args.replan):
                if w.terminated or steps >= T_max:
                    break
                w.step(A_plan[i])
                steps += 1
        env.close()
        n_ep += 1
        print(f"  [{task_id + 1}/{n_tasks}] trial {trial}  {steps} steps  "
              f"{n_probe} probes  {time.time() - t_start:.0f}s", flush=True)

def med(key):
    v = REC.get(key)
    return statistics.median(v) if v else None


def pmed(key):
    v = POS.get(key)
    if not v:
        return None
    n = min(len(x) for x in v)
    return [statistics.median([x[h] for x in v]) for h in range(n)]


print(f"\n{n_ep} episodes, {n_probe} probes, {time.time() - t_start:.0f}s")
print(f"\n=== inside the plan distribution?  d_nn / ref_dnn at m={M_MAIN} "
      f"(<= 1.00 is inside) ===")
hdr = "   k  " + "".join(f"{t:>12}" for t in ["identity", "cold"]
                         + [f"warm{t:g}" for t in TAUS])
print(hdr)
rows = []
for k in PROBES:
    ref = med(f"k{k}/m{M_MAIN}/ref/dnn")
    if ref is None:
        continue
    line = f"{k:>4}  "
    for tag in ["identity", "cold"] + [f"warm{t:g}" for t in TAUS]:
        line += f"{med(f'k{k}/m{M_MAIN}/{tag}/dnn') / ref:>12.2f}"
    print(line)
    rows.append(k)

print(f"\n=== the same, against the TYPICAL plan-to-plan spread: "
      f"d_nn / ref_spread at m={M_MAIN} ===")
print(hdr)
for k in rows:
    ref = med(f"k{k}/m{M_MAIN}/ref/spread")
    line = f"{k:>4}  "
    for tag in ["identity", "cold"] + [f"warm{t:g}" for t in TAUS]:
        line += f"{med(f'k{k}/m{M_MAIN}/{tag}/dnn') / ref:>12.2f}"
    print(line)

print(f"\n=== a real sample, or the conditional mean?  d_cent / ref_dcent at "
      f"m={M_MAIN} (~1.00 is a sample, << 1 is mode-averaged) ===")
print(hdr)
for k in rows:
    ref = med(f"k{k}/m{M_MAIN}/ref/dcent")
    line = f"{k:>4}  "
    for tag in ["identity", "cold"] + [f"warm{t:g}" for t in TAUS]:
        line += f"{med(f'k{k}/m{M_MAIN}/{tag}/dcent') / ref:>12.2f}"
    print(line)

k_star = {}
for band in ("dnn", "spread"):
    k_star[band] = None
    for k in rows:
        if med(f"k{k}/m{M_MAIN}/identity/dnn") > med(f"k{k}/m{M_MAIN}/ref/{band}"):
            k_star[band] = k
            break
print(f"\nstaleness horizon k* at m={M_MAIN}: {k_star['dnn']} steps against the "
      f"nearest plan, {k_star['spread']} against the typical plan-to-plan spread")

print(f"\n=== how many LEADING actions are still inside the band after k steps "
      f"(of the H-k that carry real planned content) ===")
TAG_MAIN = f"warm{args.tau_main:g}"
print(f"{'k':>4} {'real':>6} {'identity':>10} {TAG_MAIN:>10} {'cold':>10}")
front = {}
for k in PROBES:
    ref = pmed(f"k{k}/ref/dnn_agg")
    if not ref:
        continue
    cell = {}
    for tag in ["identity", TAG_MAIN, "cold"]:
        d = pmed(f"k{k}/{tag}/dnn_agg")
        n = 0
        for h in range(len(ref)):
            if d[h] > ref[h]:
                break
            n += 1
        cell[tag] = n
    front[k] = cell
    print(f"{k:>4} {H - k:>6} {cell['identity']:>10} {cell[TAG_MAIN]:>10} "
          f"{cell['cold']:>10}")

out = {
    "meta": dict(model=args.model, suite=args.suite, seed=args.seed,
                 tasks=n_tasks, trials=args.trials, episodes=n_ep,
                 probes=PROBES, ms=MS, m_main=M_MAIN, taus=TAUS,
                 tau_main=args.tau_main, refs=args.refs, H=H, D=D, M=M,
                 replan=args.replan, n_probe=n_probe,
                 compile=COMPILE_MODE or "eager",
                 seconds=round(time.time() - t_start, 1),
                 k_star=k_star, frontier=front),
    "aggregate": {k: v for k, v in sorted(REC.items())},
    "per_position": {k: pmed(k) for k in sorted(POS)},
}
with open(OUT, "w") as f:
    json.dump(out, f)
print(f"\nwrote {OUT}  ({os.path.getsize(OUT) // 1024} KB)")
