import os as _os
HOME = _os.environ.get(
    "CORRECTOR_HOME",
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import argparse, collections, json, math, os, statistics, sys, time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
for p in (f"{HOME}/openpi/src", f"{HOME}/openpi/packages/openpi-client/src",
          f"{HOME}/src", f"{HOME}/scripts", f"{HOME}/third_party/aac_libero"):
    sys.path.insert(0, p)
import numpy as np, torch
torch.set_num_threads(4)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from sentry.core.types import Observation
from sentry.envs.libero_data import (DELTA_ACTION_DIMS as DELTA_DIMS, LiberoPrompt,
                                     state_spec_from_openpi_norm_stats)
from sentry.envs.libero_spec import find_norm_stats, from_openpi_norm_stats
from sentry.models.openpi_adapter import OpenPiBackend, load_pi0_pytorch, _expand_cache
from action_optimization.action_entropy_pi05 import select_chunk_size

ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="libero_spatial")
ap.add_argument("--tasks", type=int, default=0)
ap.add_argument("--trials", type=int, default=10)
ap.add_argument("--trial-start", type=int, default=0,
                help="first initial state to run; with --trials it covers\n"
                     "the range [start, start+trials), so a 200-scene\n"
                     "selection run at 0..19 and a 300-scene run at 20..49\n"
                     "together cover the standard 500 without redoing any")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--configs", nargs="+", default=["aac"], choices=["baseline", "aac"])
ap.add_argument("--model", choices=["pi0", "pi05"], default="pi0",
                help="base policy.  pi05 is openpi's pi05_libero: a 10-action "
                     "chunk, quantile normalisation and LIBERO's own actions "
                     "(no delta anchor), run at replan 5 as its own table does. "
                     "AAC's paper reports pi0.5 itself, where H = 10 is the "
                     "whole chunk it chooses from")
ap.add_argument("--replan", type=int, default=0,
                help="actions executed per plan by the baseline row; 0 takes "
                     "the model's own cadence -- 10 for pi_0, 5 for pi0.5")
ap.add_argument("--n", type=int, default=20, help="chunks per plan (paper: 20)")
ap.add_argument("--move-th", type=float, default=3.0, help="alpha, the motion floor (paper: 3)")
ap.add_argument("--window", type=int, default=0,
                help="actions AAC chooses from; 0 is the model's whole chunk H "
                     "(50 on pi_0, 10 on pi0.5, which is the paper's own setting)")
ap.add_argument("--timing-reps", type=int, default=40)
ap.add_argument("--latency-from", default="",
                help="read the cost model from a run measured on an IDLE GPU "
                     "instead of timing here.  Rollouts are unaffected by a "
                     "busy GPU -- they are deterministic given the seed -- but "
                     "timings are not, so this is what makes it sound to run "
                     "several evaluations at once")
ap.add_argument("--compile", choices=["default", "autotune", "off"], default="default")
ap.add_argument("--out", default=f"{HOME}/ckpt/aac/run.json")
args = ap.parse_args()

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
             "libero_10": 520, "libero_90": 400}
DUMMY = [0.0] * 6 + [-1.0]; WAIT = 10; RES = 256
CONV, RAW = {"pi0": (f"{HOME}/openpi_assets/pi0_libero_pytorch",
                     f"{HOME}/openpi_assets/pi0_libero"),
             "pi05": (f"{HOME}/openpi_assets/pi05_libero_pytorch",
                      f"{HOME}/openpi_assets/pi05_libero")}[args.model]
TOK = f"{HOME}/assets/paligemma_tokenizer.model"
PI05 = args.model == "pi05"
DELTA = 0 if PI05 else DELTA_DIMS
REPLAN = args.replan or (5 if PI05 else 10)

model = load_pi0_pytorch(CONV, device="cuda"); model.eval()
be = OpenPiBackend(model, device="cuda", M=10, attach_adapters=False)
M, H, D = be.M, be.H, be.d_a
WINDOW = args.window or H
assert 2 <= WINDOW <= H, f"--window must be in [2, {H}]"
ns = find_norm_stats(RAW); spec = from_openpi_norm_stats(ns, d_a=D, use_quantiles=PI05)
sm, ss = state_spec_from_openpi_norm_stats(ns, d_state=8, use_quantiles=PI05)
prompt = LiberoPrompt(TOK, max_len=be.max_token_len)


def quat2axisangle(quat):
    q = np.asarray(quat, dtype=np.float64).copy(); q[3] = min(1.0, max(-1.0, q[3]))
    den = np.sqrt(1.0 - q[3] * q[3])
    return np.zeros(3) if math.isclose(den, 0.0) else (q[:3] * 2.0 * math.acos(q[3])) / den


def raw_state(d):
    return np.concatenate([d["robot0_eef_pos"], quat2axisangle(d["robot0_eef_quat"]),
                           d["robot0_gripper_qpos"]]).astype(np.float32)


def to_observation(raw, tokens, t):
    img = np.ascontiguousarray(raw["agentview_image"][::-1, ::-1])
    wri = np.ascontiguousarray(raw["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
    wri = image_tools.convert_to_uint8(image_tools.resize_with_pad(wri, 224, 224))
    frames = np.stack([img, wri]).astype(np.float32) / 127.5 - 1.0
    st = torch.zeros(D); st[:8] = (torch.from_numpy(raw_state(raw)) - sm) / ss
    return Observation(images=torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous(),
                       language=tokens, state=st, t=t)


class LiberoEnv:
    def __init__(self, env, tokens, t0_raw):
        self._env, self._tokens, self._raw = env, tokens, t0_raw
        self._t = 0; self._done = False; self._anchor = raw_state(t0_raw)
    def mark_plan_anchor(self): self._anchor = raw_state(self._raw)
    def observe(self): return to_observation(self._raw, self._tokens, self._t)
    def step(self, action):
        a = action.detach().cpu() * spec.scale + spec.mean
        a[:DELTA] += torch.from_numpy(self._anchor[:DELTA])
        self._raw, _, done, _ = self._env.step(a[:7].tolist())
        self._t += 1; self._done = self._done or bool(done)
    @property
    def terminated(self): return self._done


@torch.no_grad()
def plan_samples(obs, noise):
    k = noise.shape[0]
    with be._truncated(be.L_V, be.L_B):
        pkv, ppm, st1 = be._prefix(obs)
    pkv = _expand_cache(pkv, k)
    ppm = ppm.expand(k, ppm.shape[1]); st = st1.expand(k, st1.shape[-1])
    x = noise.to(device="cuda", dtype=torch.float32); dt = -1.0 / M
    for i in range(M):
        t = torch.full((k,), 1.0 + i * dt, dtype=torch.float32, device="cuda")
        x = x + dt * model.denoise_step(st, ppm, pkv, x, t)
    return x


def aac_dict(S, anchor):
    s = S[:, :WINDOW, :7].float().cpu()
    a = s * spec.scale[:7] + spec.mean[:7]
    a[..., :DELTA] += torch.from_numpy(anchor[:DELTA])
    a = a.numpy()
    return {"normalized_action": s.numpy(),
            "action.end_effector_position": a[..., :3],
            "action.end_effector_rotation": a[..., 3:6],
            "action.gripper_close": a[..., 6]}


def aac_decide(S, anchor):
    h, br = select_chunk_size(aac_dict(S, anchor), method="gaussian_bernoulli",
                              move_th=args.move_th)
    return int(h), max(int(np.argmax(np.diff(br["chunk_mean"]))) + 1, 2)


@torch.no_grad()
def run_baseline(env, gen, gen_x, T_max):
    steps = plans = 0
    while not env.terminated and steps < T_max:
        env.mark_plan_anchor()
        A = be.plan(env.observe(), noise=torch.randn(H, D, generator=gen)).cpu(); plans += 1
        for i in range(REPLAN):
            if env.terminated or steps >= T_max: break
            env.step(A[i]); steps += 1
    return dict(steps=steps, plans=plans, h=[], h_ent=[])


@torch.no_grad()
def run_aac(env, gen, gen_x, T_max):
    steps = plans = 0; hs, hes = [], []
    while not env.terminated and steps < T_max:
        env.mark_plan_anchor()
        noise = torch.cat([torch.randn(1, H, D, generator=gen),
                           torch.randn(args.n - 1, H, D, generator=gen_x)])
        S = plan_samples(env.observe(), noise).float().cpu()
        h, h_ent = aac_decide(S, env._anchor)
        plans += 1; hs.append(h); hes.append(h_ent)
        for i in range(h):
            if env.terminated or steps >= T_max: break
            env.step(S[0, i]); steps += 1
    return dict(steps=steps, plans=plans, h=hs, h_ent=hes)


suite = benchmark.get_benchmark_dict()[args.suite]()
n_tasks = suite.n_tasks if args.tasks == 0 else min(args.tasks, suite.n_tasks)
T_max = MAX_STEPS[args.suite]


def init_states_for(suite, task_id):
    task = suite.get_task(task_id)
    return torch.load(os.path.join(get_libero_path("init_states"), task.problem_folder,
                                   task.init_states_file), weights_only=False)


def _maxdev(a, b):
    return float(torch.linalg.vector_norm(a.float().cpu() - b.float().cpu(), ord=2, dim=-1).max())


_t0 = suite.get_task(0)
_e = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"),
                        _t0.problem_folder, _t0.bddl_file), camera_heights=RES, camera_widths=RES)
_e.seed(args.seed); _e.reset(); _raw = _e.set_init_state(init_states_for(suite, 0)[0])
for _ in range(WAIT): _raw, _, _, _ = _e.step(DUMMY)
_obs = to_observation(_raw, prompt(_t0.language), 0)
_noise = torch.randn(H, D, generator=torch.Generator().manual_seed(3))
_noise_n = torch.cat([_noise[None], torch.randn(args.n - 1, H, D,
                                                generator=torch.Generator().manual_seed(4))])
with torch.no_grad():
    _ref = be.plan(_obs, noise=_noise).float().cpu()
    _ref_n = plan_samples(_obs, _noise_n).float().cpu()
ROW0 = _maxdev(_ref_n[0], _ref)
print(f"  shared-prefix sample 0 vs pi_0's own plan (eager, same noise): max dev {ROW0:.2e}")
COMPILE_MODE = None
if args.compile != "off":
    _eager = model.denoise_step
    for _mode in {"default": ("default", "max-autotune-no-cudagraphs"),
                  "autotune": ("max-autotune", "default")}[args.compile]:
        try:
            torch._dynamo.reset()
            model.denoise_step = torch.compile(_eager, mode=_mode, dynamic=False)
            with torch.no_grad():
                _d = max(_maxdev(be.plan(_obs, noise=_noise), _ref),
                         _maxdev(plan_samples(_obs, _noise_n), _ref_n))
            if _d > 0.05: print(f"  compile[{_mode}]: moved {_d:.4f}, rejected"); continue
            COMPILE_MODE = _mode; print(f"  compile[{_mode}]: preserved ({_d:.2e}, batch 1 and {args.n})"); break
        except Exception as e:
            print(f"  compile[{_mode}]: {type(e).__name__}")
    if COMPILE_MODE is None: model.denoise_step = _eager


def _timed(fn):
    for _ in range(8): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(args.timing_reps): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) * 1e3 / args.timing_reps


_anc = raw_state(_raw)
if args.latency_from:
    _lat = json.load(open(args.latency_from))
    L_PLAN, L_PLAN_AAC = _lat["L_plan"], _lat["L_plan_aac"]
    L_DECIDE = _lat.get("L_decide_cpu", 0.0)
else:
    with torch.no_grad():
        L_PLAN = _timed(lambda: be.plan(_obs, noise=_noise))
        L_PLAN_AAC = _timed(lambda: plan_samples(_obs, _noise_n))
    L_DECIDE = _timed(lambda: aac_decide(_ref_n, _anc))
_e.close()
print(f"\n  per-plan cost ({'stored' if args.latency_from else 'idle GPU'}, {args.timing_reps} reps, compile={COMPILE_MODE or 'OFF'}):")
print(f"    pi_0 plan          {L_PLAN:7.2f} ms")
print(f"    AAC plan (N={args.n:<2})    {L_PLAN_AAC:7.2f} ms   (x{L_PLAN_AAC / L_PLAN:.2f})")
print(f"    AAC decision (CPU) {L_DECIDE:7.2f} ms   measured, not charged")
print(f"    alpha={args.move_th}  window={WINDOW}\n")
COST = {"baseline": L_PLAN, "aac": L_PLAN_AAC}
RUNNER = {"baseline": run_baseline, "aac": run_aac}

results = {}
for config in args.configs:
    print(f"{'='*74}\n{config.upper()}\n{'='*74}")
    succ_total = n_ep = 0; tot = collections.Counter(); per_ep, h_all, he_all = [], [], []
    t0 = time.time()
    for task_id in range(n_tasks):
        task = suite.get_task(task_id); tokens = prompt(task.language)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        inits = init_states_for(suite, task_id); succ = 0
        for trial in range(args.trial_start, args.trial_start + args.trials):
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=RES, camera_widths=RES)
            env.seed(args.seed); env.reset()
            raw = env.set_init_state(inits[trial % len(inits)]); done = False
            for _ in range(WAIT): raw, _, done, _ = env.step(DUMMY)
            w = LiberoEnv(env, tokens, raw); w._done = done
            s0 = args.seed * 1_000_003 + task_id * 10_007 + trial * 101
            gen = torch.Generator().manual_seed(s0)
            gen_x = torch.Generator().manual_seed(s0 + 500_000_009)
            tr = RUNNER[config](w, gen, gen_x, T_max); env.close()
            ok = int(w.terminated); succ += ok; n_ep += 1
            per_ep.append(dict(task=task_id, trial=trial, ok=ok, steps=tr["steps"], plans=tr["plans"]))
            h_all += tr["h"]; he_all += tr["h_ent"]
            for k_ in ("steps", "plans"): tot[k_] += tr[k_]
        succ_total += succ
        print(f"  [{task_id+1}/{n_tasks}] {succ}/{args.trials}  {task.language[:48]}", flush=True)
    per_step = tot["plans"] * COST[config] / max(tot["steps"], 1)
    r = dict(per_episode=per_ep, success=succ_total, episodes=n_ep,
             rate=100.0 * succ_total / max(n_ep, 1), steps=tot["steps"], plans=tot["plans"],
             steps_per_plan=tot["steps"] / max(tot["plans"], 1), ms_per_plan=COST[config],
             ms_per_step=per_step, wall_s=time.time() - t0,
             h_mean=(statistics.mean(h_all) if h_all else None),
             h_median=(statistics.median(h_all) if h_all else None),
             h_hist=(dict(collections.Counter(h_all)) if h_all else None),
             motion_bound_frac=(float(np.mean([h > e for h, e in zip(h_all, he_all)]))
                                if h_all else None))
    results[config] = r
    print(f"\n  success {succ_total}/{n_ep} = {r['rate']:.1f}%   steps/plan {r['steps_per_plan']:.1f}"
          f"   {r['plans']} plans   {per_step:.2f} ms/step   [{r['wall_s']:.0f}s]")
    if h_all:
        print(f"  h*: mean {r['h_mean']:.1f}  median {r['h_median']}   "
              f"motion floor set it on {100*r['motion_bound_frac']:.0f}% of plans")
if "baseline" in results and "aac" in results:
    b, s = results["baseline"], results["aac"]
    print(f"\n  success {b['rate']:.1f}% -> {s['rate']:.1f}%   ms/step {b['ms_per_step']:.2f} -> "
          f"{s['ms_per_step']:.2f}  ({b['ms_per_step']/s['ms_per_step']:.2f}x)")
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    json.dump(dict(suite=args.suite, tasks=n_tasks, trials=args.trials, trial_start=args.trial_start, seed=args.seed,
                   model=args.model, replan=REPLAN, n=args.n, move_th=args.move_th, window=WINDOW,
                   L_plan=L_PLAN, L_plan_aac=L_PLAN_AAC, L_decide_cpu=L_DECIDE,
                   row0_dev=ROW0, compile=COMPILE_MODE, results=results), f, indent=2)
print(f"\nwrote {args.out}")
