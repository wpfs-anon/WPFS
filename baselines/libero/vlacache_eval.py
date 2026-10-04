import os as _os
HOME = _os.environ.get(
    "CORRECTOR_HOME",
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import collections
import json
import math
import os
import statistics
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, f"{HOME}/openpi/src")
sys.path.insert(0, f"{HOME}/openpi/packages/openpi-client/src")
sys.path.insert(0, f"{HOME}/src")
sys.path.insert(0, f"{HOME}/scripts")
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

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

from vlacache_pi0 import VLACacheConfig, VLACacheState, VLACachePi0

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
ap.add_argument("--configs", nargs="+", default=["vlacache"],
                choices=["baseline", "vlacache"])
ap.add_argument("--model", choices=["pi0", "pi05"], default="pi0",
                help="base policy.  pi05 is openpi's pi05_libero: a 10-action "
                     "chunk, quantile normalisation and LIBERO's own actions "
                     "(no delta anchor), run at replan 5 as its own table does")
ap.add_argument("--replan", type=int, default=0,
                help="actions executed per plan; 0 takes the model's own "
                     "cadence -- 10 for pi_0, 5 for pi0.5")
ap.add_argument("--sim-threshold", type=float, default=0.996,
                help="a patch is static above this cosine similarity to the "
                     "previous plan's frame (paper default)")
ap.add_argument("--static-top-k", type=int, default=150,
                help="most-stable static patches kept per view, of 256")
ap.add_argument("--task-top-k", type=int, default=100,
                help="task-relevant patches EVICTED from the reusable set")
ap.add_argument("--attn-layer", type=int, default=8,
                help="backbone layer the text-to-vision score is read at; "
                     "layer 15 of OpenVLA-OFT's 32 is layer 8 of pi_0's 18")
ap.add_argument("--growth-factor", type=float, default=0.55)
ap.add_argument("--max-age", type=int, default=0,
                help="force a token's K/V to be recomputed after this many "
                     "plans; 0 reproduces the reference, which carries cached "
                     "activations forward without bound")
ap.add_argument("--timing-reps", type=int, default=40)
ap.add_argument("--latency-from", default="",
                help="read the cost model from a run measured on an IDLE GPU "
                     "instead of timing here.  Rollouts are unaffected by a "
                     "busy GPU -- they are deterministic given the seed -- but "
                     "timings are not, so this is what makes it sound to run "
                     "several evaluations at once")
ap.add_argument("--compile", choices=["default", "autotune", "off"], default="default")
ap.add_argument("--out", default=f"{HOME}/ckpt/vlacache/run.json")
args = ap.parse_args()

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
             "libero_10": 520, "libero_90": 400}
DUMMY = [0.0] * 6 + [-1.0]
WAIT = 10
RES = 256
CONV, RAW = {"pi0": (f"{HOME}/openpi_assets/pi0_libero_pytorch",
                     f"{HOME}/openpi_assets/pi0_libero"),
             "pi05": (f"{HOME}/openpi_assets/pi05_libero_pytorch",
                      f"{HOME}/openpi_assets/pi05_libero")}[args.model]
TOK = f"{HOME}/assets/paligemma_tokenizer.model"
PI05 = args.model == "pi05"
DELTA = 0 if PI05 else DELTA_DIMS
REPLAN = args.replan or (5 if PI05 else 10)
POSE = 6

model = load_pi0_pytorch(CONV, device="cuda")
model.eval()
be = OpenPiBackend(model, device="cuda", M=10, attach_adapters=False)
M, H, D = be.M, be.H, be.d_a

ns = find_norm_stats(RAW)
spec = from_openpi_norm_stats(ns, d_a=D, use_quantiles=PI05)
sm, ss = state_spec_from_openpi_norm_stats(ns, d_state=8, use_quantiles=PI05)
prompt = LiberoPrompt(TOK, max_len=be.max_token_len)

CFG = VLACacheConfig(sim_threshold=args.sim_threshold,
                     static_top_k=args.static_top_k,
                     task_top_k=args.task_top_k,
                     attn_layer=args.attn_layer,
                     growth_factor=args.growth_factor,
                     max_age=args.max_age)


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
        a[:DELTA] += torch.from_numpy(self._anchor[:DELTA])
        self._raw, _, done, _ = self._env.step(a[:7].tolist())
        self._t += 1
        self._done = self._done or bool(done)

    @property
    def terminated(self):
        return self._done


@torch.no_grad()
def run_baseline(env, gen, T_max, vc=None):
    steps = plans = 0
    while not env.terminated and steps < T_max:
        env.mark_plan_anchor()
        noise = torch.randn(H, D, generator=gen)
        A = be.plan(env.observe(), noise=noise).cpu()
        plans += 1
        for i in range(REPLAN):
            if env.terminated or steps >= T_max:
                break
            env.step(A[i])
            steps += 1
    return dict(steps=steps, plans=plans, kept=[], precise=[])


@torch.no_grad()
def run_vlacache(env, gen, T_max, vc):
    steps = plans = 0
    st = VLACacheState()
    while not env.terminated and steps < T_max:
        env.mark_plan_anchor()
        noise = torch.randn(H, D, generator=gen)
        A = vc.plan(env.observe(), st, noise=noise).cpu()
        plans += 1
        for i in range(REPLAN):
            if env.terminated or steps >= T_max:
                break
            env.step(A[i])
            steps += 1
    return dict(steps=steps, plans=plans,
                kept=[100.0 * r for r in st.reuse_hist], precise=[])


suite = benchmark.get_benchmark_dict()[args.suite]()
n_tasks = suite.n_tasks if args.tasks == 0 else min(args.tasks, suite.n_tasks)
T_max = MAX_STEPS[args.suite]


def init_states_for(suite, task_id):
    task = suite.get_task(task_id)
    return torch.load(os.path.join(get_libero_path("init_states"),
                                   task.problem_folder, task.init_states_file),
                      weights_only=False)


_task0 = suite.get_task(0)
_bddl = os.path.join(get_libero_path("bddl_files"), _task0.problem_folder,
                     _task0.bddl_file)
_e = OffScreenRenderEnv(bddl_file_name=_bddl, camera_heights=RES, camera_widths=RES)
_e.seed(args.seed)
_e.reset()
_raw = _e.set_init_state(init_states_for(suite, 0)[0])
for _ in range(WAIT):
    _raw, _, _, _ = _e.step(DUMMY)
_obs = to_observation(_raw, prompt(_task0.language), 0)
_noise = torch.randn(H, D, generator=torch.Generator().manual_seed(3))
_ref = be.plan(_obs, noise=_noise).float()

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

SP = VLACachePi0(be, CFG)
if COMPILE_MODE is not None:
    SP = VLACachePi0(be, CFG, compiled_denoise=torch.compile(
        VLACachePi0(be, CFG)._denoise_step, mode=COMPILE_MODE, dynamic=False))

_st = VLACacheState()
with torch.no_grad():
    for _ in range(8):
        SP.plan(_obs, _st, noise=_noise)


def _timed(fn):
    for _ in range(8):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.timing_reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / args.timing_reps


if args.latency_from:
    _lat = json.load(open(args.latency_from))
    L_PLAN, L_PLAN_SP = _lat["L_plan"], _lat["L_plan_vlacache"]
else:
    with torch.no_grad():
        L_PLAN = _timed(lambda: be.plan(_obs, noise=_noise))
        L_PLAN_SP = _timed(lambda: SP.plan(_obs, _st, noise=_noise))
_e.close()

print(f"\n  cost model ({'stored' if args.latency_from else 'idle GPU'}, {args.timing_reps} reps, "
      f"compile={COMPILE_MODE or 'OFF'}):")
print(f"    pi_0 plan       {L_PLAN:7.2f} ms")
print(f"    VLACache plan  {L_PLAN_SP:7.2f} ms   ->  {L_PLAN/L_PLAN_SP:.2f}x")
print(f"    tau={CFG.sim_threshold}  static_top_k={CFG.static_top_k}  "
      f"task_top_k={CFG.task_top_k}  attn_layer={CFG.attn_layer}  "
      f"max_age={CFG.max_age or 'unbounded'}\n")

COST = {"baseline": L_PLAN, "vlacache": L_PLAN_SP}
RUNNER = {"baseline": run_baseline, "vlacache": run_vlacache}

results = {}
for config in args.configs:
    print(f"{'='*74}\n{config.upper()}   replan={REPLAN}"
          + (f"   tau={CFG.sim_threshold}" if config == "vlacache" else "")
          + f"\n{'='*74}")
    succ_total = n_ep = 0
    tot = collections.Counter()
    per_ep, kept_all, precise_all = [], [], []
    t0 = time.time()

    for task_id in range(n_tasks):
        task = suite.get_task(task_id)
        tokens = prompt(task.language)
        bddl = os.path.join(get_libero_path("bddl_files"),
                            task.problem_folder, task.bddl_file)
        inits = init_states_for(suite, task_id)
        succ = 0

        for trial in range(args.trial_start, args.trial_start + args.trials):
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
                args.seed * 1_000_003 + task_id * 10_007 + trial * 101)

            tr = RUNNER[config](w, gen, T_max, SP)
            env.close()

            ok = int(w.terminated)
            succ += ok
            n_ep += 1
            per_ep.append(dict(task=task_id, trial=trial, ok=ok))
            kept_all.extend(tr["kept"])
            precise_all.extend(tr["precise"])
            for k in ("steps", "plans"):
                tot[k] += tr[k]

        succ_total += succ
        print(f"  [{task_id+1}/{n_tasks}] {succ}/{args.trials}  "
              f"{task.language[:48]}", flush=True)

    ms = tot["plans"] * COST[config]
    per_step = ms / max(tot["steps"], 1)
    r = dict(per_episode=per_ep, success=succ_total, episodes=n_ep,
             rate=100.0 * succ_total / max(n_ep, 1),
             steps=tot["steps"], plans=tot["plans"],
             steps_per_plan=tot["steps"] / max(tot["plans"], 1),
             ms_per_plan=COST[config], ms_per_step=per_step,
             wall_s=time.time() - t0,
             visual_kept_mean=(statistics.mean(kept_all) if kept_all else None),
             visual_kept_min=(min(kept_all) if kept_all else None),
             visual_kept_max=(max(kept_all) if kept_all else None),
             precise_frac=None)
    results[config] = r
    print(f"\n  success {succ_total}/{n_ep} = {r['rate']:.1f}%"
          f"   steps/plan {r['steps_per_plan']:.1f}"
          f"   {r['plans']} plans"
          f"   {per_step:.2f} ms/step   [{r['wall_s']:.0f}s]")
    if kept_all:
        print(f"  tokens reused: mean {r['visual_kept_mean']:.1f}% of the prefix "
              f"[{r['visual_kept_min']:.0f}, {r['visual_kept_max']:.0f}]")

if "baseline" in results and "vlacache" in results:
    b, s = results["baseline"], results["vlacache"]
    print(f"\n  success   {b['rate']:.1f}%  ->  {s['rate']:.1f}%")
    print(f"  ms/step   {b['ms_per_step']:.2f}  ->  {s['ms_per_step']:.2f}"
          f"   ({b['ms_per_step']/s['ms_per_step']:.2f}x)")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    json.dump(dict(suite=args.suite, tasks=n_tasks, trials=args.trials, trial_start=args.trial_start,
                   seed=args.seed, model=args.model, replan=REPLAN,
                   sim_threshold=CFG.sim_threshold,
                   static_top_k=CFG.static_top_k, task_top_k=CFG.task_top_k,
                   attn_layer=CFG.attn_layer, max_age=CFG.max_age,
                   L_plan=L_PLAN, L_plan_vlacache=L_PLAN_SP,
                   plan_speedup=L_PLAN / L_PLAN_SP,
                   compile=COMPILE_MODE, results=results), f, indent=2)
print(f"\nwrote {args.out}")
