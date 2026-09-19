"""Where does the time in a plan and in a correction actually go?

Every speedup quoted so far rested on a split of L into encoder, prefill and
denoise that was never measured in this line of work -- it was obtained by
subtracting a batch-4 velocity call from a batch-1 plan and dividing by nine.
That is not a measurement: the two calls run the action expert at different
batch sizes, so their difference is 10*d(b=1) - d(b=4), which is one equation in
two unknowns.  The derived d came out 2.03 ms against the 1.50 ms that 28_'s
own docstring reports from direct instrumentation, a 35% error, and it happened
to fall on the flattering side.

So measure the three stages directly, by timing the three primitives both paths
share:

    pwe.embed_image                 the vision encoder, once per camera
    pwe.forward with no KV cache    the prefix prefill
    model.denoise_step              one velocity evaluation

denoise_step internally calls pwe.forward *with* a cache; that call is excluded
so the prefill is not counted twice.  This is the same instrumentation 28_ uses,
which is what makes the numbers comparable to the ones already in the record.

Three questions, all of which change the ceiling:

  1. What is L_enc, L_pre, L_den at batch 1, on this GPU, compiled?
  2. How much does batch inflate the denoise step?  If d(b=4) >> d(b=1), every
     latency in gate_depth_s7.json is charging the corrector for interpolants a
     deployed loop never evaluates.
  3. How many images does the encoder actually run on?  LIBERO supplies two
     cameras.  If the checkpoint pads to three, a third of the encoder and a
     quarter of the prefill tokens are white pixels, and deleting them is a
     free speedup that needs no research at all.
"""

# --- repository layout -------------------------------------------------
# Every path hangs off one root so the tree can live anywhere.  Set
# CORRECTOR_HOME to override; by default it is the directory holding this
# scripts/ folder, which is what setup/ populates.
import os as _os
HOME = _os.environ.get(
    "CORRECTOR_HOME",
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# -----------------------------------------------------------------------


import argparse
import json
import math
import os
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

import numpy as np
import torch

torch.set_num_threads(4)

# Each batch size the corrector is timed at is a separate compiled graph, and
# so is the planner's.  The default recompile limit is 8; blowing it makes
# dynamo silently fall back to eager PART WAY THROUGH the sweep, which is how
# the first attempt reported one denoise step at 38 ms in the correction row
# and 7 ms in the plan row -- same batch, same op, five times apart.  A wrong
# number that looks plausible is worse than a crash, so raise the limit and
# assert afterwards that the two agree.
torch._dynamo.config.cache_size_limit = 64
if hasattr(torch._dynamo.config, "recompile_limit"):
    torch._dynamo.config.recompile_limit = 64

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools

from sentry.core.types import Observation
from sentry.envs.libero_data import LiberoPrompt, state_spec_from_openpi_norm_stats
from sentry.envs.libero_spec import find_norm_stats, from_openpi_norm_stats
from sentry.models.openpi_adapter import OpenPiBackend, load_pi0_pytorch

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=30)
ap.add_argument("--batches", default="1,2,4,8")
ap.add_argument("--compile", choices=["default", "off"], default="default")
ap.add_argument("--out", default=f"{HOME}/ckpt/stage_latency.json")
args = ap.parse_args()

BATCHES = [int(x) for x in args.batches.split(",")]
DUMMY = [0.0] * 6 + [-1.0]
WAIT, RES = 10, 256
CONV = f"{HOME}/openpi_assets/pi0_libero_pytorch"
RAW = f"{HOME}/openpi_assets/pi0_libero"
TOK = f"{HOME}/assets/paligemma_tokenizer.model"

model = load_pi0_pytorch(CONV, device="cuda")
be = OpenPiBackend(model, device="cuda", M=10, attach_adapters=False)
model.eval()
M, H, D = be.M, be.H, be.d_a

ns = find_norm_stats(RAW)
spec = from_openpi_norm_stats(ns, d_a=D)
sm, ss = state_spec_from_openpi_norm_stats(ns, d_state=8)
prompt = LiberoPrompt(TOK, max_len=be.max_token_len)


def quat2axisangle(q):
    q = np.asarray(q, dtype=np.float64).copy()
    q[3] = min(1.0, max(-1.0, q[3]))
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
    images = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
    st = torch.zeros(D)
    st[:8] = (torch.from_numpy(raw_state(raw)) - sm) / ss
    return Observation(images=images, language=tokens, state=st, t=t)


class StageProfiler:
    """Charge encoder, prefill and denoise separately, wherever called from.

    A plan makes one encode, one prefill and M denoise calls; a correction makes
    one encode, one prefill and ONE denoise call.  Timing the three primitives
    rather than the two entry points is what makes that asymmetry visible
    instead of buried inside a single number.
    """

    def __init__(self, backend, model):
        self.pwe, self.model = backend._pwe, model
        self._embed, self._fwd, self._den = (
            self.pwe.embed_image, self.pwe.forward, model.denoise_step)
        self.reset()

        def embed(*a, **k):
            torch.cuda.synchronize(); t = time.perf_counter()
            out = self._embed(*a, **k)
            torch.cuda.synchronize(); self.enc += time.perf_counter() - t
            self.n_enc += 1
            return out

        def fwd(*a, **k):
            if k.get("past_key_values", None) is not None:
                return self._fwd(*a, **k)          # inside denoise, not a prefill
            torch.cuda.synchronize(); t = time.perf_counter()
            out = self._fwd(*a, **k)
            torch.cuda.synchronize(); self.pre += time.perf_counter() - t
            self.n_pre += 1
            return out

        def den(*a, **k):
            torch.cuda.synchronize(); t = time.perf_counter()
            out = self._den(*a, **k)
            torch.cuda.synchronize(); self.den += time.perf_counter() - t
            self.n_den += 1
            return out

        self.pwe.embed_image, self.pwe.forward, model.denoise_step = embed, fwd, den

    def reset(self):
        self.enc = self.pre = self.den = 0.0
        self.n_enc = self.n_pre = self.n_den = 0


suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
inits = torch.load(os.path.join(get_libero_path("init_states"), task.problem_folder,
                                task.init_states_file), weights_only=False)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=RES, camera_widths=RES)
env.seed(7); env.reset()
raw = env.set_init_state(inits[0])
for _ in range(WAIT):
    raw, _, _, _ = env.step(DUMMY)
obs = to_observation(raw, prompt(task.language), 0)
noise = torch.randn(H, D, generator=torch.Generator().manual_seed(3))
ref = be.plan(obs, noise=noise).float()

COMPILE_MODE = None
if args.compile != "off":
    _eager = model.denoise_step
    for mode in ("default", "max-autotune-no-cudagraphs"):
        try:
            torch._dynamo.reset()
            model.denoise_step = torch.compile(_eager, mode=mode, dynamic=False)
            with torch.no_grad():
                d = float(torch.linalg.vector_norm(
                    be.plan(obs, noise=noise).float() - ref, ord=2, dim=-1).max())
            if d > 0.05:
                print(f"  compile[{mode}]: chunk moved {d:.4f} -- rejected"); continue
            COMPILE_MODE = mode
            print(f"  compile[{mode}]: chunk preserved ({d:.2e})")
            break
        except Exception as e:
            print(f"  compile[{mode}]: {type(e).__name__}: {str(e)[:60]}")
    if COMPILE_MODE is None:
        model.denoise_step = _eager
        print("  eager")

# trace every batch before the profiler is attached, or the first call at each
# shape pays its compilation inside the timed region
with torch.no_grad():
    for K in BATCHES:
        be.velocity(A_tau=torch.randn(K, H, D), tau=torch.full((K,), 0.5),
                    obs=obs, E_V=be.L_V, E_B=be.L_B, adapters=False)
    be.plan(obs, noise=noise)

prof = StageProfiler(be, model)


def run(label, fn, reps):
    prof.reset()
    with torch.no_grad():
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize(); total = (time.perf_counter() - t0) * 1e3 / reps
    e, p, d = prof.enc * 1e3 / reps, prof.pre * 1e3 / reps, prof.den * 1e3 / reps
    return dict(label=label, total=total, enc=e, pre=p, den=d,
                n_enc=prof.n_enc / reps, n_pre=prof.n_pre / reps,
                n_den=prof.n_den / reps, other=total - e - p - d)


rows = [run("full plan (M=10, b=1)", lambda: be.plan(obs, noise=noise), args.reps)]
for K in BATCHES:
    x = torch.randn(K, H, D, generator=torch.Generator().manual_seed(11))
    t = torch.full((K,), 0.5)
    rows.append(run(f"one correction (b={K})",
                    lambda x=x, t=t: be.velocity(A_tau=x, tau=t, obs=obs,
                                                 E_V=be.L_V, E_B=be.L_B,
                                                 adapters=False),
                    args.reps))

print(f"\n{'='*96}\nmeasured directly, compile={COMPILE_MODE}, {args.reps} reps"
      f"\n{'='*96}")
print(f"{'':<26}{'total':>9}{'encoder':>10}{'prefill':>10}{'denoise':>10}"
      f"{'other':>8}   calls (enc/pre/den)")
for r in rows:
    print(f"{r['label']:<26}{r['total']:>9.2f}{r['enc']:>10.2f}{r['pre']:>10.2f}"
          f"{r['den']:>10.2f}{r['other']:>8.2f}   "
          f"{r['n_enc']:.0f}/{r['n_pre']:.0f}/{r['n_den']:.0f}")

plan, corr1 = rows[0], rows[1]

# Cross-check: one denoise step at batch 1 is the same operation whether it was
# reached through the planner or through a single correction.  If these two
# disagree, something recompiled inside a timed region and every number in the
# table is suspect.
d_plan = plan["den"] / max(plan["n_den"], 1)
d_corr = corr1["den"] / max(corr1["n_den"], 1)
skew = max(d_plan, d_corr) / max(min(d_plan, d_corr), 1e-9)
print(f"\nconsistency check: denoise/step at b=1 is {d_plan:.2f} ms via the "
      f"planner and {d_corr:.2f} ms via a correction  ({skew:.2f}x)")
if skew > 1.25:
    sys.exit("those must be the same operation -- a graph recompiled inside a "
             "timed region, so this table cannot be reported")
print(f"\n--- the three questions")
print(f"1. per-stage at batch 1:  L_enc={corr1['enc']:.2f}  L_pre={corr1['pre']:.2f}"
      f"  L_den={plan['den']/max(plan['n_den'],1):.2f} per step")
print(f"   perception share of a CORRECTION: "
      f"{100*(corr1['enc']+corr1['pre'])/corr1['total']:.1f}%")
print(f"   perception share of a PLAN:       "
      f"{100*(plan['enc']+plan['pre'])/plan['total']:.1f}%")
b_last = rows[-1]
print(f"2. denoise at b=1 {corr1['den']:.2f} ms  vs  b={BATCHES[-1]} "
      f"{b_last['den']:.2f} ms  "
      f"({b_last['den']/max(corr1['den'],1e-9):.2f}x)")
print(f"   a correction costs {corr1['total']:.2f} ms at b=1, "
      f"{b_last['total']:.2f} ms at b={BATCHES[-1]}")
print(f"3. encoder ran {corr1['n_enc']:.0f} times for "
      f"{obs.images.shape[0]} real cameras"
      + ("   <- the extra one is a blank image; dropping it saves "
         f"{corr1['enc']/max(corr1['n_enc'],1):.2f} ms of encoder"
         if corr1['n_enc'] > obs.images.shape[0] else ""))

with open(args.out, "w") as f:
    json.dump(dict(compile=COMPILE_MODE, reps=args.reps, rows=rows,
                   n_cameras_real=int(obs.images.shape[0])), f, indent=2)
print(f"\nwrote {args.out}")
env.close()
