"""Harvest (observation, interpolant, teacher velocity) for distilling the corrector.

The corrector is queried at exactly ONE point of the flow -- model time tau0=0.5,
one Euler step, on an interpolant built from the chunk already in hand.  So the
student does not have to learn a velocity field over all of time; it has to
match v_theta on a single slice, on the inputs the deployed loop actually visits.
That makes this plain regression, not consistency distillation: there are no
steps left to remove, since the corrector already spends one.

Where the samples come from matters more than how many there are.  They are
harvested from rollouts of the WORKING mechanism (adaptive, tau=1.0, cap 25), so
the observations are the ones a corrector meets in deployment and the stale
chunks carry the compounding history a synthetic sampler would not reproduce.

Cost trick: the prefix is 35 ms of the 37.9 and is shared across a batch, so one
observation yields K_harvest interpolants for about 5 ms more.  Sixteen draws per
correction turns ~290 corrections per sweep into ~4600 training pairs.

What the student must preserve, and what a plain L2 fit can silently destroy:
the acceptance test reads the DISAGREEMENT between K draws.  Those draws differ
only because their x differs -- v_theta itself is deterministic -- so a student
that matches v pointwise reproduces the spread, while one that is merely smooth
under-disperses and makes the test accept everything.  That is why the eval
checks the N distribution and not just the loss.
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
import io
import json
import math
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, f"{HOME}/openpi/src")
sys.path.insert(0, f"{HOME}/openpi/packages/openpi-client/src")
sys.path.insert(0, f"{HOME}/src")
sys.path.insert(0, f"{HOME}/scripts")

import cv2
import numpy as np
import torch

torch.set_num_threads(4)
torch._dynamo.config.cache_size_limit = 64

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

ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="libero_spatial")
ap.add_argument("--seed", type=int, default=101,
                help="deliberately NOT 7/17/27: the evaluation seeds must stay "
                     "unseen by the student")
ap.add_argument("--tasks", type=int, default=10)
ap.add_argument("--trials", type=int, default=10)
ap.add_argument("--k-harvest", type=int, default=16,
                help="interpolants per observation; they share one prefix, so "
                     "16 cost about 5 ms more than 1")
ap.add_argument("--tau0", type=float, default=0.5)
ap.add_argument("--k-draws", type=int, default=4)
ap.add_argument("--agree-tau", type=float, default=1.0)
ap.add_argument("--n-min", type=int, default=5)
ap.add_argument("--c-max", type=int, default=25)
ap.add_argument("--c-safe", type=int, default=10)
ap.add_argument("--jpeg-q", type=int, default=92)
ap.add_argument("--out", default=f"{HOME}/distill/shard")
ap.add_argument("--net", default="",
                help="DAgger: a student checkpoint that DRIVES the rollout -- its "
                     "K draws decide acceptance and the chunk that is executed -- "
                     "while every stored label is still the teacher's velocity.  "
                     "Empty: the teacher drives, as before.")
ap.add_argument("--trial-start", type=int, default=0,
                help="first LIBERO initial state; runs trial_start .. "
                     "trial_start+trials-1.  The evaluation uses states 0-9, so 10 "
                     "keeps every evaluated scene out of the training set.")
ap.add_argument("--model", choices=["pi0", "pi05"], default="pi0",
                help="the teacher.  pi05 is openpi's pi05_libero: a 10-action chunk, "
                     "quantile normalisation and native LIBERO actions, so nothing "
                     "is re-anchored.")
ap.add_argument("--lineage", type=int, default=0,
                help="steps a plan and its corrections may run before a forced "
                     "replan; 0 is the chunk horizon H.  Harvest under the lineage "
                     "the student will be deployed with (pi05: 50), or it never "
                     "meets the states that rule produces.")
ap.add_argument("--random-scenes", action="store_true",
                help="start every episode from env.reset() under its own scene seed "
                     "instead of LIBERO's fixed initial states.  The standard "
                     "evaluation runs all 50 of those, so this is the only way to "
                     "keep every training scene out of it.")
ap.add_argument("--scene-seed", type=int, default=100000,
                help="--random-scenes: episode (task t, trial k) resets under "
                     "scene_seed + 1000 t + k.")
args = ap.parse_args()

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
             "libero_90": 400, "libero_10": 520}
DUMMY = [0.0] * 6 + [-1.0]
WAIT, RES, POSE = 10, 256, 6
CONV, RAW = {
    "pi0": (f"{HOME}/openpi_assets/pi0_libero_pytorch",
            f"{HOME}/openpi_assets/pi0_libero"),
    "pi05": (f"{HOME}/openpi_assets/pi05_libero_pytorch",
             f"{HOME}/openpi_assets/pi05_libero"),
}[args.model]
PI05 = args.model == "pi05"
# pi0_libero's first six action channels are offsets from the state at plan
# time (openpi's extra delta transform); pi05_libero's are LIBERO's own actions,
# and with zero delta channels the anchor add and frame correction are no-ops.
DELTA = 0 if PI05 else DELTA_DIMS
TOK = f"{HOME}/assets/paligemma_tokenizer.model"

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
model = load_pi0_pytorch(CONV, device="cuda")
be = OpenPiBackend(model, device="cuda", M=10, attach_adapters=False)
model.eval()
M, H, D = be.M, be.H, be.d_a
FULL_V, FULL_B = be.L_V, be.L_B
NET = None
if args.net:
    from student_net import NetCorrector
    NET = NetCorrector(args.net, model, be, H, D, device="cuda")
    print(f"  DAgger: the student drives, the teacher labels -- {NET.describe()}",
          flush=True)

ns = find_norm_stats(RAW)
spec = from_openpi_norm_stats(ns, d_a=D, use_quantiles=PI05)
sm, ss = state_spec_from_openpi_norm_stats(ns, d_state=8, use_quantiles=PI05)
prompt = LiberoPrompt(TOK, max_len=be.max_token_len)
G = spec.grip
G_LO = float((spec.grip_raw_modes[0] - spec.mean[G]) / spec.scale[G])
G_HI = float((spec.grip_raw_modes[1] - spec.mean[G]) / spec.scale[G])
G_MID = 0.5 * (G_LO + G_HI)


def quat2axisangle(q):
    q = np.asarray(q, dtype=np.float64).copy()
    q[3] = min(1.0, max(-1.0, q[3]))
    den = np.sqrt(1.0 - q[3] * q[3])
    return np.zeros(3) if math.isclose(den, 0.0) else (q[:3] * 2 * math.acos(q[3])) / den


def raw_state(d):
    return np.concatenate([d["robot0_eef_pos"], quat2axisangle(d["robot0_eef_quat"]),
                           d["robot0_gripper_qpos"]]).astype(np.float32)


def views(raw):
    a = image_tools.convert_to_uint8(image_tools.resize_with_pad(
        np.ascontiguousarray(raw["agentview_image"][::-1, ::-1]), 224, 224))
    w = image_tools.convert_to_uint8(image_tools.resize_with_pad(
        np.ascontiguousarray(raw["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224))
    return a, w


def to_observation(raw, tokens, t):
    a, w = views(raw)
    frames = np.stack([a, w]).astype(np.float32) / 127.5 - 1.0
    st = torch.zeros(D)
    st[:8] = (torch.from_numpy(raw_state(raw)) - sm) / ss
    return Observation(images=torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous(),
                       language=tokens, state=st, t=t)


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


@torch.no_grad()
def velocity_full(x, tau, obs):
    return be.velocity(A_tau=x, tau=tau, obs=obs, E_V=FULL_V, E_B=FULL_B,
                       adapters=False)


def hold_pad(A, H):
    if A.shape[0] >= H:
        return A[:H].clone()
    return torch.cat([A, A[-1:].expand(H - A.shape[0], A.shape[1])], dim=0)


def snap_grip(A):
    g = A[:, G]
    A[:, G] = torch.where((g - G_LO).abs() <= (g - G_HI).abs(),
                          torch.full_like(g, G_LO), torch.full_like(g, G_HI))
    return A


def jpg(img):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_q])
    assert ok
    return buf.tobytes()


class Env:
    def __init__(self, env, tokens, t0_raw):
        self._env, self._tokens, self._raw = env, tokens, t0_raw
        self._t, self._done = 0, False
        self._anchor = raw_state(t0_raw)

    def mark_plan_anchor(self):
        self._anchor = raw_state(self._raw)

    def observe(self):
        return to_observation(self._raw, self._tokens, self._t)

    def raw(self):
        return self._raw

    def step(self, action):
        a = action.detach().cpu() * spec.scale + spec.mean
        a[:DELTA] += torch.from_numpy(self._anchor[:DELTA])
        self._raw, _, done, _ = self._env.step(a[:7].tolist())
        self._t += 1
        self._done = self._done or bool(done)

    def frame_correction(self):
        now = raw_state(self._raw)
        d = self._anchor[:DELTA] - now[:DELTA]
        return torch.from_numpy(d).float() / spec.scale[:DELTA].cpu()

    @property
    def terminated(self):
        return self._done


suite = benchmark.get_benchmark_dict()[args.suite]()
n_tasks = min(args.tasks, suite.n_tasks)
T_max = MAX_STEPS[args.suite]


def inits_for(tid):
    t = suite.get_task(tid)
    return torch.load(os.path.join(get_libero_path("init_states"),
                                   t.problem_folder, t.init_states_file),
                      weights_only=False)


def make_env(tid, trial):
    t = suite.get_task(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
    e = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=RES, camera_widths=RES)
    if args.random_scenes:
        # A scene no evaluation runs: the task's own placement sampler under
        # this episode's seed (same seed, same scene -- checked before use).
        e.seed(args.scene_seed + 1000 * tid + trial)
        raw = e.reset()
    else:
        e.seed(args.seed)
        e.reset()
        ini = inits_for(tid)
        raw = e.set_init_state(ini[trial % len(ini)])
    done = False
    for _ in range(WAIT):
        raw, _, done, _ = e.step(DUMMY)
    w = Env(e, prompt(t.language), raw)
    w._done = done
    return e, w


_e, _w = make_env(0, 0)
_obs = _w.observe()
_noise = torch.randn(H, D, generator=torch.Generator().manual_seed(3))
_ref = be.plan(_obs, noise=_noise).float()
_eager = model.denoise_step
torch._dynamo.reset()
model.denoise_step = torch.compile(_eager, mode="default", dynamic=False)
with torch.no_grad():
    _d = float(torch.linalg.vector_norm(
        be.plan(_obs, noise=_noise).float() - _ref, ord=2, dim=-1).max())
if _d > 0.05:
    model.denoise_step = _eager
    print(f"  compile rejected ({_d:.3f}); eager")
else:
    print(f"  compile ok ({_d:.1e})")
    with torch.no_grad():
        for _K in sorted({args.k_draws, args.k_harvest}):
            integrate(torch.randn(_K, H, D), torch.full((_K,), args.tau0), 1, _obs)

_perpos = []
with torch.no_grad():
    for _tid in range(min(4, n_tasks)):
        _t = suite.get_task(_tid)
        _bd = os.path.join(get_libero_path("bddl_files"),
                           _t.problem_folder, _t.bddl_file)
        _ev = OffScreenRenderEnv(bddl_file_name=_bd, camera_heights=RES,
                                 camera_widths=RES)
        _ev.seed(args.seed)
        _ev.reset()
        _r = _ev.set_init_state(inits_for(_tid)[0])
        for _ in range(WAIT):
            _r, _, _, _ = _ev.step(DUMMY)
        for _off in (0, 25):
            for _ in range(_off):
                _r, _, _, _ = _ev.step(DUMMY)
            _o = to_observation(_r, prompt(_t.language), 0)
            _R = integrate(torch.randn(8, H, D,
                                       generator=torch.Generator().manual_seed(9)),
                           torch.ones(8), M, _o).cpu()
            _pp = _R[:, :args.c_max, :POSE]
            _perpos.append(torch.stack(
                [torch.linalg.vector_norm(_pp[i] - _pp[j], ord=2, dim=-1)
                 for i in range(8) for j in range(i + 1, 8)]).median(dim=0).values)
        _ev.close()
SREF = torch.stack(_perpos).median(dim=0).values.clamp(min=1e-6)
_e.close()
print(f"  per-position spread h=0 {SREF[0]:.3f}  h={len(SREF)-1} {SREF[-1]:.3f}")

# ==========================================================================
# harvest
# ==========================================================================
buf = dict(jpg_a=[], jpg_w=[], state=[], lang=[], x=[], v=[], tau=[], meta=[])
n_pairs = 0
t0 = time.time()

for tid in range(n_tasks):
    for trial in range(args.trial_start, args.trial_start + args.trials):
        e, w = make_env(tid, trial)
        gen = torch.Generator().manual_seed(
            args.seed * 1_000_003 + tid * 10_007 + trial * 101)
        A, used, last_seg, just_planned, steps = None, 0, args.c_max, False, 0
        K = args.k_draws
        while not w.terminated and steps < T_max:
            if A is None:
                w.mark_plan_anchor()
                A = be.plan(w.observe(), noise=torch.randn(H, D, generator=gen)).cpu()
                used, just_planned = 0, True
            else:
                just_planned = False
                obs_now, fc = w.observe(), w.frame_correction()
                # a segment can use the whole chunk once corrections outlive the
                # horizon (lineage > H); the stale guess is then the last action held
                rest = A[last_seg:]
                A_stale = hold_pad(rest if rest.shape[0] else A[-1:], H)
                A_stale[:, :DELTA] += fc

                # -- the training pairs: many interpolants, one prefix
                eh = torch.randn(args.k_harvest, H, D, generator=gen)
                xh = args.tau0 * eh + (1.0 - args.tau0) * A_stale.unsqueeze(0)
                tauh = torch.full((args.k_harvest,), args.tau0)
                vh = velocity_full(xh, tauh, obs_now).cpu()

                a_img, w_img = views(w.raw())
                buf["jpg_a"].append(jpg(a_img))
                buf["jpg_w"].append(jpg(w_img))
                buf["state"].append(obs_now.state.clone().half())
                buf["lang"].append(obs_now.language.clone().to(torch.int32))
                buf["x"].append(xh.half())
                buf["v"].append(vh.half())
                buf["tau"].append(tauh.half())
                buf["meta"].append((tid, trial, steps))
                n_pairs += args.k_harvest

                # -- the deployed mechanism decides what happens next; the first
                #    k_draws interpolants double as its draws, so no extra cost.
                #    Under --net the student makes that decision, so the states and
                #    stale chunks stored from here on are the ones it produces.
                if NET is not None:
                    vk = NET.velocity(xh[:K], tauh[:K], NET.memory(obs_now)).float()
                else:
                    vk = vh[:K].float().to(be.device)
                out = xh[:K].to(be.device) - args.tau0 * vk
                out = out.cpu()
                pose = out[:, :args.c_max, :POSE]
                spread = torch.stack(
                    [torch.linalg.vector_norm(pose[i] - pose[j], ord=2, dim=-1)
                     for i in range(K) for j in range(i + 1, K)]).median(dim=0).values
                side = out[:, :args.c_max, G] > G_MID
                ok = ((spread <= args.agree_tau * SREF[:args.c_max])
                      & (side.all(dim=0) | (~side).all(dim=0))).tolist()
                n_agree = 0
                for good in ok:
                    if not good:
                        break
                    n_agree += 1
                if n_agree < args.n_min:
                    A = None
                    continue
                A = snap_grip(out[0].clone())
                w.mark_plan_anchor()

            seg = args.c_safe if just_planned else max(args.n_min,
                                                       min(n_agree, args.c_max))
            capped = used + seg >= (args.lineage or H)
            for i in range(seg):
                if w.terminated or steps >= T_max:
                    break
                w.step(A[i])
                steps += 1
                used += 1
            last_seg = seg
            if capped:
                A = None
        e.close()
    print(f"  task {tid+1}/{n_tasks}  {n_pairs} pairs  "
          f"{time.time()-t0:.0f}s", flush=True)

path = f"{args.out}_s{args.seed}.pt"
torch.save(dict(
    jpg_a=buf["jpg_a"], jpg_w=buf["jpg_w"],
    state=torch.stack(buf["state"]), lang=torch.stack(buf["lang"]),
    x=torch.stack(buf["x"]), v=torch.stack(buf["v"]), tau=torch.stack(buf["tau"]),
    meta=buf["meta"], tau0=args.tau0, k_harvest=args.k_harvest,
    sref=SREF.cpu(), seed=args.seed, net=args.net, trial_start=args.trial_start,
    c_max=args.c_max, model=args.model, lineage=(args.lineage or H),
    random_scenes=args.random_scenes, scene_seed=args.scene_seed,
), path)
mb = os.path.getsize(path) / 1e6
print(f"\nwrote {path}  {len(buf['x'])} observations x {args.k_harvest} "
      f"= {n_pairs} pairs  {mb:.0f} MB  [{time.time()-t0:.0f}s]")
