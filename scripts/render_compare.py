"""Side-by-side video of the episodes where pi_0 fails and the corrector does not.

Two passes, because which episodes diverge is not known until both have run and
holding every frame of 100 episodes in memory to find out would cost gigabytes:

  1. run both loops over the whole suite, recording only success per episode
  2. re-run just the divergent ones with rendering on

The seeds make this exact rather than approximate.  The noise for the n-th draw
of a given (task, trial) comes from a generator seeded by that triple, so pass 2
reproduces pass 1 frame for frame, and the two loops see identical noise wherever
they make identical choices.  The draw ORDER therefore has to match the driver
exactly -- plan noise first, then K-wide eps per correction -- or the replay
diverges from the run it is supposed to be showing.

Left panel is pi_0 replanning every 10 steps; right is the adaptive corrector.
The instruction is drawn above both, since a viewer cannot tell whether a
trajectory succeeded without knowing what it was asked to do.
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
import statistics
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
from PIL import Image, ImageDraw, ImageFont

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
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--tasks", type=int, default=10)
ap.add_argument("--trials", type=int, default=5)
ap.add_argument("--replan", type=int, default=10)
ap.add_argument("--c", type=int, default=25,
                help="only the initial value of last_seg, which the driver "
                     "leaves at its own default; it is overwritten before the "
                     "first correction reads it, but is kept here so the two "
                     "programs are literally the same program.")
ap.add_argument("--tau0", type=float, default=0.5)
ap.add_argument("--k-draws", type=int, default=4)
ap.add_argument("--agree-tau", type=float, default=1.0)
ap.add_argument("--n-min", type=int, default=5)
ap.add_argument("--c-max", type=int, default=25)
ap.add_argument("--c-safe", type=int, default=10)
ap.add_argument("--max-clips", type=int, default=4)
ap.add_argument("--fps", type=int, default=20)
ap.add_argument("--scale", type=int, default=2)
ap.add_argument("--student", default="",
                help="render the distilled corrector instead of the teacher. "
                     "Corrections run at the student's depth with its adapters "
                     "on; plans and the SREF reference plans stay full depth, "
                     "because a plan is the anchor and SREF is the yardstick.")
ap.add_argument("--net", default="",
                help="render a 56_/54_ network corrector instead of the "
                     "teacher or a LoRA student.")
ap.add_argument("--render", default="wins",
                choices=["wins", "losses", "both"],
                help="which divergent episodes to write out.")
ap.add_argument("--out-dir", default=f"{HOME}/clips")
ap.add_argument("--scan-json", default="",
                help="reuse a previous pass-1 scan instead of re-running it")
args = ap.parse_args()

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
             "libero_10": 520, "libero_90": 400}
DUMMY = [0.0] * 6 + [-1.0]
WAIT, RES, POSE = 10, 256, 6
CONV = f"{HOME}/openpi_assets/pi0_libero_pytorch"
RAW = f"{HOME}/openpi_assets/pi0_libero"
TOK = f"{HOME}/assets/paligemma_tokenizer.model"

os.makedirs(args.out_dir, exist_ok=True)
model = load_pi0_pytorch(CONV, device="cuda")
if args.student:
    _st = torch.load(args.student, weights_only=False)
    S_EV, S_EB = int(_st["e_v"]), int(_st["e_b"])
    be = OpenPiBackend(model, device="cuda", M=10, E_V_max=S_EV, E_B_max=S_EB,
                       n_rungs=1, lora_rank=int(_st["rank"]),
                       lora_rank_readout=int(_st["rank_readout"]),
                       lora_dropout=0.0, attach_adapters=True)
    _res = model.load_state_dict(_st["lora"], strict=False)
    from sentry.models.lora import set_adapter_slot as _sslot
    _sslot(model, 0)
    print(f"  student ({S_EV}, {S_EB}): rel L2 {_st['rel']:.3f}, "
          f"draw spread {_st['spread']:.2f}x")
    if len(_st["lora"]) == 0 or len(_res.unexpected_keys) > 0:
        sys.exit("adapter load failed -- backend built at a different depth "
                 "or rank than the checkpoint")
else:
    S_EV, S_EB = None, None
    be = OpenPiBackend(model, device="cuda", M=10, attach_adapters=False)
model.eval()
M, H, D = be.M, be.H, be.d_a
NET = None
if args.net:
    if args.student:
        sys.exit("--net and --student are two different correctors; pick one")
    from student_net import NetCorrector
    NET = NetCorrector(args.net, model, be, be.H, be.d_a, device="cuda")
    print(f"  {NET.describe()}")
FULL_V, FULL_B = be.L_V, be.L_B

ns = find_norm_stats(RAW)
spec = from_openpi_norm_stats(ns, d_a=D)
sm, ss = state_spec_from_openpi_norm_stats(ns, d_state=8)
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


def agentview(raw):
    return np.ascontiguousarray(raw["agentview_image"][::-1, ::-1])


def to_observation(raw, tokens, t):
    img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(agentview(raw), 224, 224))
    wri = image_tools.convert_to_uint8(image_tools.resize_with_pad(
        np.ascontiguousarray(raw["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224))
    frames = np.stack([img, wri]).astype(np.float32) / 127.5 - 1.0
    st = torch.zeros(D)
    st[:8] = (torch.from_numpy(raw_state(raw)) - sm) / ss
    return Observation(images=torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous(),
                       language=tokens, state=st, t=t)


@torch.no_grad()
def integrate(x0, tau0, n, obs, E_V=None, E_B=None):
    """A call that names no depth is a CORRECTION and takes the student when one
    is loaded.  The reference plans behind SREF pass full depth explicitly: they
    are the yardstick and must not move when the corrector changes."""
    ev = (S_EV if (E_V is None and S_EV is not None)
          else (FULL_V if E_V is None else E_V))
    eb = (S_EB if (E_B is None and S_EB is not None)
          else (FULL_B if E_B is None else E_B))
    use_ad = bool(args.student) and E_V is None
    x = x0.clone().float().to(be.device)
    tau = tau0.clone().float().to(be.device)
    step = (tau / max(n, 1)).view(-1, 1, 1)
    # One memory per correction, shared by the K draws, exactly as trained.
    mem = NET.memory(obs) if (NET is not None and E_V is None) else None
    for _ in range(n):
        if mem is not None:
            v = NET.velocity(x, tau, mem)
        else:
            v = be.velocity(A_tau=x, tau=tau, obs=obs, E_V=ev, E_B=eb,
                            adapters=use_ad)
        x = x - step * v.float()
        tau = tau - step.view(-1)
    return x


def hold_pad(A, H):
    if A.shape[0] >= H:
        return A[:H].clone()
    return torch.cat([A, A[-1:].expand(H - A.shape[0], A.shape[1])], dim=0)


def snap_grip(A):
    g = A[:, G]
    A[:, G] = torch.where((g - G_LO).abs() <= (g - G_HI).abs(),
                          torch.full_like(g, G_LO), torch.full_like(g, G_HI))
    return A


class Env:
    def __init__(self, env, tokens, t0_raw, record=False):
        self._env, self._tokens, self._raw = env, tokens, t0_raw
        self._t, self._done = 0, False
        self._anchor = raw_state(t0_raw)
        self.frames = [agentview(t0_raw)] if record else None
        self.marks = []          # (step, kind) for the caption strip

    def mark_plan_anchor(self):
        self._anchor = raw_state(self._raw)

    def observe(self):
        return to_observation(self._raw, self._tokens, self._t)

    def step(self, action):
        a = action.detach().cpu() * spec.scale + spec.mean
        a[:DELTA_DIMS] += torch.from_numpy(self._anchor[:DELTA_DIMS])
        self._raw, _, done, _ = self._env.step(a[:7].tolist())
        self._t += 1
        self._done = self._done or bool(done)
        if self.frames is not None:
            self.frames.append(agentview(self._raw))

    def frame_correction(self):
        now = raw_state(self._raw)
        d = self._anchor[:DELTA_DIMS] - now[:DELTA_DIMS]
        return torch.from_numpy(d).float() / spec.scale[:DELTA_DIMS].cpu()

    @property
    def terminated(self):
        return self._done


# ==========================================================================
# the two loops -- draw order must match 43_corrector_loop.py exactly
# ==========================================================================
@torch.no_grad()
def run_baseline(env, gen, T_max):
    steps, plans = 0, 0
    while not env.terminated and steps < T_max:
        env.mark_plan_anchor()
        A = be.plan(env.observe(), noise=torch.randn(H, D, generator=gen)).cpu()
        plans += 1
        env.marks.append((steps, "plan"))
        for i in range(args.replan):
            if env.terminated or steps >= T_max:
                break
            env.step(A[i])
            steps += 1
    return dict(steps=steps, plans=plans, corrections=0)


@torch.no_grad()
def run_adaptive(env, gen, T_max, SREF):
    steps = plans = corrections = 0
    A, used, last_seg, just_planned = None, 0, args.c, False
    capped = False
    K = args.k_draws
    while not env.terminated and steps < T_max:
        if A is None:
            env.mark_plan_anchor()
            A = be.plan(env.observe(), noise=torch.randn(H, D, generator=gen)).cpu()
            plans += 1
            used, just_planned = 0, True
            env.marks.append((steps, "plan"))
        else:
            just_planned = False
            obs_now, fc = env.observe(), env.frame_correction()
            A_stale = hold_pad(A[last_seg:], H)
            A_stale[:, :DELTA_DIMS] += fc
            eps = torch.randn(K, H, D, generator=gen)
            x0 = args.tau0 * eps + (1.0 - args.tau0) * A_stale.unsqueeze(0)
            out = integrate(x0, torch.full((K,), args.tau0), 1, obs_now).cpu()

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
            corrections += 1

            if n_agree < args.n_min:
                A = None
                env.marks.append((steps, "fallback"))
                continue
            A = snap_grip(out[0].clone())
            env.mark_plan_anchor()
            env.marks.append((steps, f"correct N={n_agree}"))

        # Exactly the driver's bookkeeping, including the part that looks
        # wrong: the segment is NOT clamped to the chunk's remaining planned
        # content, so a late segment runs past H and into the hold-padding.
        # Clamping it here -- which is what a careful reimplementation does --
        # changed 50/50 into 46/50, because the trajectories separate the first
        # time a segment would have overrun.  A replay that improves on the run
        # it is illustrating is still the wrong replay.
        seg = args.c_safe if just_planned else max(args.n_min,
                                                   min(n_agree, args.c_max))
        capped = used + seg >= H
        for i in range(seg):
            if env.terminated or steps >= T_max:
                break
            env.step(A[i])
            steps += 1
            used += 1
        last_seg = seg
        if capped:
            A = None
    return dict(steps=steps, plans=plans, corrections=corrections)


# ==========================================================================
# setup
# ==========================================================================
suite = benchmark.get_benchmark_dict()[args.suite]()
n_tasks = min(args.tasks, suite.n_tasks)
T_max = MAX_STEPS[args.suite]


def inits_for(tid):
    t = suite.get_task(tid)
    return torch.load(os.path.join(get_libero_path("init_states"),
                                   t.problem_folder, t.init_states_file),
                      weights_only=False)


def make_env(tid, trial, record=False):
    t = suite.get_task(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
    e = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=RES, camera_widths=RES)
    e.seed(args.seed)
    e.reset()
    raw = e.set_init_state(inits_for(tid)[trial % len(inits_for(tid))])
    done = False
    for _ in range(WAIT):
        raw, _, done, _ = e.step(DUMMY)
    w = Env(e, prompt(t.language), raw, record=record)
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
        integrate(torch.randn(args.k_draws, H, D), torch.full((args.k_draws,), 0.5),
                  1, _obs)

# Per-position reference spread -- the scale every acceptance decision is made
# against, so it has to be built by the driver's procedure exactly.
#
# The first attempt advanced the probe env with a normalised zero action through
# the wrapper, which denormalises to spec.mean plus the anchor: a real motion
# command, not a no-op.  The driver steps the RAW env with DUMMY.  The probe
# observations were therefore taken somewhere else entirely, SREF came out
# different, and every accept/reject downstream drifted with it -- 47/50 instead
# of 50/50, from a line that looks like it does nothing.
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
                           torch.ones(8), M, _o,
                           E_V=FULL_V, E_B=FULL_B).cpu()
            _pp = _R[:, :args.c_max, :POSE]
            _perpos.append(torch.stack(
                [torch.linalg.vector_norm(_pp[i] - _pp[j], ord=2, dim=-1)
                 for i in range(8) for j in range(i + 1, 8)]).median(dim=0).values)
        _ev.close()
SREF = torch.stack(_perpos).median(dim=0).values.clamp(min=1e-6)
_e.close()
print(f"  per-position spread h=0 {SREF[0]:.3f}  h={len(SREF)-1} {SREF[-1]:.3f}")


# ==========================================================================
# pass 1 -- who diverges
# ==========================================================================
def episode_gen(tid, trial):
    return torch.Generator().manual_seed(
        args.seed * 1_000_003 + tid * 10_007 + trial * 101)


if args.scan_json and os.path.exists(args.scan_json):
    with open(args.scan_json) as f:
        scan = json.load(f)
    print(f"  reusing scan from {args.scan_json}")
else:
    scan = []
    t0 = time.time()
    for tid in range(n_tasks):
        for trial in range(args.trials):
            row = dict(task=tid, trial=trial,
                       language=suite.get_task(tid).language)
            for name, fn in (("baseline", run_baseline), ("adaptive", None)):
                e, w = make_env(tid, trial)
                g = episode_gen(tid, trial)
                if name == "baseline":
                    run_baseline(w, g, T_max)
                else:
                    run_adaptive(w, g, T_max, SREF)
                row[name] = int(w.terminated)
                e.close()
            scan.append(row)
        print(f"  scanned task {tid+1}/{n_tasks}  {time.time()-t0:.0f}s", flush=True)
    with open(os.path.join(args.out_dir, f"scan_s{args.seed}.json"), "w") as f:
        json.dump(scan, f, indent=2)

wins = [r for r in scan if r["baseline"] == 0 and r["adaptive"] == 1]
loss = [r for r in scan if r["baseline"] == 1 and r["adaptive"] == 0]
print(f"\n  baseline {sum(r['baseline'] for r in scan)}/{len(scan)}   "
      f"adaptive {sum(r['adaptive'] for r in scan)}/{len(scan)}")
print(f"  corrector wins {len(wins)}, loses {len(loss)}")
for r in wins:
    print(f"    WIN  task {r['task']} trial {r['trial']}  {r['language'][:56]}")
for r in loss:
    print(f"    LOSS task {r['task']} trial {r['trial']}  {r['language'][:56]}")


# ==========================================================================
# pass 2 -- render
# ==========================================================================
FONT = None
for cand in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
    if os.path.exists(cand):
        FONT = cand
        break


def font(sz):
    return ImageFont.truetype(FONT, sz) if FONT else ImageFont.load_default()


def compose(fa, fb, lang, ta, tb, ok_a, ok_b, i, n):
    S = args.scale
    w, h = RES * S, RES * S
    head, foot = 74, 40
    canvas = Image.new("RGB", (w * 2 + 12, h + head + foot), (14, 17, 22))
    d = ImageDraw.Draw(canvas)

    d.text((16, 14), lang, font=font(21), fill=(238, 242, 247))
    d.text((16, 46), f"LIBERO-Spatial · seed {args.seed} · step {i}/{n}",
           font=font(15), fill=(132, 146, 160))

    for k, (fr, lab, okv, xo) in enumerate((
            (fa, ta, ok_a, 0), (fb, tb, ok_b, w + 12))):
        im = Image.fromarray(fr).resize((w, h), Image.NEAREST)
        canvas.paste(im, (xo, head))
        col = (79, 184, 194) if okv else (224, 120, 142)
        d.rectangle([xo, head, xo + w - 1, head + h - 1], outline=col, width=3)
        d.text((xo + 10, head + h + 10), lab, font=font(17), fill=col)
        d.text((xo + w - 96, head + h + 10),
               "SUCCESS" if okv else "FAILURE", font=font(17), fill=col)
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


rendered = []
_pick = {"wins": wins, "losses": loss, "both": wins + loss}[args.render]
for r in _pick[:args.max_clips]:
    tid, trial = r["task"], r["trial"]
    out = {}
    for name, fn in (("baseline", run_baseline), ("adaptive", None)):
        e, w = make_env(tid, trial, record=True)
        g = episode_gen(tid, trial)
        if name == "baseline":
            run_baseline(w, g, T_max)
        else:
            run_adaptive(w, g, T_max, SREF)
        out[name] = (w.frames, int(w.terminated), w.marks)
        e.close()

    fa, ok_a, ma = out["baseline"]
    fb, ok_b, mb = out["adaptive"]
    n = max(len(fa), len(fb))
    fa = fa + [fa[-1]] * (n - len(fa))
    fb = fb + [fb[-1]] * (n - len(fb))

    path = os.path.join(args.out_dir,
                        f"s{args.seed}_t{tid}_r{trial}.mp4")
    first = compose(fa[0], fb[0], r["language"], "", "", ok_a, ok_b, 0, n)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                         (first.shape[1], first.shape[0]))
    for i in range(n):
        la = f"π₀  replan=10   ({len([m for m in ma if m[0] <= i])} plans)"
        nb = len([m for m in mb if m[0] <= i and m[1].startswith("plan")])
        nc = len([m for m in mb if m[0] <= i and m[1].startswith("correct")])
        lb = f"adaptive τ=1.0   ({nb} plans, {nc} corrections)"
        vw.write(compose(fa[i], fb[i], r["language"], la, lb, ok_a, ok_b, i, n))
    for _ in range(args.fps):                      # hold the last frame
        vw.write(compose(fa[-1], fb[-1], r["language"], la, lb, ok_a, ok_b, n, n))
    vw.release()
    rendered.append(path)
    print(f"  wrote {path}  ({n} frames)")

print("\n".join(["", "RENDERED:"] + rendered) if rendered
      else "\nno divergent episode to render")
