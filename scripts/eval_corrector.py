"""The corrector loop, end to end, against pi_0 replanning -- success, not geometry.

Three gates got the direction this far, and all three measured the same proxy:
whether a corrected chunk lands inside the distribution of plans the target
would draw right now.  That is a proxy.  This runs the loop:

    BASELINE    plan at full depth, execute N actions, plan again.
    CORRECTOR   plan at full depth ONCE, then forever: take the live suffix,
                re-noise it to model-time tau0, run ONE velocity evaluation
                under the CURRENT observation, execute c actions of the result,
                repeat.

Nothing is trained.  The corrector is the target's own flow field, queried once
instead of ten times, from the previous chunk instead of from noise.  If this
holds success, the first contribution of the paper needs no new weights at all.

Two things the probe gates could not see, and this one does:

  * COMPOUNDING.  Every probe corrected a chunk the TARGET had planned.  Here
    correction n+1 acts on the output of correction n, so whatever error the
    operator introduces is fed back into itself.  This is the failure mode most
    likely to kill the direction, and it cannot be measured any other way.
  * WHETHER GEOMETRY IS THE RIGHT PROXY.  A chunk can sit inside the plan
    distribution and still not accomplish the task.

Latency is NOT timed here.  The GPU is shared with another job, and a contended
clock produces a speedup that is an artefact of scheduling.  Instead the loop
counts the operations it performs -- full plans and single-evaluation
corrections -- and the report multiplies those counts by per-call costs measured
on an idle GPU (gate_depth_s7.json).  Counts are exact and contention-free; the
cost table is the one every earlier number in this line of work already used.

Conventions that cost previous sessions real time:

  * openpi integrates model-time 1 -> 0 with x = t*noise + (1-t)*clean, so t=1
    is pure NOISE and t=0 is clean, and one Euler step is x <- x - dt*v.
  * pi0_libero writes the six pose channels as (action - state) against ONE
    anchor per chunk.  A correction produces a chunk conditioned on the CURRENT
    observation, so its deltas are relative to the current state: the anchor
    must be re-marked at every correction, exactly as it is at every plan.
    Forgetting that would leave the robot executing deltas against a state it
    left twenty steps ago.
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

ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="libero_spatial")
ap.add_argument("--tasks", type=int, default=0)
ap.add_argument("--trial-ids", default="",
                help="comma list of 0-based trials to run instead of range(--trials); episodes are seeded by (seed, task, trial), so a subset reproduces those episodes of the full run exactly")
ap.add_argument("--video-dir", default="",
                help="write an agentview mp4 per episode, each frame tagged with where its action came from (orange = a plan, green = a correction)")
ap.add_argument("--task-ids", default="",
                help="comma list of 1-based task ids to run instead of the first --tasks; each episode is seeded by (seed, task, trial), so a subset reproduces the full run")
ap.add_argument("--trials", type=int, default=10)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--configs", nargs="+", default=["baseline", "corrector"],
                choices=["baseline", "corrector"])
ap.add_argument("--replan", type=int, default=10,
                help="baseline: actions executed per full-depth plan")
ap.add_argument("--c", type=int, default=25,
                help="corrector: actions executed per correction.  The horizon "
                     "gate validated the diagonal k = m = c out to 25.")
ap.add_argument("--tau0", type=float, default=0.5,
                help="model-time the live suffix is re-noised to.  0.5 was the "
                     "only setting usable at every c on the diagonal; 0.3 "
                     "stays stale and 0.7 collapses to the conditional mean.")
ap.add_argument("--max-corrections", type=int, default=0,
                help="force a full replan after this many consecutive "
                     "corrections; 0 disables the cap.")
ap.add_argument("--k-draws", type=int, default=1,
                help="noise draws the correction runs at once.  They share the "
                     "observation, so the encoder and the prefill are paid once "
                     "and only the action expert widens: K=4 measured 39.28 ms "
                     "against 37.90 at K=1, about +4%%.  Their DISAGREEMENT is "
                     "the fallback signal this design has been missing -- "
                     "displacement measures how far the plan moved, which is "
                     "large even when nothing is wrong, whereas disagreement "
                     "measures how sure the flow is.")
ap.add_argument("--k-reduce", choices=["first", "median"], default="first",
                help="which of the K draws to execute.  'first' keeps a genuine "
                     "sample; 'median' averages them, which on a multi-modal "
                     "policy is the mode-averaging failure the whole design "
                     "exists to avoid -- offered as an ablation, not a default.")
ap.add_argument("--k-fallback", type=float, default=0.0,
                help="replan when the K draws disagree by more than this many "
                     "multiples of the natural plan-to-plan spread; 0 disables.")
ap.add_argument("--adaptive", action="store_true",
                help="decide HOW MANY actions to execute from the K draws, "
                     "instead of accepting or rejecting the whole chunk.\n"
                     "The draws already disagree per position; the longest "
                     "prefix they agree on is how far the flow is confident, "
                     "and it costs nothing extra to read.  This makes the "
                     "execution interval adaptive, which is the property the "
                     "fixed-c sweeps have been standing in for.")
ap.add_argument("--random-scenes", action="store_true",
                help="run episodes from env.reset() under their own scene seed "
                     "instead of LIBERO's 50 fixed initial states.  Every reported "
                     "number uses those 50, so this is how a checkpoint gets picked "
                     "without tuning on them.")
ap.add_argument("--scene-seed", type=int, default=900000,
                help="--random-scenes: episode (task t, trial k) resets under "
                     "scene_seed + 1000 t + k.")
ap.add_argument("--cascade-min", type=int, default=0,
                help="--cascade: ask the teacher only when the student draws "
                     "agreed on at least this many leading positions.  At 0 every "
                     "failed certificate goes to the teacher and 83% of those "
                     "calls were wasted; raising it spends them on the borderline "
                     "cases only.")
ap.add_argument("--cascade", action="store_true",
                help="with --net: a student correction that fails the certificate "
                     "is retried once by the teacher instead of going straight to "
                     "a replan.  The teacher correction costs L_CORR (43.3 ms on "
                     "pi05) against L_PLAN (63.0 ms), so this is cheaper than the "
                     "fallback it replaces, and only if the teacher's own draws "
                     "disagree does the plan get thrown away.")
ap.add_argument("--agree-tau", type=float, default=1.0,
                help="a position is accepted when the draws' spread there is "
                     "within this multiple of the spread two honest plans "
                     "already show at the same position.")
ap.add_argument("--n-min", type=int, default=5,
                help="if the agreed prefix is shorter than this, fall back.")
ap.add_argument("--c-max", type=int, default=25,
                help="cap on the adaptive interval.")
ap.add_argument("--c-safe", type=int, default=10,
                help="actions executed after a fallback plan.  Falling back "
                     "must return to a SAFE schedule: at c=25 a fallback that "
                     "keeps the interval simply becomes pi_0 replanning every "
                     "25 steps, which is worse than the corrector it replaced "
                     "-- so the earlier threshold sweeps could not have shown a "
                     "gain even with a perfect detector.")
ap.add_argument("--blind-fallback", type=float, default=0.0,
                help="replan with this probability at each correction, ignoring "
                     "every signal.  The control the disagreement threshold has "
                     "to beat: if replanning at random with the same frequency "
                     "does as well, the statistic contributes nothing and what "
                     "helps is simply planning more often.  Without this, any "
                     "gain from a threshold is unattributable.")
ap.add_argument("--steps", type=int, default=1,
                help="Euler steps the correction spends.  One is the whole "
                     "premise, but the denoise stage is only 2.27 ms of a "
                     "37.9 ms correction -- the other 35 are the encoder and "
                     "the prefill, which a second step does not repeat -- so "
                     "each extra step costs about 6%%, not 100%%.  If accuracy "
                     "improves at all, that is a cheap trade.")
ap.add_argument("--drain", action="store_true",
                help="on the last cycle before a capped replan, execute the "
                     "chunk's whole remaining planned content instead of just "
                     "c actions.\n"
                     "At c=15 the cap fires after three segments of 15, which "
                     "leaves 5 of the 50 planned actions unused.  Spending them "
                     "as a fourth cycle would cost a whole extra correction for "
                     "five steps and is worse than wasting them (3.53 ms/step "
                     "against 3.04); folding them into the third segment costs "
                     "nothing and buys 50 steps for the same two corrections "
                     "(2.74).  Only helps when c does not divide H -- c=10 and "
                     "c=25 already consume the chunk exactly.")
ap.add_argument("--auto-cap", action="store_true",
                help="replan once the chunk has no PLANNED content left.\n"
                     "A correction is fed hold_pad(A[c:], H): H-c real entries "
                     "and c of filler.  The real content therefore shrinks by c "
                     "every cycle and reaches zero after floor(H/c) "
                     "corrections, from which point every executed action "
                     "descends from padding this code invented -- a regime no "
                     "gate validated, because the horizon gate deliberately "
                     "skipped every (k,m) pair with m > H-k.  At c=25 that is "
                     "correction 2; at c=10, correction 5.  This replans one "
                     "cycle before it happens.")
ap.add_argument("--fallback", type=float, default=0.0,
                help="replan when the correction moves the chunk further than "
                     "this many multiples of the natural plan-to-plan spread. "
                     "0 disables.  The displacement is free -- it is the "
                     "operator's own output -- so this is the cheap fallback "
                     "the design has been assuming exists.")
ap.add_argument("--grip", choices=["correct", "hold", "snap"], default="correct",
                help="what the correction is allowed to do to the gripper "
                     "channel.  correct: whatever the flow returns, the current "
                     "behaviour.  hold: keep the stale plan's command and "
                     "correct only the pose.  snap: keep the sign the flow "
                     "chose but restore the magnitude to a real mode.  Every "
                     "geometry gate so far scored the six pose channels and "
                     "excluded this one, so nothing measured to date says "
                     "whether the corrector grips at the right moment.")
ap.add_argument("--net", default="",
                help="checkpoint from 54_train_net.py.  Corrections then run on "
                     "the purpose-built network instead of pi0; PLANS stay full "
                     "pi0 because a plan is the anchor every correction is "
                     "measured against.")
ap.add_argument("--net-latency", type=float, default=7.57,
                help="ms for one --net correction at k_draws=4, measured the "
                     "same way L_CORR was (idle GPU, batch 4, eager).")
ap.add_argument("--student", default="",
                help="checkpoint from 49_train_student.py.  Corrections then run "
                     "at the student's depth with its adapters on; PLANS stay at "
                     "full depth with adapters off, because a plan is the anchor "
                     "every later correction is measured against and has to stay "
                     "the pretrained policy exactly.")
ap.add_argument("--latency-from", default=f"{HOME}/ckpt/gate_depth_s7.json")
ap.add_argument("--compile", choices=["default", "autotune", "off"],
                default="default")
ap.add_argument("--out", default=f"{HOME}/ckpt/corrector_loop.json")
ap.add_argument("--model", choices=["pi0", "pi05", "lerobot05"], default="pi0",
                help="base policy.  pi05 is openpi's pi05_libero: a 10-action "
                     "chunk, quantile normalisation, and native LIBERO actions "
                     "(no extra delta transform), so nothing is re-anchored.  "
                     "lerobot05 is lerobot/pi05_libero_finetuned: the same "
                     "network with a 50-action chunk, MEAN_STD statistics and "
                     "the discretised state inside the prompt.  Pass each its "
                     "own --latency-from: pi0's figures do not apply.")
ap.add_argument("--lineage", type=int, default=0,
                help="adaptive mode: steps a plan and its corrections may run "
                     "before a forced replan.  0 is the chunk horizon H, the "
                     "rule every pi0 result used; larger lets corrections extend "
                     "a short chunk, with the certificate deciding how far.")
args = ap.parse_args()

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
             "libero_10": 520, "libero_90": 400}
DUMMY = [0.0] * 6 + [-1.0]
WAIT = 10
RES = 256
CONV, RAW = {
    "pi0": (f"{HOME}/openpi_assets/pi0_libero_pytorch",
            f"{HOME}/openpi_assets/pi0_libero"),
    "pi05": (f"{HOME}/openpi_assets/pi05_libero_pytorch",
             f"{HOME}/openpi_assets/pi05_libero"),
    # convert_lerobot_pi05.py writes weights, config and norm_stats in one place
    "lerobot05": (f"{HOME}/openpi_assets/lerobot_pi05_libero_pytorch",
                  f"{HOME}/openpi_assets/lerobot_pi05_libero_pytorch"),
}[args.model]
PI05 = args.model == "pi05"
# pi0_libero was trained with openpi's extra delta transform, so its first six
# action channels are offsets from the state at plan time and every stale chunk
# must be re-anchored.  Neither pi0.5 checkpoint was: their actions are LIBERO's
# own, and with zero delta channels the anchor add and the frame correction are
# no-ops.
DELTA = 0 if args.model in ("pi05", "lerobot05") else DELTA_DIMS
# openpi normalises pi0.5 by quantiles; LeRobot's run used MEAN_STD for both.
QUANT = PI05
# LeRobot's pi05 preprocessor writes the normalised state into the prompt
# ("Task: ..., State: <8 bins>;\nAction: "), so its tokens change every step.
STATE_PROMPT = args.model == "lerobot05"
TOK = f"{HOME}/assets/paligemma_tokenizer.model"
POSE = 6

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
    print(f"  student {args.student}: depth ({S_EV}, {S_EB}), "
          f"{len(_st['lora'])} adapter tensors, rel L2 {_st['rel']:.3f} "
          f"(floor {_st['floor']['rel']:.3f}), draw spread {_st['spread']:.2f}x")
    if len(_st["lora"]) == 0 or len(_res.unexpected_keys) > 0:
        sys.exit(f"adapter load failed: {len(_res.unexpected_keys)} unexpected "
                 "keys -- backend built at a different depth or rank than the "
                 "checkpoint")
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
    # the plan's own hidden state, which the plan computes and would otherwise
    # throw away; only read when the checkpoint was trained to use it
    TAP = {}
    if getattr(NET.net, "plan_dim", 0):
        model.action_out_proj.register_forward_hook(
            lambda m, inp, out: TAP.__setitem__("h", inp[0].detach()))
    print(f"  {NET.describe()}")
FULL_V, FULL_B = be.L_V, be.L_B

ns = find_norm_stats(RAW)
# openpi picks the normalisation by model type: z-score for pi0, quantiles for
# pi0.5 -- and a mismatch still produces plausible-looking actions.
spec = from_openpi_norm_stats(ns, d_a=D, use_quantiles=QUANT)
sm, ss = state_spec_from_openpi_norm_stats(ns, d_state=8, use_quantiles=QUANT)
_lp = LiberoPrompt(TOK, max_len=be.max_token_len)
_BINS = np.linspace(-1, 1, 256 + 1)[:-1]


def state_tokens(text, st8):
    """LeRobot's pi05 prompt, rebuilt from its processor_pi05 step verbatim.

    The normalised 8-dim state -- not padded, not clipped -- is digitised into
    256 bins over [-1, 1] and written into the text, which the PaliGemma
    tokenizer then encodes with BOS and right-pads to 200.
    """
    cleaned = text.strip().replace("_", " ").replace("\n", " ")
    bins = " ".join(map(str, np.digitize(st8, bins=_BINS) - 1))
    ids = _lp._sp.encode(f"Task: {cleaned}, State: {bins};\nAction: ", add_bos=True)
    ids = ids[: be.max_token_len]
    return torch.tensor(ids + [0] * (be.max_token_len - len(ids)), dtype=torch.int64)


# With the state in the prompt, a task's text can only be tokenised together
# with an observation, so prompt() hands the text through and to_observation
# does the rest.
prompt = (lambda text: text) if STATE_PROMPT else _lp


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
    lang = state_tokens(tokens, st[:8].numpy()) if isinstance(tokens, str) else tokens
    return Observation(images=images, language=lang, state=st, t=t)


@torch.no_grad()
def integrate(x0, tau0, n, obs, E_V=None, E_B=None, plan=None, age=None):
    """Integrate model-time from tau0 down to 0 in n equal Euler steps.

    A call that names no depth is a CORRECTION, so it runs the student when one
    is loaded and the teacher otherwise.  Callers that need the teacher whatever
    is loaded -- the reference plans that define SREF and the natural spread --
    pass the full depth explicitly, because those are the yardstick and must not
    move when the corrector changes.
    """
    ev = (S_EV if (E_V is None and S_EV is not None)
          else (FULL_V if E_V is None else E_V))
    eb = (S_EB if (E_B is None and S_EB is not None)
          else (FULL_B if E_B is None else E_B))
    use_ad = bool(args.student) and E_V is None
    x = x0.clone().float().to(be.device)
    tau = tau0.clone().float().to(be.device)
    step = (tau / max(n, 1)).view(-1, 1, 1)
    # The memory is built once and shared by the K draws, exactly as the trainer
    # did it -- that sharing is why a correction costs one encoder pass, not K.
    if NET is not None and E_V is None and getattr(NET.net, "plan_dim", 0) and plan is None:
        # the compile warm-up traces this before any episode has planned; use the
        # last plan the teacher made, or zeros when none has run yet
        _h = TAP.get("h")
        plan = (_h[0].float() if _h is not None
                else torch.zeros(NET.net.H, NET.net.plan_dim, device=be.device))
    mem = NET.memory(obs, plan=plan) if (NET is not None and E_V is None) else None
    for _ in range(n):
        if mem is not None:
            v = NET.velocity(x, tau, mem, age=age)
        else:
            v = be.velocity(A_tau=x, tau=tau, obs=obs, E_V=ev, E_B=eb,
                            adapters=use_ad)
        x = x - step * v.float()
        tau = tau - step.view(-1)
    return x


def dist(X, Y, m):
    return float(torch.linalg.vector_norm(
        X[:m, :POSE] - Y[:m, :POSE], ord=2, dim=-1).mean())


def _pct(v, f):
    if not v:
        return None
    v = sorted(v)
    return v[min(len(v) - 1, int(f * (len(v) - 1)))]


def hold_pad(A, H):
    if A.shape[0] >= H:
        return A[:H].clone()
    return torch.cat([A, A[-1:].expand(H - A.shape[0], A.shape[1])], dim=0)


# The gripper is binary in intent -- LIBERO encodes the two modes as +-1 -- but
# the flow treats it as one more continuous channel.  That asymmetry is the
# whole hypothesis: for a bimodal channel the conditional mean sits BETWEEN the
# modes, and "half closed" is a command that grasps nothing.  Averaging two
# nearby pose trajectories gives a usable trajectory; averaging open and closed
# gives neither.  These are the two mode locations in the policy's normalised
# space, which is where the correction actually operates.
G = spec.grip
G_LO = float((spec.grip_raw_modes[0] - spec.mean[G]) / spec.scale[G])
G_HI = float((spec.grip_raw_modes[1] - spec.mean[G]) / spec.scale[G])
G_MID = 0.5 * (G_LO + G_HI)
G_HALF = 0.5 * abs(G_HI - G_LO)


def grip_stats(A_new, A_stale, m):
    """How crisp is the corrected gripper, did it change its mind, does it chatter?

    Snapping the gripper to its nearer mode moved success from 92% to 98% while
    the MEAN distance it had to travel was 2.6% of the half-separation.  A mean
    that small producing an effect that large means the mean is the wrong
    statistic, and two different mechanisms fit the evidence equally well:

      tail     a handful of positions -- at the grasp transitions, where it
               matters -- sit far from either mode, and the average hides them.
      chatter  the value loiters near the midpoint so the commanded side flips
               between ADJACENT executed steps, and the gripper never commits.

    So return the per-position ambiguities rather than their mean (percentiles
    are computed at the end), and count side changes between adjacent positions
    of the executed segment -- for the corrected chunk and for the stale chunk it
    came from, since chatter is only evidence if the correction introduced it.
    """
    g_new, g_old = A_new[:m, G], A_stale[:m, G]
    near = torch.minimum((g_new - G_LO).abs(), (g_new - G_HI).abs())
    side_new, side_old = g_new > G_MID, g_old > G_MID

    def chatter(side):
        return (float((side[1:] != side[:-1]).float().mean())
                if side.numel() > 1 else 0.0)

    return ((near / G_HALF).tolist(),
            float((side_new != side_old).float().mean()),
            chatter(side_new), chatter(side_old))


def apply_grip_mode(A_new, A_stale):
    if args.grip == "hold":
        A_new[:, G] = A_stale[:, G]
    elif args.grip == "snap":
        g = A_new[:, G]
        A_new[:, G] = torch.where((g - G_LO).abs() <= (g - G_HI).abs(),
                                  torch.full_like(g, G_LO),
                                  torch.full_like(g, G_HI))
    return A_new


def write_episode_video(frames, ok, language, config, path, scale=2, fps=20):
    """Agentview frames with a bar showing the action source: orange while a plan's chunk
    runs, green while a correction's does; the last frames carry the outcome."""
    import cv2
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    h, w = frames[0][0].shape[:2]
    H, W, bar = h * scale, w * scale, 28
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H + bar))
    col = {"plan": (0, 140, 255), "correction": (80, 200, 60)}      # BGR
    name = "replan every %d" % args.replan if config == "baseline" else "corrector"
    for i, (f, kind) in enumerate(frames + [frames[-1]] * fps):
        img = cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2BGR), (W, H), interpolation=cv2.INTER_NEAREST)
        top = np.zeros((bar, W, 3), np.uint8)
        top[:] = col[kind]
        label = f"{name}  step {min(i, len(frames) - 1)}  {'PLAN' if kind == 'plan' else 'CORRECTION'}"
        if i >= len(frames):
            label = f"{name}  {'SUCCESS' if ok else 'FAILURE'} at step {len(frames) - 1}"
            top[:] = (60, 160, 60) if ok else (60, 60, 200)
        cv2.putText(top, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(np.vstack([top, img]))
    vw.release()


class LiberoEnv:
    def __init__(self, env, tokens, t0_raw):
        self._env, self._tokens, self._raw = env, tokens, t0_raw
        self._t = 0
        self._done = False
        self._anchor = raw_state(t0_raw)
        self.kind = "plan"                     # source of the chunk now executing
        self.frames = None                     # [(frame, kind)] when recording

    def mark_plan_anchor(self):
        self._anchor = raw_state(self._raw)

    def observe(self):
        return to_observation(self._raw, self._tokens, self._t)

    def step(self, action):
        a = action.detach().cpu() * spec.scale + spec.mean
        a[:DELTA] += torch.from_numpy(self._anchor[:DELTA])
        self._raw, _, done, _ = self._env.step(a[:7].tolist())
        self._t += 1
        if self.frames is not None:
            self.frames.append((np.ascontiguousarray(self._raw["agentview_image"][::-1, ::-1]), self.kind))
        self._done = self._done or bool(done)

    def frame_correction(self):
        now = raw_state(self._raw)
        d = self._anchor[:DELTA] - now[:DELTA]
        return torch.from_numpy(d).float() / spec.scale[:DELTA].cpu()

    @property
    def terminated(self):
        return self._done


# ==========================================================================
# the two loops
# ==========================================================================
@torch.no_grad()
def run_baseline(env, gen, T_max):
    """pi_0 as it ships: full plan, execute args.replan actions, full plan."""
    steps = plans = 0
    while not env.terminated and steps < T_max:
        env.mark_plan_anchor()
        noise = torch.randn(H, D, generator=gen)
        A = be.plan(env.observe(), noise=noise).cpu()
        plans += 1
        env.kind = "plan"
        for i in range(args.replan):
            if env.terminated or steps >= T_max:
                break
            env.step(A[i])
            steps += 1
    return dict(steps=steps, plans=plans, corrections=0, fallbacks=0,
                displacements=[], grip_amb=[], grip_flip=[],
                grip_chat=[], grip_chat_stale=[], real_hist=[], exhausted=0,
                disagree=[], n_hist=[])


@torch.no_grad()
def run_corrector(env, gen, T_max):
    """Plan once, then correct.  A full replan happens only on fallback."""
    steps = plans = corrections = fallbacks = 0
    teacher_fix = teacher_saves = 0   # --cascade: teacher retries, and those that held
    disp, grip_amb, grip_flip, disagree, n_hist = [], [], [], [], []
    grip_chat, grip_chat_stale = [], []
    real_hist = []
    A = None
    since_plan = 0
    capped = False
    just_planned = False
    used = 0
    last_seg = args.c      # how many actions the previous cycle executed
    real_left = 0          # planned entries still in the chunk, before padding
    plan_feat = None       # the teacher's state when the live plan was made
    exhausted = 0          # corrections executed with none left

    # One cycle before the chunk runs out of planned content.  floor(H/c) is
    # when it hits zero, so the last cycle with anything real is one earlier.
    auto = max(1, H // max(args.c, 1) - 1) if args.auto_cap else 0

    while not env.terminated and steps < T_max:
        if A is None:
            env.mark_plan_anchor()
            noise = torch.randn(H, D, generator=gen)
            A = be.plan(env.observe(), noise=noise).cpu()
            if NET is not None and getattr(NET.net, "plan_dim", 0) and "h" in TAP:
                plan_feat = TAP["h"][0].float()
            plans += 1
            since_plan = 0
            real_left = H
            used = 0
            just_planned = True
        else:
            just_planned = False
            obs_now = env.observe()
            fc = env.frame_correction()
            # index 0 must mean "act now", and the six pose channels must be
            # re-expressed against the state the model is about to condition on
            # A segment can consume the whole chunk once corrections outlive the
            # horizon (--lineage > H with c_max = H, reachable on pi05's H = 10,
            # never on pi0's 50 with c_max 25); the stale guess is then the last
            # action held, which is what hold_pad does to any tail.
            rest = A[last_seg:]
            A_stale = hold_pad(rest if rest.shape[0] else A[-1:], H)
            A_stale[:, :DELTA] += fc

            K = args.k_draws
            eps = torch.randn(K, H, D, generator=gen)
            x0 = args.tau0 * eps + (1.0 - args.tau0) * A_stale.unsqueeze(0)
            out = integrate(x0, torch.full((K,), args.tau0),
                            args.steps, obs_now, plan=plan_feat, age=used).cpu()
            if K > 1:
                # How far apart the draws land, over the actions about to be
                # executed.  Same observation, same stale chunk, same tau0 --
                # only the noise differs, so this is the flow's own spread at
                # this point and nothing else.
                dis = statistics.median(
                    [dist(out[i], out[j], min(args.c, args.c_max))
                     for i in range(K) for j in range(i + 1, K)])
                disagree.append(dis)
            else:
                dis = 0.0
            A_new = (out[0] if args.k_reduce == "first"
                     else out.median(dim=0).values).clone()

            # -- how far do the draws agree, position by position?
            #
            # Two tests per position, and the second is the one that matters:
            # the pose spread says whether the arm's path is settled, while the
            # gripper test says whether the draws agree on opening or closing.
            # Every distance in this file excludes the gripper -- correctly, for
            # a continuous metric -- which means every verify signal tried so far
            # has been blind to the one channel where the failures were measured
            # to live (snap moved 92% to 98% by touching nothing else).
            n_agree = args.c
            if args.adaptive and K > 1:
                pose = out[:, :args.c_max, :POSE]
                spread = torch.stack(
                    [torch.linalg.vector_norm(pose[i] - pose[j], ord=2, dim=-1)
                     for i in range(K) for j in range(i + 1, K)]).median(dim=0).values
                ok_pose = spread <= args.agree_tau * SREF[:args.c_max]
                side = out[:, :args.c_max, G] > G_MID
                ok_grip = (side.all(dim=0) | (~side).all(dim=0))
                ok = (ok_pose & ok_grip).tolist()
                n_agree = 0
                for good in ok:
                    if not good:
                        break
                    n_agree += 1
                n_hist.append(n_agree)
            ambs, flip, ch_new, ch_old = grip_stats(A_new, A_stale,
                                                    min(args.c, args.c_max))
            grip_amb.extend(ambs)
            grip_flip.append(flip)
            grip_chat.append(ch_new)
            grip_chat_stale.append(ch_old)
            A_new = apply_grip_mode(A_new, A_stale)
            corrections += 1
            since_plan += 1
            real_left = max(0, real_left - last_seg)
            if real_left == 0:
                exhausted += 1
            real_hist.append(real_left)

            # The displacement is the operator's own output, so this costs
            # nothing: a correction that has to move the chunk a long way is one
            # the flow is dragging somewhere else entirely, which is exactly the
            # case a single Euler step has no business resolving.
            d = dist(A_new, A_stale, min(args.c, args.c_max))
            disp.append(d)

            blind = (args.blind_fallback > 0
                     and float(torch.rand(1, generator=gen)) < args.blind_fallback)
            too_far = ((args.adaptive and n_agree < args.n_min)
                       or (args.fallback > 0 and d > args.fallback * NATURAL)
                       or (args.k_fallback > 0 and dis > args.k_fallback * NATURAL)
                       or blind)
            capped = ((args.max_corrections > 0
                       and since_plan >= args.max_corrections)
                      or (auto > 0 and since_plan >= auto))

            # A cap says "this is the LAST correction before a replan", not
            # "throw this one away".  Discarding it -- which is what an early
            # `continue` here did -- pays 39.58 ms for a chunk that is never
            # executed and then pays for a full plan on top, so c=25 with a cap
            # of one correction cost 3.90 ms/step instead of the 1.95 the design
            # intends, and the configuration that was supposed to be tested
            # never ran.  Execute the correction, then let the next iteration
            # replan.
            if (too_far and args.cascade and NET is not None and not blind
                    and n_agree >= args.cascade_min):
                # The student says it is unsure.  Ask the operator whose
                # certificate already holds up, for one Euler step on the same
                # draws -- cheaper than the replan this would otherwise cost.
                out_t = integrate(x0, torch.full((K,), args.tau0), args.steps,
                                  obs_now, E_V=FULL_V, E_B=FULL_B).cpu()
                teacher_fix += 1
                A_t = (out_t[0] if args.k_reduce == "first"
                       else out_t.median(dim=0).values).clone()
                n_t = args.c
                if args.adaptive and K > 1:
                    pose_t = out_t[:, :args.c_max, :POSE]
                    spread_t = torch.stack(
                        [torch.linalg.vector_norm(pose_t[i] - pose_t[j], ord=2, dim=-1)
                         for i in range(K) for j in range(i + 1, K)]).median(dim=0).values
                    side_t = out_t[:, :args.c_max, G] > G_MID
                    ok_t = ((spread_t <= args.agree_tau * SREF[:args.c_max])
                            & (side_t.all(dim=0) | (~side_t).all(dim=0))).tolist()
                    n_t = 0
                    for good in ok_t:
                        if not good:
                            break
                        n_t += 1
                d_t = dist(A_t, A_stale, min(args.c, args.c_max))
                if not ((args.adaptive and n_t < args.n_min)
                        or (args.fallback > 0 and d_t > args.fallback * NATURAL)):
                    A_new = apply_grip_mode(A_t, A_stale)
                    n_agree, too_far = n_t, False
                    teacher_saves += 1

            if too_far:
                fallbacks += 1
                A = None
                continue

            # the corrected chunk is conditioned on o_now, so its deltas are
            # relative to the CURRENT state -- re-anchor before executing it
            env.mark_plan_anchor()
            A = A_new
            env.kind = "correction"

        # Normally one segment is c actions.  On the cycle that will replan,
        # --drain lets it absorb whatever planned content is left rather than
        # stranding it: those actions are already paid for and executing them
        # costs nothing.
        if args.adaptive:
            # After a plan there are no draws to read, so take the safe
            # interval; after a correction, take what the draws agreed on.
            seg = args.c_safe if just_planned else max(args.n_min,
                                                       min(n_agree, args.c_max))
            capped = used + seg >= (args.lineage or H)   # spent: replan next
        else:
            seg = args.c
            if args.drain and capped:
                seg = max(args.c, H - used)
        for i in range(seg):
            if env.terminated or steps >= T_max:
                break
            env.step(A[i])
            steps += 1
            used += 1

        last_seg = seg
        if capped:
            A = None          # replan on the next iteration, having used this one
        capped = False

    return dict(steps=steps, plans=plans, corrections=corrections,
                teacher_fix=teacher_fix, teacher_saves=teacher_saves,
                fallbacks=fallbacks, displacements=disp,
                grip_amb=grip_amb, grip_flip=grip_flip,
                grip_chat=grip_chat, grip_chat_stale=grip_chat_stale,
                real_hist=real_hist, exhausted=exhausted, disagree=disagree,
                n_hist=n_hist)


# ==========================================================================
# setup
# ==========================================================================
suite = benchmark.get_benchmark_dict()[args.suite]()
n_tasks = suite.n_tasks if args.tasks == 0 else min(args.tasks, suite.n_tasks)
T_max = MAX_STEPS[args.suite]


def init_states_for(suite, task_id):
    task = suite.get_task(task_id)
    return torch.load(os.path.join(get_libero_path("init_states"),
                                   task.problem_folder, task.init_states_file),
                      weights_only=False)


_task0 = suite.get_task(0)
_bddl = os.path.join(get_libero_path("bddl_files"),
                     _task0.problem_folder, _task0.bddl_file)
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
    else:
        with torch.no_grad():                    # trace both shapes used
            integrate(torch.randn(1, H, D), torch.tensor([0.5]), 1, _obs)
            be.plan(_obs, noise=_noise)

# The scale every distance is quoted against: how far two honest full-depth
# plans from the SAME observation already sit apart.  Estimated over several
# observations, not one.  The single-observation estimate used earlier swung
# between 0.45 and 0.64 from run to run, which moved every fallback threshold
# expressed as a multiple of it -- so the displacement fallback was tested
# against a yardstick that changed underneath it.
_scales, _perpos = [], []
with torch.no_grad():
    for _tid in range(min(4, suite.n_tasks)):
        _t = suite.get_task(_tid)
        _b = os.path.join(get_libero_path("bddl_files"), _t.problem_folder, _t.bddl_file)
        _ev = OffScreenRenderEnv(bddl_file_name=_b, camera_heights=RES, camera_widths=RES)
        _ev.seed(args.seed); _ev.reset()
        _r = _ev.set_init_state(init_states_for(suite, _tid)[0])
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
            _scales.append(statistics.median(
                [dist(_R[i], _R[j], args.c)
                 for i in range(8) for j in range(i + 1, 8)]))
            _pp = _R[:, :args.c_max, :POSE]
            _perpos.append(torch.stack(
                [torch.linalg.vector_norm(_pp[i] - _pp[j], ord=2, dim=-1)
                 for i in range(8) for j in range(i + 1, 8)]).median(dim=0).values)
        _ev.close()
NATURAL = statistics.median(_scales)

# Per-POSITION reference spread.  Two honest plans agree closely on the action
# to take now and drift apart further out, so a single scalar threshold would
# accept the far end too readily and reject the near end too harshly.  This is
# the same estimate resolved along the chunk instead of averaged over it.
_sref = torch.stack(_perpos).median(dim=0).values          # (c_max,)
SREF = _sref.clamp(min=1e-6)
print(f"  per-position spread: h=0 {SREF[0]:.3f}  h=5 {SREF[min(5,len(SREF)-1)]:.3f}"
      f"  h={len(SREF)-1} {SREF[-1]:.3f}")
print(f"  natural plan-to-plan spread over {args.c} actions: {NATURAL:.4f}"
      f"   [{len(_scales)} observations, {min(_scales):.3f}-{max(_scales):.3f}]")
_e.close()

with open(args.latency_from) as f:
    LAT = json.load(f)["latency_ms"]
L_PLAN = LAT["plan-plan"]
_ck = f"{S_EV}-{S_EB}" if args.student else f"{FULL_V}-{FULL_B}"
if _ck not in LAT:
    sys.exit(f"no measured latency for depth {_ck} in {args.latency_from}")
L_CORR = LAT[_ck]
# One velocity evaluation, measured directly at batch 1 (44_/45_): the encoder
# and prefill are paid once whatever the step count, so extra steps cost only
# the denoise term.  L_CORR above was timed at batch 4 and so overstates a
# batch-1 correction by about 4%; it is kept as the base anyway, because every
# speedup already on record uses it and a silent change of basis would make the
# numbers incomparable.  The bias is conservative -- it understates the method.
L_DEN = 2.27
# Widening the batch costs only the action expert: 37.90 ms at K=1 against
# 39.28 at K=4 (45_probe_overhead, idle GPU), so about 0.46 ms per extra draw.
# The K=1 base stays at the batch-4 figure every earlier number used, which
# overstates the corrector by ~4% -- conservative, and comparable.
if args.net:
    # The net does not sit on pi0's depth ladder, so the depth-keyed table does
    # not describe it.  Its figure was timed the same way: idle GPU, K=4 draws
    # sharing one memory, eager -- so it is comparable to L_CORR above.
    L_CORR_N = args.net_latency
    _ck = "net"
else:
    L_CORR_N = L_CORR + (args.steps - 1) * L_DEN + (args.k_draws - 1) * 0.46
print(f"  cost model (idle-GPU measurements): plan {L_PLAN:.2f} ms, "
      f"correction {L_CORR_N:.2f} ms @ {_ck}"
      + (f" ({args.steps} Euler steps)" if args.steps > 1 else "") + "\n")

# ==========================================================================
# run
# ==========================================================================
results = {}
for config in args.configs:
    print(f"{'='*74}\n{config.upper()}"
          + (f"   replan={args.replan}" if config == "baseline"
             else f"   c={args.c}  tau0={args.tau0}  steps={args.steps}"
                  f"  K={args.k_draws}/{args.k_reduce}  grip={args.grip}"
                  f"{'  kfb=' + str(args.k_fallback) if args.k_fallback else ''}"
                  f"{'  blind=' + str(args.blind_fallback) if args.blind_fallback else ''}"
                  f"{'  fallback=' + str(args.fallback) if args.fallback else ''}"
                  f"{'  cap=' + str(args.max_corrections) if args.max_corrections else ''}")
          + f"\n{'='*74}")
    succ_total = n_ep = 0
    tot = collections.Counter()
    disp_all, amb_all, flip_all, chat_all, chat_st_all = [], [], [], [], []
    per_ep = []
    real_all, dis_all, n_all = [], [], []
    t0 = time.time()

    for task_id in ([int(x) - 1 for x in args.task_ids.split(",")] if args.task_ids
                    else range(n_tasks)):
        task = suite.get_task(task_id)
        tokens = prompt(task.language)
        bddl = os.path.join(get_libero_path("bddl_files"),
                            task.problem_folder, task.bddl_file)
        inits = init_states_for(suite, task_id)
        succ = 0

        for trial in ([int(x) for x in args.trial_ids.split(",")] if args.trial_ids
                      else range(args.trials)):
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=RES,
                                     camera_widths=RES)
            if args.random_scenes:
                env.seed(args.scene_seed + 1000 * task_id + trial)
                raw = env.reset()
            else:
                env.seed(args.seed)
                env.reset()
                raw = env.set_init_state(inits[trial % len(inits)])
            done = False
            for _ in range(WAIT):
                raw, _, done, _ = env.step(DUMMY)
            w = LiberoEnv(env, tokens, raw)
            w._done = done
            if args.video_dir:
                w.frames = [(np.ascontiguousarray(raw["agentview_image"][::-1, ::-1]), "plan")]

            # Identical across configurations, so the n-th draw of a given
            # (task, trial) is the same on both sides and episodes the corrector
            # never diverts contribute nothing to the variance of the
            # difference.  That is a paired comparison and it is worth far more
            # than the same number of unpaired episodes.
            gen = torch.Generator().manual_seed(
                args.seed * 1_000_003 + task_id * 10_007 + trial * 101)

            tr = (run_baseline if config == "baseline" else run_corrector)(
                w, gen, T_max)
            env.close()

            ok = int(w.terminated)
            succ += ok
            if args.video_dir and w.frames:
                write_episode_video(w.frames, ok, task.language, config,
                                    f"{args.video_dir}/{args.suite}_t{task_id}_r{trial}_{config}_"
                                    f"{'success' if ok else 'failure'}.mp4")
            n_ep += 1
            # Per-episode outcomes, so two configurations can be compared as
            # PAIRS rather than as two totals.  A 2-episode gap between 147/150
            # and 145/150 says nothing on its own; the discordant pairs -- how
            # often one wins where the other loses -- say whether it is a real
            # difference or the same quality reshuffled by trajectory
            # divergence.
            per_ep.append(dict(task=task_id, trial=trial, ok=ok))
            real_all.extend(tr["real_hist"])
            dis_all.extend(tr["disagree"])
            n_all.extend(tr["n_hist"])
            for k in ("steps", "plans", "corrections", "fallbacks", "exhausted",
                      "teacher_fix", "teacher_saves"):
                tot[k] += tr.get(k, 0)
            disp_all.extend(tr["displacements"])
            amb_all.extend(tr["grip_amb"])
            flip_all.extend(tr["grip_flip"])
            chat_all.extend(tr["grip_chat"])
            chat_st_all.extend(tr["grip_chat_stale"])

        succ_total += succ
        print(f"  [{task_id+1}/{n_tasks}] {succ}/{args.trials}  "
              f"{task.language[:48]}", flush=True)

    ms = (tot["plans"] * L_PLAN + tot["corrections"] * L_CORR_N
          + tot["teacher_fix"] * L_CORR)
    per_step = ms / max(tot["steps"], 1)
    r = dict(per_episode=per_ep, success=succ_total, episodes=n_ep,
             rate=100.0 * succ_total / max(n_ep, 1),
             steps=tot["steps"], plans=tot["plans"],
             corrections=tot["corrections"], fallbacks=tot["fallbacks"],
             teacher_fix=tot["teacher_fix"], teacher_saves=tot["teacher_saves"],
             exhausted=tot["exhausted"],
             frac_exhausted=(tot["exhausted"] / max(tot["corrections"], 1)),
             steps_per_plan=tot["steps"] / max(tot["plans"], 1),
             ms_per_step=per_step, wall_s=time.time() - t0,
             disp_median=(statistics.median(disp_all) if disp_all else None),
             disp_p90=(sorted(disp_all)[int(0.9 * len(disp_all))]
                       if disp_all else None),
             grip_ambiguity=(statistics.mean(amb_all) if amb_all else None),
             grip_amb_p90=_pct(amb_all, 0.90), grip_amb_p99=_pct(amb_all, 0.99),
             grip_amb_max=(max(amb_all) if amb_all else None),
             grip_flip_rate=(statistics.mean(flip_all) if flip_all else None),
             grip_chatter=(statistics.mean(chat_all) if chat_all else None),
             grip_chatter_stale=(statistics.mean(chat_st_all) if chat_st_all else None),
             n_mean=(statistics.mean(n_all) if n_all else None),
             n_p10=_pct(n_all, 0.10), n_med=_pct(n_all, 0.50),
             n_p90=_pct(n_all, 0.90),
             disagree_med=_pct(dis_all, 0.50), disagree_p75=_pct(dis_all, 0.75),
             disagree_p90=_pct(dis_all, 0.90), disagree_p99=_pct(dis_all, 0.99))
    results[config] = r
    print(f"\n  success {succ_total}/{n_ep} = {r['rate']:.1f}%"
          f"   steps/plan {r['steps_per_plan']:.1f}"
          f"   {r['plans']} plans + {r['corrections']} corrections"
          f"   {per_step:.2f} ms/step   [{r['wall_s']:.0f}s]")
    if real_all:
        print(f"  planned content: {100*r['frac_exhausted']:.0f}% of corrections "
              f"ran with NO planned entries left "
              f"({tot['exhausted']}/{tot['corrections']}) -- those executed "
              f"actions descend from hold-padding")
    if n_all:
        print(f"  adaptive interval N: mean {r['n_mean']:.1f}  "
              f"p10 {r['n_p10']}  median {r['n_med']}  p90 {r['n_p90']}"
              f"   (n_min {args.n_min}, cap {args.c_max})")
    if dis_all:
        print(f"  K-draw disagreement, in units of the natural spread "
              f"{NATURAL:.4f}:  median {r['disagree_med']/NATURAL:.2f}"
              f"  p75 {r['disagree_p75']/NATURAL:.2f}"
              f"  p90 {r['disagree_p90']/NATURAL:.2f}"
              f"  p99 {r['disagree_p99']/NATURAL:.2f}")
    if disp_all:
        print(f"  correction displacement: median {r['disp_median']:.4f}"
              f"  p90 {r['disp_p90']:.4f}   (natural spread {NATURAL:.4f})")
        print(f"  gripper ambiguity (0=a real mode, 1=midway): "
              f"mean {r['grip_ambiguity']:.3f}  p90 {r['grip_amb_p90']:.3f}  "
              f"p99 {r['grip_amb_p99']:.3f}  max {r['grip_amb_max']:.3f}")
        print(f"  gripper side changed by the correction at "
              f"{100*r['grip_flip_rate']:.1f}% of positions;  chatter between "
              f"adjacent executed steps {100*r['grip_chatter']:.1f}% "
              f"(stale chunk: {100*r['grip_chatter_stale']:.1f}%)")
    print()

if "baseline" in results and "corrector" in results:
    b, c = results["baseline"], results["corrector"]
    n = b["episodes"]
    se = math.sqrt(max(b["rate"] * (100 - b["rate"]), 1e-9) / n) / 100 * 100
    print(f"{'='*74}")
    print(f"success   {b['rate']:.1f}%  ->  {c['rate']:.1f}%   "
          f"({c['rate']-b['rate']:+.1f} points, binomial s.e. ~{se:.1f} on "
          f"{n} episodes)")
    print(f"ms/step   {b['ms_per_step']:.2f}  ->  {c['ms_per_step']:.2f}   "
          f"= {b['ms_per_step']/c['ms_per_step']:.2f}x")
    print(f"{'='*74}")

with open(args.out, "w") as f:
    json.dump(dict(suite=args.suite, tasks=n_tasks, trials=args.trials,
                   seed=args.seed, replan=args.replan, c=args.c, tau0=args.tau0,
                   fallback=args.fallback, max_corrections=args.max_corrections,
                   grip=args.grip, steps=args.steps, L_corr_n=L_CORR_N,
                   k_draws=args.k_draws, k_reduce=args.k_reduce,
                   k_fallback=args.k_fallback,
                   blind_fallback=args.blind_fallback,
                   natural=NATURAL, L_plan=L_PLAN, L_corr=L_CORR,
                   model=args.model, lineage=(args.lineage or H), horizon=H,
                   compile=COMPILE_MODE, results=results), f, indent=2)
print(f"wrote {args.out}")
