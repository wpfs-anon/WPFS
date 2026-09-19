"""Distil the corrector into a truncated pi_0 with LoRA.

The corrector is queried at one point of the flow -- model time tau0=0.5, one
Euler step -- so this is regression on a single time slice, not consistency
distillation: there are no sampling steps left to remove.

What the floor measurement (48_) says about the problem, and why it shapes the
design:

    depth 14-12, untrained:   rel L2 1.995   cosine 0.849   draw spread 20.4x

Direction is largely preserved and magnitude is not: solving
r^2 - 2r*cos + 1 = relL2^2 gives ||v_shallow|| / ||v_teacher|| ~ 2.8, and a single
scalar rescale would already take rel L2 to sqrt(2 - 2*0.849) = 0.55.  The
dominant error is therefore a gain on the read-out, exactly the mismatch a
low-rank update to one projection is shaped to fix.  Everything below is aimed
at that, with the trunk adapters as the second-order correction.

Two things this file will not let happen quietly:

  * `be.velocity` is decorated `@torch.no_grad()`, so it cannot be trained
    through.  The forward here is rebuilt from `_prefix` + `denoise_step`, which
    are not, and is checked against `be.velocity` at startup: if the two disagree
    the student is being trained on a different function than the one deployed.

  * A student can drive rel L2 down while collapsing the disagreement between
    draws, which is the quantity the acceptance test reads.  Loss would look
    perfect and the mechanism would accept everything.  So the draw spread is
    logged every validation pass, and it has to land near 1.0 -- approached from
    ABOVE, since truncation starts at 20x.
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

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, f"{HOME}/openpi/src")
sys.path.insert(0, f"{HOME}/openpi/packages/openpi-client/src")
sys.path.insert(0, f"{HOME}/src")
sys.path.insert(0, f"{HOME}/scripts")

import cv2
import numpy as np
import torch

torch.set_num_threads(4)

from sentry.core.types import Observation
from sentry.models.lora import (
    adapters as adapters_ctx, lora_parameters, count_lora_parameters,
    set_adapter_slot,
)
from sentry.models.openpi_adapter import (
    OpenPiBackend, load_pi0_pytorch, _expand_cache, _Namespace,
)

ap = argparse.ArgumentParser()
ap.add_argument("--shard", default=f"{HOME}/distill/train_s101.pt",
                help="one or more shards, comma separated.  Scene diversity is "
                     "what a second shard buys: LIBERO-Spatial has 10 tasks and "
                     "50 init states and the first harvest used all 500, so "
                     "another seed of the same suite gives different "
                     "trajectories through the SAME rooms.  A different suite "
                     "gives different rooms.")
ap.add_argument("--e-v", type=int, default=14)
ap.add_argument("--e-b", type=int, default=12)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--rank-readout", type=int, default=32)
ap.add_argument("--k-train", type=int, default=8,
                help="interpolants per observation per step; they share the "
                     "prefix, so this is nearly free next to the encoder")
ap.add_argument("--obs-batch", type=int, default=1,
                help="observations whose encoder and prefill run in ONE pass.\n"
                     "The backend's _prefix is written for batch 1, so the loop "
                     "has been paying SigLIP and the backbone prefill once per "
                     "observation -- latency-bound work that leaves any GPU "
                     "idle.  Batching them is worth more than a faster card: it "
                     "is the same arithmetic with the accelerator actually fed. "
                     "Guarded at startup against the sequential path.")
ap.add_argument("--accum", type=int, default=4,
                help="observations per optimiser step.  _prefix runs at batch 1, "
                     "so several observations means accumulation, not a wider "
                     "batch")
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--val-frac", type=float, default=0.05)
ap.add_argument("--val-every", type=int, default=200,
                help="optimiser steps between validation passes")
ap.add_argument("--val-obs", type=int, default=64)
ap.add_argument("--warmup", type=int, default=100)
ap.add_argument("--out", default=f"{HOME}/distill/student")
ap.add_argument("--mask-channels", type=int, default=1,
                help="restrict the loss to the action channels LIBERO actually "
                     "uses.\n"
                     "d_a is 32 but the robot takes 7 -- six pose and the "
                     "gripper -- and the other 25 are padding the policy fills "
                     "with large, easy, meaningless values.  An unmasked MSE "
                     "therefore spends 78%% of its gradient on channels nothing "
                     "reads, which is what a breakdown of the first student "
                     "showed: 0.081 over all 32 channels against 0.134 on pose "
                     "and 0.148 on the gripper.  The headline number was diluted "
                     "by the padding, and so was the training signal.")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

CONV = f"{HOME}/openpi_assets/pi0_libero_pytorch"
POSE, CMAX = 6, 25
ACT = 7          # six pose channels plus the gripper; 7..31 are padding
torch.manual_seed(args.seed)

# ==========================================================================
# data
# ==========================================================================
paths = [q.strip() for q in args.shard.split(",") if q.strip()]
parts = []
for q in paths:
    print(f"loading {q} ...", flush=True)
    parts.append(torch.load(q, weights_only=False))
TAU0 = float(parts[0]["tau0"])
for q, d in zip(paths, parts):
    if float(d["tau0"]) != TAU0:
        sys.exit(f"{q} was harvested at tau0={d['tau0']}, not {TAU0}; the "
                 "student is only ever queried at one flow time, so mixing "
                 "them would train it on a slice it never sees")
    if int(d["k_harvest"]) != int(parts[0]["k_harvest"]):
        sys.exit(f"{q} has k_harvest={d['k_harvest']}, not {parts[0]['k_harvest']}")

# sref is per-shard and per-suite; it is a rollout-time yardstick and is not
# read during training, so the shards concatenate cleanly without it.
sh = dict(
    jpg_a=[b for d in parts for b in d["jpg_a"]],
    jpg_w=[b for d in parts for b in d["jpg_w"]],
    state=torch.cat([d["state"] for d in parts]),
    lang=torch.cat([d["lang"] for d in parts]),
    x=torch.cat([d["x"] for d in parts]),
    v=torch.cat([d["v"] for d in parts]),
    k_harvest=parts[0]["k_harvest"], tau0=TAU0,
)
del parts
N = sh["x"].shape[0]
print(f"  {len(paths)} shard(s): {N} observations x {sh['k_harvest']} "
      f"interpolants, tau0={TAU0}")

g = torch.Generator().manual_seed(args.seed)
perm = torch.randperm(N, generator=g).tolist()
n_val = max(16, int(args.val_frac * N))
VAL, TRAIN = perm[:n_val], perm[n_val:]
print(f"  {len(TRAIN)} train / {len(VAL)} val observations")


def unjpg(buf):
    a = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def obs_of(i):
    frames = np.stack([unjpg(sh["jpg_a"][i]),
                       unjpg(sh["jpg_w"][i])]).astype(np.float32) / 127.5 - 1.0
    return Observation(
        images=torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous(),
        language=sh["lang"][i].to(torch.int64),
        state=sh["state"][i].float(), t=0)


def batch_of(i, k, gen=None):
    """k interpolants and their teacher velocities for observation i."""
    kk = sh["x"].shape[1]
    sel = (torch.randperm(kk, generator=gen)[:k] if gen is not None
           else torch.arange(k))
    return sh["x"][i, sel].float(), sh["v"][i, sel].float()


# ==========================================================================
# model
# ==========================================================================
model = load_pi0_pytorch(CONV, device="cuda")
be = OpenPiBackend(model, device="cuda", M=10,
                   E_V_max=args.e_v, E_B_max=args.e_b, n_rungs=1,
                   lora_rank=args.rank, lora_rank_readout=args.rank_readout,
                   lora_dropout=0.0, attach_adapters=True)
model.eval()                       # no dropout anywhere; adapters are gated
set_adapter_slot(model, 0)
for p in model.parameters():
    p.requires_grad_(False)
params = [p for p in lora_parameters(model)]
for p in params:
    p.requires_grad_(True)
n_lora = count_lora_parameters(model)
print(f"  adapters at ({args.e_v}, {args.e_b}): {len(params)} tensors, "
      f"{n_lora/1e6:.2f}M trainable of {sum(p.numel() for p in model.parameters())/1e9:.2f}B")


def _batched_obs(obs_list):
    """The openpi observation namespace built from B DIFFERENT observations.

    ``be._openpi_obs`` replicates ONE observation across the batch, which is
    what the K interpolants need and not what several observations need.  This
    builds the same fields with a genuinely different row per observation, so
    the encoder and the prefill run once for the whole group.
    """
    dev = be.device
    ref = obs_list[0].images[0]
    images, masks = {}, {}
    for j, key in enumerate(be.image_keys):
        frames, present = [], []
        for o in obs_list:
            if j < o.images.shape[0]:
                frames.append(o.images[j].to(device=dev, dtype=torch.float32))
                present.append(True)
            else:
                frames.append(torch.zeros_like(ref).to(device=dev, dtype=torch.float32))
                present.append(False)
        images[key] = torch.stack(frames)
        masks[key] = torch.tensor(present, dtype=torch.bool, device=dev)
    tok = torch.stack([o.language.to(dev) for o in obs_list])
    return _Namespace(
        images=images, image_masks=masks,
        state=torch.stack([o.state.to(device=dev, dtype=torch.float32)
                           for o in obs_list]),
        tokenized_prompt=tok, tokenized_prompt_mask=(tok != 0).bool(),
        token_ar_mask=None, token_loss_mask=None,
    )


def _repeat_cache(cache, K):
    """Give each of the B prefixes K consecutive rows.

    ``_expand_cache`` widens a batch-1 cache; here every observation needs its
    own K copies, and they must be interleaved so row b*K+j belongs to
    observation b -- the order the flattened interpolants are stacked in.
    """
    if K == 1:
        return cache
    for lst in (cache.key_cache, cache.value_cache):
        for i in range(len(lst)):
            if lst[i] is not None:
                lst[i] = lst[i].repeat_interleave(K, dim=0)
    return cache


def student_v(x, tau, obs, grad=True):
    """v at (E_V, E_B) with adapters ON, for one observation.

    Rebuilt from the primitives because ``be.velocity`` is under
    ``@torch.no_grad()``; checked against it at startup so the training target
    is the deployed function and not a lookalike.
    """
    return student_v_batch([obs], x.unsqueeze(0), tau.unsqueeze(0), grad)[0]


def student_v_batch(obs_list, x, tau, grad=True):
    """(B, K, H, D) velocities from B observations in one encoder+prefill pass."""
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
    B, K = x.shape[0], x.shape[1]
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, adapters_ctx(model, True), be._truncated(args.e_v, args.e_b):
        o = _batched_obs(obs_list)
        imgs, img_m, lang, lang_m, state = model._preprocess_observation(o, train=False)
        embs, pad_m, att_m = model.embed_prefix(imgs, img_m, lang, lang_m)
        att2d = make_att_2d_masks(pad_m, att_m)
        be._vlm.config._attn_implementation = "eager"
        _, pkv = be._pwe.forward(
            attention_mask=model._prepare_attention_masks_4d(att2d),
            position_ids=torch.cumsum(pad_m, dim=1) - 1,
            past_key_values=None, inputs_embeds=[embs, None], use_cache=True)
        pkv = _repeat_cache(pkv, K)
        v = model.denoise_step(
            state.repeat_interleave(K, dim=0),
            pad_m.repeat_interleave(K, dim=0),
            pkv,
            x.reshape(B * K, *x.shape[2:]).to(be.device, torch.float32),
            tau.reshape(B * K).to(be.device, torch.float32))
    return v.reshape(B, K, *v.shape[1:])


# -- the rebuilt forward must equal the deployed one, adapters off (they are
#    zero-initialised, so both should be the plain truncated model)
_o = obs_of(TRAIN[0])
_x, _ = batch_of(TRAIN[0], 4)
_t = torch.full((4,), TAU0)
with torch.no_grad():
    _mine = student_v(_x, _t, _o, grad=False).float().cpu()
    _theirs = be.velocity(A_tau=_x, tau=_t, obs=_o, E_V=args.e_v, E_B=args.e_b,
                          adapters=True).float().cpu()
_gap = float((_mine - _theirs).abs().max())
print(f"  rebuilt forward vs be.velocity: max abs gap {_gap:.2e}")
if _gap > 1e-3:
    sys.exit("the training forward is not the deployed forward -- refusing to "
             "train a student on a different function than the one evaluated")

# Batching changes the compute path, so it gets its own guard -- but the guard
# has to measure the right thing.  Two checks, because they catch different
# faults:
#
#   independence  the same observation duplicated across the batch must give
#                 rows that agree EXACTLY.  Any disagreement means the group is
#                 leaking -- a mask applied to the wrong row, a cache repeated
#                 in the wrong order -- and the loss would be computed against
#                 the wrong observation while looking perfectly healthy.
#
#   agreement     batched against sequential, in RELATIVE terms.  bf16 changes
#                 its reduction order with the batch width, so an exact match is
#                 not on offer: measured at 0.4% between K=1 and K=16, against a
#                 student error of 13.7%.  A max-abs threshold rejects that
#                 harmless noise by reading the few largest elements; the
#                 relative norm is the scale the training objective works in.
if args.obs_batch > 1:
    _oa, _ob = obs_of(TRAIN[0]), obs_of(TRAIN[1])
    _xa, _ = batch_of(TRAIN[0], 4)
    _xb, _ = batch_of(TRAIN[1], 4)
    _t4 = torch.full((4,), TAU0)
    with torch.no_grad():
        _dup = student_v_batch([_oa, _oa], torch.stack([_xa, _xa]),
                               torch.stack([_t4, _t4]), grad=False).float().cpu()
        _seq = torch.stack([student_v(_xa, _t4, _oa, grad=False),
                            student_v(_xb, _t4, _ob, grad=False)]).float().cpu()
        _bat = student_v_batch([_oa, _ob], torch.stack([_xa, _xb]),
                               torch.stack([_t4, _t4]), grad=False).float().cpu()
    _leak = float((_dup[0] - _dup[1]).abs().max())
    _rel = float((_seq - _bat).pow(2).sum().sqrt() / _seq.pow(2).sum().sqrt())
    print(f"  batching: row independence {_leak:.2e}   "
          f"rel vs sequential {_rel:.4f}")
    if _leak > 0:
        sys.exit("rows of a batch are not independent -- the same observation "
                 "duplicated gave different answers, so the group is mixing")
    if _rel > 0.02:
        sys.exit(f"batched forward differs from sequential by {_rel:.3f} "
                 "relative -- far past bf16 reduction-order noise (0.004)")

opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
steps_per_epoch = max(1, len(TRAIN) // args.accum)
total_steps = steps_per_epoch * args.epochs
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / max(args.warmup, 1))
    * (0.5 * (1 + math.cos(math.pi * min(1.0, s / max(total_steps, 1))))))
print(f"  {steps_per_epoch} steps/epoch x {args.epochs} = {total_steps} steps")


def spread_ratio(out_s, out_t):
    """How the K corrected chunks scatter, student against teacher.

    This is what the acceptance test reads.  Below ~0.7 the student has
    under-dispersed and the test will accept everything; truncation starts far
    ABOVE 1.0, so training should bring it down toward one and stop there.
    """
    def med(o):
        K = o.shape[0]
        return torch.stack([torch.linalg.vector_norm(o[a] - o[b], ord=2, dim=-1)
                            for a in range(K) for b in range(a + 1, K)]).median()
    return float(med(out_s) / max(float(med(out_t)), 1e-9))


@torch.no_grad()
def validate(n_obs):
    """Two rel L2 figures, because they answer different questions.

    ``rel`` covers the seven channels the robot reads and is the honest measure
    of how well the student corrects.  ``rel32`` covers all thirty-two and is
    reported only so this run can be compared with the earlier one, which was
    trained and scored unmasked -- its 0.093 is the diluted number.
    """
    num = den = cos = spr = 0.0
    num32 = den32 = 0.0
    for i in VAL[:n_obs]:
        o = obs_of(i)
        x, v_t = batch_of(i, args.k_train)
        v_s = student_v(x, torch.full((x.shape[0],), TAU0), o,
                        grad=False).float().cpu()
        num32 += float((v_s - v_t).pow(2).sum())
        den32 += float(v_t.pow(2).sum())
        num += float((v_s[..., :ACT] - v_t[..., :ACT]).pow(2).sum())
        den += float(v_t[..., :ACT].pow(2).sum())
        cos += float(torch.nn.functional.cosine_similarity(
            v_s.flatten(1), v_t.flatten(1), dim=1).mean())
        spr += spread_ratio((x - TAU0 * v_s)[:, :CMAX, :POSE],
                            (x - TAU0 * v_t)[:, :CMAX, :POSE])
    n = len(VAL[:n_obs])
    return (math.sqrt(num / max(den, 1e-12)), cos / n, spr / n,
            math.sqrt(num32 / max(den32, 1e-12)))


rel0, cos0, spr0, rel32_0 = validate(args.val_obs)
print(f"\n  FLOOR (adapters zero-init = plain truncation): "
      f"rel L2(7ch) {rel0:.3f}  [32ch {rel32_0:.3f}]  cosine {cos0:.3f}  "
      f"spread {spr0:.1f}x")
print(f"  loss masked to {ACT} channels: {'YES' if args.mask_channels else 'no'}")
print(f"  target: rel L2 well below {rel0:.3f}, spread down toward 1.0\n")

hist = [dict(step=0, rel=rel0, rel32=rel32_0, cos=cos0, spread=spr0)]
best = rel0
gen = torch.Generator().manual_seed(args.seed + 1)
step = 0
t0 = time.time()

for ep in range(args.epochs):
    order = [TRAIN[j] for j in torch.randperm(len(TRAIN), generator=gen).tolist()]
    run_loss, run_n = 0.0, 0
    for b in range(steps_per_epoch):
        opt.zero_grad(set_to_none=True)
        grp = order[b * args.accum:(b + 1) * args.accum]
        for g0 in range(0, len(grp), args.obs_batch):
            sub = grp[g0:g0 + args.obs_batch]
            obs_l = [obs_of(i) for i in sub]
            xv = [batch_of(i, args.k_train, gen) for i in sub]
            xb = torch.stack([q[0] for q in xv])
            vb = torch.stack([q[1] for q in xv])
            tb = torch.full((len(sub), args.k_train), TAU0)
            v_s = student_v_batch(obs_l, xb, tb)
            aa, bb = v_s.float(), vb.to(v_s.device).float()
            if args.mask_channels:
                aa, bb = aa[..., :ACT], bb[..., :ACT]
            loss = torch.nn.functional.mse_loss(aa, bb)
            (loss * len(sub) / args.accum).backward()
            run_loss += float(loss) * len(sub)
            run_n += len(sub)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        step += 1

        if step % args.val_every == 0 or step == total_steps:
            rel, cos, spr, rel32 = validate(args.val_obs)
            hist.append(dict(step=step, rel=rel, rel32=rel32, cos=cos,
                             spread=spr, train_mse=run_loss / max(run_n, 1)))
            flag = ("  <-- UNDER-DISPERSED, the acceptance test will accept "
                    "everything" if spr < 0.7 else "")
            print(f"  ep {ep+1} step {step:>5}/{total_steps}  "
                  f"train mse {run_loss/max(run_n,1):.4f}   "
                  f"val rel L2(7ch) {rel:.3f} ({100*(1-rel/rel0):+.0f}%)  "
                  f"[32ch {rel32:.3f}]   "
                  f"cos {cos:.3f}   spread {spr:.2f}x{flag}", flush=True)
            run_loss, run_n = 0.0, 0
            if rel < best:
                best = rel
                torch.save(dict(
                    lora={k: v.detach().cpu() for k, v in model.state_dict().items()
                          if "lora_" in k},
                    e_v=args.e_v, e_b=args.e_b, rank=args.rank,
                    rank_readout=args.rank_readout, tau0=TAU0,
                    step=step, rel=rel, cos=cos, spread=spr,
                    floor=dict(rel=rel0, cos=cos0, spread=spr0),
                ), f"{args.out}_{args.e_v}-{args.e_b}.pt")

with open(f"{args.out}_{args.e_v}-{args.e_b}_hist.json", "w") as f:
    json.dump(dict(args=vars(args), floor=hist[0], history=hist), f, indent=2)

r = hist[-1]
print(f"""
{'='*78}
floor  rel L2(7ch) {rel0:.3f}   cosine {cos0:.3f}   spread {spr0:.1f}x
final  rel L2(7ch) {r['rel']:.3f}   cosine {r['cos']:.3f}   spread {r['spread']:.2f}x
best   rel L2(7ch) {best:.3f}   saved to {args.out}_{args.e_v}-{args.e_b}.pt
{'='*78}
READ THIS BEFORE BELIEVING THE LOSS
  rel L2 falling is necessary and not sufficient.  The mechanism reads the
  disagreement between draws, so check that spread landed near 1.0.  Far above
  and the acceptance test rejects everything; far below and it accepts
  everything, at which point success collapses while this table still looks
  good.  The only verdict that counts is the N distribution and the success
  rate with the student plugged into the loop.
  [{time.time()-t0:.0f}s]""")
