"""The corrector conditioned on pi0's LANGUAGE-MODEL tokens, not raw SigLIP.

54_ gave the net the vision tower's output and nothing else, so everything the
Gemma stack computes -- binding the image to the instruction, spatial relations,
task context -- it had to rediscover from the harvest.  The LoRA student never
had to: it reads the same tokens after fourteen Gemma layers.  That gap is the
most likely reason a net with MORE data still lands at rel 0.155 where LoRA
reaches 0.116, and --gemma-layers closes as much of it as the latency budget
allows.

Two corrections to 54_ ride along, and they are corrections, not variables:

  * --shard-weight.  Pouring 20k libero_90 observations into a 12k spatial set
    cut spatial from 34% of training to 20% and cost 5 points in the loop while
    every val metric improved.  Weighting restores the deployment mix while
    keeping the extra diversity available.

  * --select-shards.  Choosing the checkpoint on aggregate val rel, when 62% of
    val was libero_90, selected for competence on tasks nobody evaluates.
    Selection now reads only the shards that match the evaluation suite.
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
import argparse, hashlib, math, os, sys, time
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(v, "8")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
for p in (f"{HOME}/openpi/src", f"{HOME}/openpi/packages/openpi-client/src",
          f"{HOME}/src", f"{HOME}/scripts"): sys.path.insert(0, p)
import cv2, numpy as np, torch, torch.nn as nn
import torch.nn.functional as Fn
from student_net import StudentNet, prefix_tokens

ap = argparse.ArgumentParser()
ap.add_argument("--shard", required=True)
ap.add_argument("--gemma-layers", type=int, default=4,
                help="prefix layers to run before the tokens become memory.  0 "
                     "reproduces 54_ (raw SigLIP); each layer costs ~1.3 ms per "
                     "correction and buys representation the net need not learn.")
ap.add_argument("--shard-weight", default="",
                help="comma list, one per shard, giving each shard's share of "
                     "every epoch.  Empty means proportional to size.")
ap.add_argument("--select-shards", default="",
                help="comma list of shard indices whose val observations decide "
                     "which checkpoint is kept.  Empty means all of them.")
ap.add_argument("--cache", default=f"{HOME}/distill/gcache")
ap.add_argument("--out", default=f"{HOME}/distill/gnet")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--val-frac", type=float, default=0.05)
ap.add_argument("--val-obs", type=int, default=64)
ap.add_argument("--k-train", type=int, default=8)
ap.add_argument("--obs-batch", type=int, default=16)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--d-model", type=int, default=512)
ap.add_argument("--blocks", type=int, default=6)
ap.add_argument("--heads", type=int, default=8)
ap.add_argument("--ffn-mult", type=int, default=4)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--warmup", type=int, default=500)
ap.add_argument("--val-every", type=int, default=2000)
ap.add_argument("--grip-weight", type=float, default=1.0,
                help="weight on the gripper channel in the velocity loss.  The "
                     "plain mean spreads itself over 10 positions and 32 padded "
                     "dimensions, while success turns on this one channel: "
                     "against the teacher the student is twice as ambiguous "
                     "about the gripper (0.049 to 0.027) and 4.5x at p90.")
ap.add_argument("--lambda-grip", type=float, default=0.0,
                help="weight on matching the teacher's draw-to-draw gripper "
                     "disagreement.  --lambda-spread does this for the six pose "
                     "channels only, which is why nothing so far has taught the "
                     "student the channel its certificate keeps failing on.")
ap.add_argument("--pos-tail", type=float, default=1.0,
                help="weight of the last position relative to the first, linearly "
                     "interpolated.  Only the leading ~7 actions of a chunk are "
                     "executed before the next correction replaces the tail.")
ap.add_argument("--feat-cache", default="",
                help="teacher hidden states from harvest_feats.py, (N, KH, H, W) "
                     "float16.  With --lambda-feat the student also learns to "
                     "reproduce them through a head that is thrown away at "
                     "deployment, so a correction costs exactly what it did.")
ap.add_argument("--lambda-feat", type=float, default=0.0,
                help="weight on the feature term, against a velocity term that "
                     "runs about 0.03 and a feature term that starts near 1.0 "
                     "because the targets are normalised to unit RMS.")
ap.add_argument("--teacher-head", default="",
                help="the teacher's action_out_proj (1024 -> 32) saved as a .pt, "
                     "frozen, used as the student's last layer.  The student then "
                     "predicts the teacher's hidden state and the velocity is read "
                     "off it exactly as the teacher reads it, so the feature and "
                     "velocity losses compose instead of competing.")
ap.add_argument("--use-plan", action="store_true",
                help="feed the teacher's plan-time hidden state, stored by the "
                     "harvest, as extra memory tokens.")
ap.add_argument("--use-age", action="store_true",
                help="feed the age of the plan in steps.  Under L50 most "
                     "corrections run on padding and the student could not tell.")
ap.add_argument("--init-from", default="",
                help="start from this checkpoint instead of from scratch.  A DAgger "
                     "round can then train on the new round alone -- the union "
                     "would need a 232 GB memory cache on a 188 GB machine -- and "
                     "keep the earlier rounds in the weights.")
ap.add_argument("--save-every", type=int, default=0,
                help="also keep a checkpoint every N steps, named _s<step>, for a "
                     "real evaluation to choose between.")
ap.add_argument("--mem-noise", type=float, default=0.0,
                help="train under Gaussian noise on the memory, this multiple of "
                     "its RMS.  The students memorise (val rel about 2x train rel); "
                     "a JPEG round trip already moves this memory by 0.05 in the "
                     "same units (jpeg_gap.py), so noise of that size is jitter the "
                     "encoder really produces, not an arbitrary regulariser.")
ap.add_argument("--lambda-spread", type=float, default=0.0,
                help="weight on matching the teacher's per-position "
                     "draw disagreement.  The velocity loss constrains "
                     "nothing about it, yet that disagreement is the "
                     "only quantity the acceptance test reads.")
ap.add_argument("--build-cache-only", action="store_true")
ap.add_argument("--model", choices=["pi0", "pi05"], default="pi0",
                help="whose frozen encoder builds the memory cache -- it must be the "
                     "teacher the shards were labelled by and the policy the student "
                     "will be deployed beside.")
args = ap.parse_args()
CONV = {"pi0": f"{HOME}/openpi_assets/pi0_libero_pytorch",
        "pi05": f"{HOME}/openpi_assets/pi05_libero_pytorch"}[args.model]

POSE, CMAX, ACT = 6, 25, 7
dev = "cuda"

paths = [q.strip() for q in args.shard.split(",") if q.strip()]
parts = [torch.load(q, weights_only=False) for q in paths]
TAU0 = float(parts[0]["tau0"])
JA = [b for d in parts for b in d["jpg_a"]]
JW = [b for d in parts for b in d["jpg_w"]]
SHARD_OF = torch.tensor([i for i, d in enumerate(parts) for _ in range(d["x"].shape[0])])
X = torch.cat([d["x"] for d in parts])
V = torch.cat([d["v"] for d in parts])
ST = torch.cat([d["state"] for d in parts]).to(torch.float32)
LG = torch.cat([d["lang"] for d in parts]).to(torch.int64)
KH = parts[0]["k_harvest"]
PLAN = (torch.cat([d["plan_feat"] for d in parts]) if "plan_feat" in parts[0] else None)
AGE = (torch.cat([d["age"] for d in parts]) if "age" in parts[0] else None)
USE_PLAN = bool(args.use_plan and PLAN is not None)
USE_AGE = bool(args.use_age and AGE is not None)
PLAN_T = (PLAN.shape[1] if USE_PLAN else 0)
if args.use_plan and PLAN is None:
    sys.exit("--use-plan but these shards carry no plan_feat; harvest with harvest_dagger.py")
if args.use_age and AGE is None:
    sys.exit("--use-age but these shards carry no age")
sizes = [d["x"].shape[0] for d in parts]
del parts
N, H, D = X.shape[0], X.shape[2], X.shape[3]
S_DIM = ST.shape[1]
print(f"  {len(paths)} shard(s): {N} obs x {KH} interpolants, chunk {H}x{D}, "
      f"tau0={TAU0}, gemma_layers={args.gemma_layers}")

g = torch.Generator().manual_seed(args.seed)
perm = torch.randperm(N, generator=g).tolist()
n_val = max(16, int(args.val_frac * N))
VAL, TRAIN = perm[:n_val], perm[n_val:]
SEL = ([int(q) for q in args.select_shards.split(",") if q.strip() != ""]
       if args.select_shards else list(range(len(paths))))
print(f"  {len(TRAIN)} train / {len(VAL)} val;  checkpoint selected on shards "
      f"{SEL} -> {', '.join(os.path.basename(paths[i]) for i in SEL)}")

# per-epoch sampling weights
if args.shard_weight:
    w = [float(q) for q in args.shard_weight.split(",")]
    if len(w) != len(paths):
        sys.exit(f"--shard-weight has {len(w)} entries for {len(paths)} shards")
else:
    w = [s / N for s in sizes]
w = [q / sum(w) for q in w]
TR_BY_SHARD = [[i for i in TRAIN if int(SHARD_OF[i]) == s] for s in range(len(paths))]
print("  epoch mix: " + "  ".join(
    f"{os.path.basename(paths[s])[:18]} {100*w[s]:.0f}% ({len(TR_BY_SHARD[s])} obs)"
    for s in range(len(paths))))

# ------------------------------------------------------------------ cache
def unjpg(b): return cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8),
                                               cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

# pi0 tags are left exactly as they were, so existing caches stay valid
_tag = hashlib.md5(("|".join(paths) + f"|g{args.gemma_layers}"
                    + ("" if args.model == "pi0" else f"|{args.model}")).encode()).hexdigest()[:8]
C_MEM = f"{args.cache}_{_tag}_mem.npy"


def build_cache():
    from sentry.models.openpi_adapter import OpenPiBackend, load_pi0_pytorch
    model = load_pi0_pytorch(CONV, device=dev)
    be = OpenPiBackend(model, device=dev, M=10, attach_adapters=False)
    model.eval()
    B = 16
    with torch.no_grad():
        ims, msk = {}, {}
        ref = torch.zeros(1, 3, 224, 224, device=dev)
        for j, key in enumerate(be.image_keys):
            ims[key] = ref.clone(); msk[key] = torch.tensor([j < 2], device=dev)
        probe = prefix_tokens(model, be, ims, msk, LG[:1].to(dev), S_DIM, dev, args.gemma_layers)
    T, C = probe.shape[1], probe.shape[2]
    print(f"  memory tokens {T} x {C};  cache {N*T*C*2/1e9:.1f} GB -> {C_MEM}")
    # built under a temporary name and renamed only once flushed, so a build cut
    # short (hard lock, kill) is never mistaken for a finished cache on a rerun
    part = C_MEM + ".part"
    f = np.lib.format.open_memmap(part, mode="w+", dtype=np.float16, shape=(N, T, C))
    t0 = time.perf_counter()
    with torch.no_grad():
        for s in range(0, N, B):
            e = min(s + B, N)
            imgs = []
            for lst in (JA, JW):
                arr = np.stack([unjpg(lst[i]) for i in range(s, e)])
                t = torch.from_numpy(arr.astype(np.float32) / 127.5 - 1.0)
                imgs.append(t.permute(0, 3, 1, 2).contiguous().to(dev))
            ims, msk = {}, {}
            for j, key in enumerate(be.image_keys):
                ims[key] = imgs[j] if j < 2 else torch.zeros_like(imgs[0])
                msk[key] = torch.tensor([j < 2] * (e - s), device=dev)
            f[s:e] = prefix_tokens(model, be, ims, msk, LG[s:e].to(dev), S_DIM,
                                   dev, args.gemma_layers
                                   ).to(torch.float16).cpu().numpy()
            if s % (B * 100) == 0:
                el = time.perf_counter() - t0
                print(f"    {e}/{N}  {el/60:.1f}m, {el/max(e,1)*(N-e)/60:.1f}m left",
                      flush=True)
    f.flush(); del model, be, f
    os.replace(part, C_MEM)
    torch.cuda.empty_cache()


need = not os.path.exists(C_MEM)
if not need:
    a = np.load(C_MEM, mmap_mode="r")
    need = a.shape[0] != N
    del a
if need: build_cache()
MEM = np.load(C_MEM, mmap_mode="r")
T_MEM, C_TOK = MEM.shape[1], MEM.shape[2]
print(f"  memory cache: {MEM.shape}")
FEAT = FEAT_W = FEAT_RMS = None
if args.feat_cache:
    FEAT = np.load(args.feat_cache, mmap_mode="r")
    assert FEAT.shape[0] == N, (FEAT.shape, N)
    FEAT_W = FEAT.shape[-1]
    sample = np.asarray(FEAT[:64], dtype=np.float32)
    FEAT_RMS = float(np.sqrt((sample ** 2).mean()))
    print(f"  teacher features: {FEAT.shape}, rms {FEAT_RMS:.3f}")
if args.build_cache_only: sys.exit(0)

# ------------------------------------------------------------------ model
class Net(nn.Module):
    """StudentNet with one memory stream instead of image+language+state."""

    def __init__(self):
        super().__init__()
        self.inner = StudentNet(args.d_model, args.blocks, args.heads,
                                args.ffn_mult, C_TOK, S_DIM, H, D,
                                feat_dim=(FEAT_W if args.teacher_head else 0),
                                plan_dim=(PLAN.shape[-1] if USE_PLAN else 0),
                                use_age=USE_AGE)
        if args.teacher_head:
            _hd = torch.load(args.teacher_head, map_location="cpu", weights_only=False)
            self.inner.head_w.copy_(_hd["weight"].float())
            self.inner.head_b.copy_(_hd["bias"].float())
        # trained, then discarded: it never runs in the loop
        # LayerNorm first: h is the raw block output, tens of times larger than
        # the unit-RMS targets, which put 97% of the gradient on this term
        self.feat_head = (nn.Sequential(nn.LayerNorm(args.d_model),
                                        nn.Linear(args.d_model, FEAT_W))
                          if FEAT_W else None)

    def memory(self, tok, state, plan=None):
        b = self.inner
        m = [b.img_in(tok) + b.mem_type[0], b.st_in(state)[:, None] + b.mem_type[2]]
        if b.plan_dim and plan is not None:
            m.append(b.plan_in(plan[None] if plan.dim() == 2 else plan) + b.plan_type)
        return torch.cat(m, dim=1)

    def forward(self, x, tau, mem, age=None):
        return self.inner(x, tau, mem, age)

    def forward_feat(self, x, tau, mem, age=None):
        """velocity and the feature it was read off."""
        if self.inner.feat_dim:
            return self.inner.forward_feat(x, tau, mem, age)
        h = self.inner.hidden(x, tau, mem, age)
        return self.inner.out(h), self.feat_head(h)


# The loop executes the leading positions of a chunk and lives or dies on the
# gripper; a flat mean over positions and padded dimensions trains neither.
POS_W = torch.linspace(1.0, args.pos_tail, H).view(1, H, 1)
DIM_W = torch.ones(D)
DIM_W[POSE] = args.grip_weight
LOSS_W = (POS_W * DIM_W.view(1, 1, D)).to(dev)
LOSS_W = LOSS_W / LOSS_W.mean()

net = Net().to(dev)
if args.init_from:
    _ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
    net.inner.load_state_dict(_ck["net"], strict=True)
    print(f"  warm start from {os.path.basename(args.init_from)} "
          f"(step {_ck['step']}, rel {_ck['rel']:.3f}, spread {_ck['spread']:.2f}x)")
print(f"  student: d={args.d_model} x {args.blocks} blocks, "
      f"{sum(p.numel() for p in net.parameters())/1e6:.1f}M params, "
      f"memory {T_MEM + 1} tokens")
nn.init.zeros_(net.inner.out[1].weight); nn.init.zeros_(net.inner.out[1].bias)

opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
spe = max(1, len(TRAIN) // args.obs_batch)
total = spe * args.epochs
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / max(args.warmup, 1))
    * (0.5 * (1 + math.cos(math.pi * min(1.0, s / max(total, 1))))))
print(f"  {spe} steps/epoch x {args.epochs} = {total} steps")


def mem_of(ids):
    tok = torch.from_numpy(np.ascontiguousarray(MEM[ids])).to(dev).float()
    plan = PLAN[ids].to(dev).float() if USE_PLAN else None
    return net.memory(tok, ST[ids].to(dev), plan)


def age_of(ids, k):
    """one age per observation, repeated over its k interpolants"""
    return (AGE[ids].to(dev).float().repeat_interleave(k) if USE_AGE else None)


def pair_spread(o):
    """(B, K, C, P) -> (B, C): mean distance between draws at each position.

    The rollout test reads the MEDIAN pairwise distance; the mean is used here
    because it carries a gradient, and the two agree closely at K=8.
    """
    d = torch.cdist(o.transpose(1, 2), o.transpose(1, 2))
    K = o.shape[1]
    return d.sum(dim=(-1, -2)) / (K * (K - 1))


def spread_ratio(o_s, o_t):
    def med(o):
        K = o.shape[0]
        return torch.stack([torch.linalg.vector_norm(o[a] - o[b], ord=2, dim=-1)
                            for a in range(K) for b in range(a + 1, K)]).median()
    return float(med(o_s) / max(float(med(o_t)), 1e-9))


@torch.no_grad()
def validate(n_obs):
    """rel and spread per shard, because one aggregate hid a 5-point regression."""
    net.eval()
    acc = {s: [0.0, 0.0, 0.0, 0] for s in range(len(paths))}
    # n_obs counts observations from the SELECTED shards.  A flat prefix of VAL
    # meant that when libero_90 was 62% of the data the checkpoint was chosen
    # on about two dozen samples.
    sel_ids = [i for i in VAL if int(SHARD_OF[i]) in SEL][:n_obs]
    oth_ids = [i for i in VAL if int(SHARD_OF[i]) not in SEL][:max(16, n_obs // 4)]
    for i in sel_ids + oth_ids:
        x = X[i, :args.k_train].to(dev).float()
        v_t = V[i, :args.k_train].to(dev).float()
        K = x.shape[0]
        v_s = net(x, torch.full((K,), TAU0, device=dev),
                  mem_of([i]).expand(K, -1, -1), age_of([i], K)).float()
        a = acc[int(SHARD_OF[i])]
        a[0] += float((v_s[..., :ACT] - v_t[..., :ACT]).pow(2).sum())
        a[1] += float(v_t[..., :ACT].pow(2).sum())
        a[2] += spread_ratio((x - TAU0 * v_s)[:, :CMAX, :POSE],
                             (x - TAU0 * v_t)[:, :CMAX, :POSE])
        a[3] += 1
    per = {s: (math.sqrt(a[0] / max(a[1], 1e-12)), a[2] / max(a[3], 1), a[3])
           for s, a in acc.items() if a[3] > 0}
    num = sum(acc[s][0] for s in SEL if acc[s][3])
    den = sum(acc[s][1] for s in SEL if acc[s][3])
    spr = sum(acc[s][2] for s in SEL if acc[s][3])
    cnt = sum(acc[s][3] for s in SEL if acc[s][3])
    net.train()
    return math.sqrt(num / max(den, 1e-12)), spr / max(cnt, 1), per


def ckpt_dict(step, rel, spr, per):
    return dict(net=net.inner.state_dict(), d_model=args.d_model,
                blocks=args.blocks, heads=args.heads,
                ffn_mult=args.ffn_mult, c_tok=C_TOK, s_dim=S_DIM,
                gemma_layers=args.gemma_layers, mem_tokens=T_MEM + 1 + PLAN_T,
                plan_dim=(PLAN.shape[-1] if USE_PLAN else 0), use_age=USE_AGE,
                lambda_spread=args.lambda_spread, tau0=TAU0,
                mem_noise=args.mem_noise, grip_weight=args.grip_weight,
                lambda_feat=args.lambda_feat,
                feat_dim=(FEAT_W if args.teacher_head else 0),
                lambda_grip=args.lambda_grip, pos_tail=args.pos_tail,
                model=args.model,
                step=step, rel=rel, spread=spr, per_shard=per)


rel0, spr0, per0 = validate(args.val_obs)
print(f"\n  FLOOR: rel(sel) {rel0:.3f}  spread(sel) {spr0:.1f}x")
print("  targets: rel <= 0.12, spread ~1.0   "
      "(LoRA 0.116/0.99; net120 0.155/1.10 gave -2 pts at 3.05x)\n")

gen = torch.Generator().manual_seed(args.seed)
best, step, t0 = 1e9, 0, time.perf_counter()
for ep in range(1, args.epochs + 1):
    order = []
    for s in range(len(paths)):
        k = int(round(w[s] * len(TRAIN)))
        pool = TR_BY_SHARD[s]
        if not pool: continue
        idx = torch.randint(0, len(pool), (k,), generator=gen).tolist()
        order += [pool[j] for j in idx]
    order = [order[j] for j in torch.randperm(len(order), generator=gen).tolist()]
    for b in range(spe):
        ids = order[b * args.obs_batch:(b + 1) * args.obs_batch]
        if not ids: continue
        sel = torch.randint(0, KH, (len(ids), args.k_train), generator=gen)
        x = torch.stack([X[i][sel[j]] for j, i in enumerate(ids)]).to(dev).float()
        vt = torch.stack([V[i][sel[j]] for j, i in enumerate(ids)]).to(dev).float()
        B_, K = x.shape[0], x.shape[1]
        mem = mem_of(ids)
        if args.mem_noise > 0:
            # one draw per observation, not per interpolant: at deployment every
            # interpolant of a correction shares the one memory
            mem = mem + args.mem_noise * mem.pow(2).mean().sqrt() * torch.randn_like(mem)
        mem = mem.repeat_interleave(K, dim=0)
        opt.zero_grad(set_to_none=True)
        ag = age_of(ids, K)
        if FEAT is not None:
            pred, fpred = net.forward_feat(x.reshape(B_ * K, H, D),
                                           torch.full((B_ * K,), TAU0, device=dev), mem, ag)
        else:
            pred = net(x.reshape(B_ * K, H, D),
                       torch.full((B_ * K,), TAU0, device=dev), mem, ag)
        tgt = vt.reshape(B_ * K, H, D)
        loss_v = ((pred - tgt).pow(2) * LOSS_W).mean()
        loss, loss_s = loss_v, torch.zeros((), device=dev)
        loss_g = loss_f = torch.zeros((), device=dev)
        if FEAT is not None and args.lambda_feat > 0:
            ft = torch.from_numpy(np.stack(
                [np.asarray(FEAT[i][sel[j]], dtype=np.float32)
                 for j, i in enumerate(ids)])).to(dev)
            # chained: the frozen head expects the teacher's own scale, so match
            # raw features and divide the loss instead, to keep the number
            # comparable with the side-target runs (1.0 = predicting zero)
            if not args.teacher_head:
                ft = ft / FEAT_RMS
            loss_f = Fn.mse_loss(fpred, ft.reshape(B_ * K, H, FEAT_W))
            if args.teacher_head:
                loss_f = loss_f / (FEAT_RMS ** 2)
            loss = loss + args.lambda_feat * loss_f
        if args.lambda_spread > 0:
            o_s = (x - TAU0 * pred.reshape(B_, K, H, D))[:, :, :CMAX, :POSE]
            o_t = (x - TAU0 * tgt.reshape(B_, K, H, D))[:, :, :CMAX, :POSE]
            loss_s = Fn.mse_loss(pair_spread(o_s), pair_spread(o_t))
            loss = loss_v + args.lambda_spread * loss_s
        if args.lambda_grip > 0:
            # the same test the certificate applies to the gripper: do the draws
            # agree?  pair_spread on one channel is exactly that disagreement
            g_s = (x - TAU0 * pred.reshape(B_, K, H, D))[:, :, :CMAX, POSE:POSE + 1]
            g_t = (x - TAU0 * tgt.reshape(B_, K, H, D))[:, :, :CMAX, POSE:POSE + 1]
            loss_g = Fn.mse_loss(pair_spread(g_s), pair_spread(g_t))
            loss = loss + args.lambda_grip * loss_g
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step(); step += 1
        if step % args.val_every == 0 or step == total:
            rel, spr, per = validate(args.val_obs)
            detail = "  ".join(f"s{s}:{r:.3f}/{sp:.2f}" for s, (r, sp, _) in
                               sorted(per.items()))
            print(f"  ep {ep:>3} step {step}/{total}  mse {float(loss_v):.4f}"
                  + (f" sprd {float(loss_s):.4f}" if args.lambda_spread > 0 else "")
                  + (f" grip {float(loss_g):.4f}" if args.lambda_grip > 0 else "")
                  + (f" feat {float(loss_f):.4f}" if args.lambda_feat > 0 else "")
                  + f"  "
                  f"SEL rel {rel:.3f} spread {spr:.2f}x   [{detail}]  "
                  f"[{(time.perf_counter()-t0)/60:.0f}m]", flush=True)
            if args.save_every and step % args.save_every == 0:
                torch.save(ckpt_dict(step, rel, spr, per),
                           f"{args.out}_g{args.gemma_layers}_s{step}.pt")
            if rel < best:
                best = rel
                torch.save(ckpt_dict(step, rel, spr, per),
                           f"{args.out}_g{args.gemma_layers}.pt")
print(f"\n  best SEL rel {best:.3f} -> {args.out}_g{args.gemma_layers}.pt")
