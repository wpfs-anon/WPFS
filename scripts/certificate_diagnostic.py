"""Does the certificate see anything?

The acceptance test rejects a chunk position when the K draws disagree there by
more than agree_tau x SREF.  That is only useful if disagreement actually marks
the positions where the corrector is wrong.  For the teacher it does -- that is
why SREF means something.  For a distilled student nobody has checked, and a
student whose draws scatter by the right AMOUNT in the wrong PLACES would look
perfect on every metric we have tracked while leaving the certificate blind.

Two numbers per checkpoint, both computed per chunk position:

  corr    Spearman between draw spread and error against the teacher.  If this
          is near zero the test cannot tell a good correction from a bad one.

  ratio   mean error of the positions the test REJECTS over the positions it
          ACCEPTS, at the operating threshold the driver uses.  This is the
          quantity the mechanism actually depends on: above 1 the test removes
          worse-than-average positions, at 1 it removes nothing useful.
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

import argparse, math, os, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(v, "8")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
for p in (f"{HOME}/openpi/src", f"{HOME}/openpi/packages/openpi-client/src",
          f"{HOME}/src", f"{HOME}/scripts"): sys.path.insert(0, p)
import cv2, numpy as np, torch
from sentry.core.types import Observation

ap = argparse.ArgumentParser()
ap.add_argument("--shard", default=f"{HOME}/distill/train_s101.pt,"
                                   f"{HOME}/distill/libero_spatial_s102.pt",
                help="scored on spatial only -- that is the eval suite")
ap.add_argument("--nets", default=f"{HOME}/distill/net120_512x6.pt,"
                                  f"{HOME}/distill/net100_512x6.pt,"
                                  f"{HOME}/distill/gnet_g4.pt")
ap.add_argument("--lora", default=f"{HOME}/distill/final30_14-12.pt")
ap.add_argument("--n-obs", type=int, default=128)
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--agree-tau", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=1234)
args = ap.parse_args()

POSE, CMAX, TAU0 = 6, 25, 0.5
dev = "cuda"

paths = [q.strip() for q in args.shard.split(",") if q.strip()]
parts = [torch.load(q, weights_only=False) for q in paths]
JA = [b for d in parts for b in d["jpg_a"]]
JW = [b for d in parts for b in d["jpg_w"]]
X = torch.cat([d["x"] for d in parts])
V = torch.cat([d["v"] for d in parts])
ST = torch.cat([d["state"] for d in parts]).to(torch.float32)
LG = torch.cat([d["lang"] for d in parts]).to(torch.int64)
SREF = parts[0]["sref"].float()
N, H, D = X.shape[0], X.shape[2], X.shape[3]
del parts
g = torch.Generator().manual_seed(args.seed)
IDX = torch.randperm(N, generator=g)[:args.n_obs].tolist()
print(f"  {N} spatial observations, scoring {len(IDX)}, K={args.k}, "
      f"agree_tau={args.agree_tau}")
print(f"  SREF[:5] = {[round(float(q),3) for q in SREF[:5]]}")


def unjpg(b): return cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8),
                                               cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def obs_of(i):
    f = np.stack([unjpg(JA[i]), unjpg(JW[i])]).astype(np.float32) / 127.5 - 1.0
    return Observation(images=torch.from_numpy(f).permute(0, 3, 1, 2).contiguous(),
                       language=LG[i], state=ST[i], t=0)


def spearman(a, b):
    ra = torch.argsort(torch.argsort(a)).float()
    rb = torch.argsort(torch.argsort(b)).float()
    ra = (ra - ra.mean()) / (ra.std() + 1e-9)
    rb = (rb - rb.mean()) / (rb.std() + 1e-9)
    return float((ra * rb).mean())


def per_position(out_s, out_t):
    """spread and error at each of the first CMAX positions, pose channels only."""
    K = out_s.shape[0]
    pairs = [(a, b) for a in range(K) for b in range(a + 1, K)]
    spread = torch.stack([torch.linalg.vector_norm(
        out_s[a, :CMAX, :POSE] - out_s[b, :CMAX, :POSE], dim=-1)
        for a, b in pairs]).median(dim=0).values          # (CMAX,)
    err = torch.linalg.vector_norm(
        out_s[:, :CMAX, :POSE].mean(0) - out_t[:, :CMAX, :POSE].mean(0),
        dim=-1)                                            # (CMAX,)
    return spread.cpu(), err.cpu()


def report(name, velocity_fn):
    S, E = [], []
    for i in IDX:
        x = X[i, :args.k].to(dev).float()
        v_t = V[i, :args.k].to(dev).float()
        v_s = velocity_fn(i, x)
        s, e = per_position(x - TAU0 * v_s, x - TAU0 * v_t)
        S.append(s); E.append(e)
    S = torch.cat(S); E = torch.cat(E)
    thr = (args.agree_tau * SREF[:CMAX]).repeat(len(IDX))
    rej = S > thr
    acc = ~rej
    ratio = (float(E[rej].mean()) / max(float(E[acc].mean()), 1e-9)
             if rej.any() and acc.any() else float("nan"))
    print(f"  {name:<22} corr {spearman(S, E):+.3f}   "
          f"reject/accept error ratio {ratio:5.2f}   "
          f"rejected {100*float(rej.float().mean()):4.1f}%   "
          f"mean err {float(E.mean()):.4f}")


print(f"\n  {'checkpoint':<22} {'corr':>10}   {'ratio':>28}   {'rejected':>12}   {'err':>12}")
print(f"  {'-'*22} {'-'*10}   {'-'*28}   {'-'*12}   {'-'*12}")

from sentry.models.openpi_adapter import load_pi0_pytorch
model = load_pi0_pytorch(f"{HOME}/openpi_assets/pi0_libero_pytorch", device=dev)
model.eval()

# --- the from-scratch nets
from sentry.models.openpi_adapter import OpenPiBackend
be_plain = OpenPiBackend(model, device=dev, M=10, attach_adapters=False)
from student_net import NetCorrector
for path in [q.strip() for q in args.nets.split(",") if q.strip()]:
    if not os.path.exists(path):
        print(f"  {os.path.basename(path):<22} MISSING"); continue
    nc = NetCorrector(path, model, be_plain, H, D, device=dev)
    mem_cache = {}

    def vel(i, x, nc=nc, mem_cache=mem_cache):
        if i not in mem_cache: mem_cache[i] = nc.memory(obs_of(i))
        return nc.velocity(x, torch.full((x.shape[0],), TAU0, device=dev),
                           mem_cache[i]).float()
    with torch.no_grad():
        report(os.path.basename(path).replace("_512x6.pt", "").replace(".pt", ""), vel)
    del nc
    torch.cuda.empty_cache()

# --- the LoRA student, the one config that reached parity in the loop
if args.lora and os.path.exists(args.lora):
    st = torch.load(args.lora, weights_only=False)
    from sentry.models.lora import set_adapter_slot
    be = OpenPiBackend(model, device=dev, M=10, E_V_max=int(st["e_v"]),
                       E_B_max=int(st["e_b"]), n_rungs=1,
                       lora_rank=int(st["rank"]),
                       lora_rank_readout=int(st["rank_readout"]),
                       lora_dropout=0.0, attach_adapters=True)
    model.load_state_dict(st["lora"], strict=False)
    set_adapter_slot(model, 0)
    model.eval()

    def vel_lora(i, x):
        return be.velocity(A_tau=x, tau=torch.full((x.shape[0],), TAU0, device=dev),
                           obs=obs_of(i), E_V=int(st["e_v"]), E_B=int(st["e_b"]),
                           adapters=True).float()
    with torch.no_grad():
        report("final30 (LoRA)", vel_lora)

print("\n  corr near 0 means the acceptance test cannot separate good corrections\n"
      "  from bad ones, whatever its spread ratio says.\n"
      "  ratio near 1 means rejecting on disagreement removes nothing useful.")
