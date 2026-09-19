"""Plan and correction latency for a converted openpi LIBERO policy.

Writes the table eval_corrector.py reads through --latency-from:
    {"latency_ms": {"plan-plan": <ms>, "<L_V>-<L_B>": <ms>}}
plan-plan is one full plan at batch 1 (M Euler steps); the depth key is one
full-depth velocity call at batch K -- a teacher correction.

Timed the way the loop runs: denoise_step compiled (kept only if the chunk is
preserved), CUDA-synchronised wall clock over many repetitions on an idle GPU.
Run it on pi0 too.  Reproducing pi0's recorded 57.81 / 39.58 ms is what makes
the pi05 figures comparable with every speedup already reported.

    python latency_probe.py --model pi05 --out $CORRECTOR_HOME/ckpt/latency_pi05.json
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
import argparse, json, sys, time
for p in (f"{HOME}/openpi/src", f"{HOME}/openpi/packages/openpi-client/src",
          f"{HOME}/src", f"{HOME}/scripts"):
    sys.path.insert(0, p)
import torch
from sentry.core.types import Observation
from sentry.envs.libero_data import LiberoPrompt
from sentry.models.openpi_adapter import OpenPiBackend, load_pi0_pytorch

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["pi0", "pi05", "lerobot05"], default="pi0")
ap.add_argument("--reps", type=int, default=40)
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--compile", choices=["default", "off"], default="default")
ap.add_argument("--out", required=True)
ap.add_argument("--net", default="",
                help="also time this student checkpoint: memory (encoder) plus one "
                     "K-draw velocity, the figure 43_ takes as --net-latency")
args = ap.parse_args()

CONV = {"pi0": f"{HOME}/openpi_assets/pi0_libero_pytorch",
        "pi05": f"{HOME}/openpi_assets/pi05_libero_pytorch",
        "lerobot05": f"{HOME}/openpi_assets/lerobot_pi05_libero_pytorch"}[args.model]
TOK = f"{HOME}/assets/paligemma_tokenizer.model"

model = load_pi0_pytorch(CONV, device="cuda")
be = OpenPiBackend(model, device="cuda", M=10, attach_adapters=False)
H, D = be.H, be.d_a
print(f"  {args.model}: pi05={model.pi05}  H={H}  d_a={D}  "
      f"prompt tokens={be.max_token_len}  L_V={be.L_V}  L_B={be.L_B}")

# Cost does not depend on pixel values, but it does depend on sequence length,
# so the prompt is a real instruction padded exactly as the loop pads it.
tokens = LiberoPrompt(TOK, max_len=be.max_token_len)(
    "pick up the black bowl between the plate and the ramekin and place it on the plate")
g = torch.Generator().manual_seed(0)
obs = Observation(images=torch.rand(2, 3, 224, 224, generator=g) * 2 - 1,
                  language=tokens, state=torch.zeros(D), t=0)
noise = torch.randn(H, D, generator=g)
ref = be.plan(obs, noise=noise).float()

mode = None
if args.compile != "off":
    eager = model.denoise_step
    torch._dynamo.reset()
    model.denoise_step = torch.compile(eager, mode="default", dynamic=False)
    with torch.no_grad():
        d = float((be.plan(obs, noise=noise).float() - ref).norm(dim=-1).max())
    if d > 0.05:
        model.denoise_step = eager
        print(f"  compile rejected (chunk moved {d:.4f}); eager")
    else:
        mode = "default"
        print(f"  compile ok ({d:.1e})")

xk = torch.randn(args.k, H, D, generator=g)
tk = torch.full((args.k,), 0.5)


def timeit(fn, reps):
    for _ in range(5):                      # warm-up, and every compiled shape traced
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / reps


with torch.no_grad():
    l_plan = timeit(lambda: be.plan(obs, noise=noise), args.reps)
    l_corr = timeit(lambda: be.velocity(A_tau=xk, tau=tk, obs=obs, E_V=be.L_V,
                                        E_B=be.L_B, adapters=False), args.reps)
key = f"{be.L_V}-{be.L_B}"
print(f"  plan {l_plan:.2f} ms   correction (K={args.k}, full depth {key}) {l_corr:.2f} ms"
      f"   ratio {l_plan / l_corr:.2f}")
lat = {"plan-plan": l_plan, key: l_corr}
if args.net:
    from student_net import NetCorrector
    NET = NetCorrector(args.net, model, be, H, D, device="cuda")
    with torch.no_grad():
        pdim = getattr(NET.net, "plan_dim", 0)
        pf = torch.randn(H, pdim, device="cuda") if pdim else None
        lat["net"] = timeit(
            lambda: NET.velocity(xk, tk, NET.memory(obs, plan=pf), age=20), args.reps)
    print(f"  student {NET.describe()}:  {lat['net']:.2f} ms per correction (K={args.k})")
with open(args.out, "w") as f:
    json.dump(dict(model=args.model, horizon=H, k=args.k, reps=args.reps, compile=mode,
                   latency_ms=lat), f, indent=2)
print(f"  wrote {args.out}")
