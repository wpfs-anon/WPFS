#!/usr/bin/env python
"""Confirm the tree is complete and pi0 loads, before anything long runs.

    python setup/04_check.py
"""
import os
import pathlib
import sys

HOME = pathlib.Path(os.environ.get(
    "CORRECTOR_HOME", pathlib.Path(__file__).resolve().parent.parent))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
for p in ("openpi/src", "openpi/packages/openpi-client/src", "src", "scripts"):
    sys.path.insert(0, str(HOME / p))

ok = True
for label, path in (
        ("pi0 (pytorch)", HOME / "openpi_assets" / "pi0_libero_pytorch"),
        ("norm stats", HOME / "openpi_assets" / "pi0_libero" / "assets"),
        ("tokenizer", HOME / "assets" / "paligemma_tokenizer.model"),
        ("openpi", HOME / "openpi" / "src"),
        ("LIBERO", HOME / "LIBERO"),
        ("LoRA student", HOME / "checkpoints" / "final30_14-12.pt"),
        ("network student", HOME / "checkpoints" / "spnet_g0.pt")):
    good = path.exists()
    ok &= good
    print(f"  [{'ok ' if good else 'MISSING'}] {label:<18} {path}")

if not ok:
    sys.exit("\nrun setup/01..03 first")

from sentry.models.openpi_adapter import load_pi0_pytorch, OpenPiBackend
import torch

model = load_pi0_pytorch(str(HOME / "openpi_assets" / "pi0_libero_pytorch"),
                         device="cuda" if torch.cuda.is_available() else "cpu")
be = OpenPiBackend(model, device=str(next(model.parameters()).device), M=10,
                   attach_adapters=False)
print(f"\n  pi0 loaded: {be.L_V} prefix layers, {be.L_B} expert layers, "
      f"chunk {be.H} x {be.d_a}, {be.M} denoise steps")
print("  ready")
