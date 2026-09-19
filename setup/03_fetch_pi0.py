#!/usr/bin/env python
"""Download the pi0-LIBERO checkpoint and convert it to PyTorch.

    python setup/03_fetch_pi0.py

Two artifacts land under openpi_assets/:

  pi0_libero/          the released JAX checkpoint.  Only assets/norm_stats.json
                       is read at run time -- the 12 GB of params/ is needed
                       once, for the conversion, and can be deleted afterwards.
  pi0_libero_pytorch/  the converted weights every script actually loads.

The conversion runs on CPU and takes several minutes.  torch.compile is off:
on a one-shot weight conversion its autotune pass costs more than it saves.
"""
import os
import pathlib
import sys

HOME = pathlib.Path(os.environ.get(
    "CORRECTOR_HOME", pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(HOME / "openpi" / "src"))
sys.path.insert(0, str(HOME / "openpi" / "packages" / "openpi-client" / "src"))
sys.path.insert(0, str(HOME / "openpi" / "examples"))

RAW = HOME / "openpi_assets" / "pi0_libero"
OUT = HOME / "openpi_assets" / "pi0_libero_pytorch"
TOK = HOME / "assets" / "paligemma_tokenizer.model"

from openpi.shared import download                       # noqa: E402
from openpi.models.pi0_config import Pi0Config           # noqa: E402

if not (RAW / "params").exists():
    print("downloading gs://openpi-assets/checkpoints/pi0_libero (~12 GB)")
    got = download.maybe_download("gs://openpi-assets/checkpoints/pi0_libero")
    RAW.parent.mkdir(parents=True, exist_ok=True)
    if pathlib.Path(got).resolve() != RAW.resolve():
        os.symlink(got, RAW)
print(f"checkpoint: {RAW}  (assets: {(RAW / 'assets').exists()})")

if not TOK.exists():
    TOK.parent.mkdir(parents=True, exist_ok=True)
    got = download.maybe_download(
        "gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    pathlib.Path(got).replace(TOK)
print(f"tokenizer:  {TOK}")

if OUT.exists() and any(OUT.iterdir()):
    print(f"already converted: {OUT}")
    sys.exit(0)

import importlib.util                                    # noqa: E402
spec = importlib.util.spec_from_file_location(
    "convert_jax_model_to_pytorch",
    str(HOME / "openpi" / "examples" / "convert_jax_model_to_pytorch.py"))
conv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conv)

# pi0_libero is Pi0Config() with its defaults: action_dim=32,
# action_horizon=50, pi05=False, gemma_2b + gemma_300m.
conv.convert_pi0_checkpoint(str(RAW), "bfloat16", str(OUT), Pi0Config())
print(f"converted -> {OUT}")
