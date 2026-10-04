#!/usr/bin/env python
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

from openpi.shared import download
from openpi.models.pi0_config import Pi0Config

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

import importlib.util
spec = importlib.util.spec_from_file_location(
    "convert_jax_model_to_pytorch",
    str(HOME / "openpi" / "examples" / "convert_jax_model_to_pytorch.py"))
conv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conv)

conv.convert_pi0_checkpoint(str(RAW), "bfloat16", str(OUT), Pi0Config())
print(f"converted -> {OUT}")
