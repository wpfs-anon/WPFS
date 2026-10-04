#!/usr/bin/env python
import os
import pathlib
import subprocess
import sys

HOME = pathlib.Path(os.environ.get(
    "CORRECTOR_HOME", pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(HOME / "openpi" / "src"))
sys.path.insert(0, str(HOME / "openpi" / "packages" / "openpi-client" / "src"))

RAW = HOME / "openpi_assets" / "pi05_libero"
OUT = HOME / "openpi_assets" / "pi05_libero_pytorch"

from openpi.shared import download

if not (RAW / "params").exists():
    print("downloading gs://openpi-assets/checkpoints/pi05_libero")
    got = download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")
    RAW.parent.mkdir(parents=True, exist_ok=True)
    if pathlib.Path(got).resolve() != RAW.resolve():
        os.symlink(got, RAW)
print(f"checkpoint: {RAW}")

if (OUT / "model.safetensors").exists():
    print(f"already converted: {OUT}")
    sys.exit(0)
env = dict(os.environ, PYTHONPATH=f"{HOME}/openpi/src:{HOME}/openpi/packages/openpi-client/src")
subprocess.run([sys.executable, "examples/convert_jax_model_to_pytorch.py", "--config_name", "pi05_libero",
                "--checkpoint_dir", str(RAW), "--output_path", str(OUT)],
               cwd=HOME / "openpi", env=env, check=True)
print(f"converted -> {OUT}")
