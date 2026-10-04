#!/bin/bash
set -eu
: "${CORRECTOR_HOME:=$(cd "$(dirname "$0")/.." && pwd)}"
D=$CORRECTOR_HOME/dboft
VENV_DEX=${VENV_DEX:-$D/venv_dex}
VENV_CLIENT=${VENV_CLIENT:-$D/venv_client}
DEXBOTIC_REF=${DEXBOTIC_REF:-main}
BENCHMARK_REF=${BENCHMARK_REF:-main}
SIMPLER_REF=${SIMPLER_REF:-main}
mkdir -p "$D" "$D/bin" "$D/logs" "$D/hf"
export HF_HOME=$D/hf

echo "[1/6] system packages (SAPIEN renders through Vulkan)"
if [ "$(id -u)" = 0 ] && command -v apt-get >/dev/null; then
    apt-get update -qq
    apt-get install -y -qq vulkan-tools libglew-dev libosmesa6-dev git-lfs >/dev/null
else
    echo "  not root: make sure vulkan-tools, libglew-dev, libosmesa6-dev and git-lfs are installed"
fi

echo "[2/6] Vulkan ICD"
if [ ! -f /usr/share/vulkan/icd.d/nvidia_icd.json ] && [ -f /etc/vulkan/icd.d/nvidia_icd.json ]; then
    mkdir -p /usr/share/vulkan/icd.d
    ln -sf /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
fi
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json XDG_RUNTIME_DIR=/tmp
vulkaninfo --summary 2>/dev/null | grep -m1 deviceName || echo "  WARNING: vulkaninfo sees no GPU"

echo "[3/6] upstream sources"
fetch() {
    local url=$1 dir=$2 ref=$3
    [ -d "$dir/.git" ] || git clone --recurse-submodules "$url" "$dir"
    git -C "$dir" fetch -q origin
    git -C "$dir" checkout -q "$ref"
    git -C "$dir" submodule update -q --init --recursive
    echo "  $(basename "$dir") @ $(git -C "$dir" rev-parse --short HEAD)"
}
fetch https://github.com/Dexmal/dexbotic "$D/dexbotic" "$DEXBOTIC_REF"
fetch https://github.com/Dexmal/dexbotic-benchmark "$D/dexbotic-benchmark" "$BENCHMARK_REF"
fetch https://github.com/simpler-env/SimplerEnv "$D/SimplerEnv" "$SIMPLER_REF"

echo "[4/6] server environment (dexbotic)"
uv venv --python 3.10 "$VENV_DEX"
source "$VENV_DEX/bin/activate"
uv pip install --quiet "torch==2.7.1" "torchvision==0.22.1" --index-url https://download.pytorch.org/whl/cu128
uv pip install --quiet -e "$D/dexbotic"
uv pip install --quiet "pyarrow<15" "transformers==4.57.6" huggingface_hub diffusers
uv pip uninstall --quiet kernels || true
python - <<'PY'
import numpy as np, torch, transformers, site, os, io
print("  server:", torch.__version__, transformers.__version__, np.__version__, torch.cuda.is_available())
p = os.path.join(site.getsitepackages()[0], "nptyping", "typing_.py")
if os.path.exists(p):
    s = io.open(p).read()
    for a, b in {"np.bool8": "np.bool_", "np.object0": "np.object_", "np.int0": "np.intp",
                 "np.uint0": "np.uintp", "np.str0": "np.str_", "np.bytes0": "np.bytes_",
                 "np.void0": "np.void"}.items():
        s = s.replace(a, b)
    io.open(p, "w").write(s)
PY
echo "[5/6] DB-OFT checkpoint (Dexmal/simpler-db-oft) into $HF_HOME"
python -c "from huggingface_hub import snapshot_download; print('  ', snapshot_download('Dexmal/simpler-db-oft', max_workers=8))"
deactivate

echo "[6/6] client environment (SimplerEnv)"
uv venv --python 3.10 "$VENV_CLIENT"
source "$VENV_CLIENT/bin/activate"
uv pip install --quiet -e "$D/SimplerEnv" -e "$D/SimplerEnv/ManiSkill2_real2sim"
uv pip install --quiet "numpy==1.24.4" "sapien==2.2.2" "setuptools<70" omegaconf hydra-core websockets msgpack requests pyyaml imageio-ffmpeg transforms3d
ln -sf "$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" "$D/bin/ffmpeg"
python -c "import sapien, numpy, mani_skill2_real2sim; print('  client:', sapien.__version__, numpy.__version__)"
deactivate

echo "done.  Next: python setup/06_patch_new_scenes.py; python setup/07_patch_eval_robust.py; python setup/08_dboft_client.py"
