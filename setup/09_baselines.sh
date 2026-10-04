#!/bin/bash
set -eu
: "${CORRECTOR_HOME:=$(cd "$(dirname "$0")/.." && pwd)}"
AAC_REF=${AAC_REF:-ff8c8e421f7805b5845a72d07fbd8f1171df74bf}
T=$CORRECTOR_HOME/third_party/aac_libero
mkdir -p "$CORRECTOR_HOME/third_party"
[ -d "$T/.git" ] || git clone https://github.com/Adaptive-Action-Chunking/libero "$T"
git -C "$T" checkout -q "$AAC_REF"
test -f "$T/action_optimization/action_entropy_pi05.py"
touch "$T/action_optimization/__init__.py"
echo "AAC decision code @ $(git -C "$T" rev-parse --short HEAD)"
for py in "$CORRECTOR_HOME/.venv/bin/python" "${VENV_DEX:-$CORRECTOR_HOME/dboft/venv_dex}/bin/python"; do
    [ -x "$py" ] || continue
    uv pip install --quiet --python "$py" scipy matplotlib opencv-python-headless
    echo "  baseline deps installed for $py"
done
