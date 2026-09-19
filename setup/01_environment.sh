#!/usr/bin/env bash
# Build the python environment.  Everything else in setup/ assumes this ran.
#
#   bash setup/01_environment.sh
#
# The pins in requirements-frozen.txt are the exact versions every number in
# README.md was produced with.  torch is pinned to a CUDA 12.8 build; if your
# driver is older, install a matching torch first and then run the rest.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv (the venv here was built with it, not pip)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.cargo/bin:$PATH"
fi

uv venv --python 3.12 .venv
# LIBERO and openpi are installed from source in 03_; everything else is pinned.
grep -viE "^(libero|openpi)==" requirements-frozen.txt > /tmp/req.txt
uv pip install --python .venv/bin/python -r /tmp/req.txt

echo
echo "environment ready:  $HERE/.venv/bin/python"
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
