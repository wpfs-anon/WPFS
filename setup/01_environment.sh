#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv (the venv here was built with it, not pip)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.cargo/bin:$PATH"
fi

uv venv --python 3.12 .venv
grep -viE "^(libero|openpi)==" requirements-frozen.txt > /tmp/req.txt
uv pip install --python .venv/bin/python -r /tmp/req.txt

echo
echo "environment ready:  $HERE/.venv/bin/python"
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
