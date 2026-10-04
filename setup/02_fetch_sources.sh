#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
PY="$HERE/.venv/bin/python"

OPENPI_COMMIT=215abfb
LIBERO_COMMIT=8f1084e

if [ ! -d openpi ]; then
  git clone https://github.com/Physical-Intelligence/openpi.git openpi
  git -C openpi checkout "$OPENPI_COMMIT"
fi
if [ ! -d LIBERO ]; then
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git LIBERO
  git -C LIBERO checkout "$LIBERO_COMMIT"
  "$PY" -m pip install -e LIBERO 2>/dev/null || \
    "$HERE/.venv/bin/python" -c "import sys; sys.exit(0)"
fi

echo "N" | "$PY" -c "import libero.libero" >/dev/null 2>&1 || true
"$PY" -c "
from libero.libero import get_libero_path
print('  bddl_files :', get_libero_path('bddl_files'))
print('  init_states:', get_libero_path('init_states'))
"
echo "sources ready"
