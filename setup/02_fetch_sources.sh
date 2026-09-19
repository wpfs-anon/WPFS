#!/usr/bin/env bash
# Clone the two upstream repositories at the commits every result used, and
# install them into the environment.
#
#   bash setup/02_fetch_sources.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
PY="$HERE/.venv/bin/python"

OPENPI_COMMIT=215abfb          # openpi @ 2026-08-25
LIBERO_COMMIT=8f1084e          # LIBERO @ 2025-03-15

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

# LIBERO asks for a dataset path interactively on first import and blocks a
# non-interactive run; answering N writes the default config and is enough for
# rollouts, which need only bddl files and init states, not the demos.
echo "N" | "$PY" -c "import libero.libero" >/dev/null 2>&1 || true
"$PY" -c "
from libero.libero import get_libero_path
print('  bddl_files :', get_libero_path('bddl_files'))
print('  init_states:', get_libero_path('init_states'))
"
echo "sources ready"
