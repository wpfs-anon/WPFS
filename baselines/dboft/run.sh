#!/bin/bash
source "$(cd "$(dirname "$0")/../.." && pwd)/pipelines/dboft/common.sh"
SERVER=$CORRECTOR_HOME/baselines/dboft/dboft_baseline_server.py
for m in ${@:-effvla aac vlacache specprune}; do
  case $m in
    effvla)    ARGS="--mode effvla --ev-interval 2" ;;
    aac)       ARGS="--mode aac --aac-n 5 --aac-move-th 0.5" ;;
    vlacache)  ARGS="--mode vlacache --vc-max-age 0" ;;
    specprune) ARGS="--mode specprune --sp-alpha 2.0 --sp-impl paper" ;;
    *) echo "unknown baseline $m"; exit 1 ;;
  esac
  for RNG in $SEEDS; do export RNG; eval_run ${m}_r$RNG $ARGS; done
done
"$PY_CLIENT" "$CORRECTOR_HOME/scripts/dboft_table.py" --seeds "${SEEDS// /,}" baseline=baseline \
    effvla=effvla aac=aac vlacache=vlacache specprune=specprune
