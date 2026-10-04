#!/bin/bash
set -u
MODEL=${1:?usage: run.sh pi0|pi05 [sp vc ev aac]}; shift
ROOT="${CORRECTOR_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${CPUS:+taskset -c $CPUS }$ROOT/.venv/bin/python"
OUT=$ROOT/ckpt/$MODEL
L=$ROOT/logs/baselines
SUITES=${SUITES:-"libero_spatial libero_object libero_goal libero_10"}
export JAX_PLATFORMS=cpu MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=1
mkdir -p "$OUT" "$L"
cd "$ROOT/baselines/libero"

if [ "$MODEL" = pi0 ]; then
  LAT=$ROOT/data/latency
  SP="--alpha 2.0 --global-from expert --capture-every 1 --latency-from $LAT/pi0_specprune.json"
  VC="--max-age 1 --latency-from $LAT/pi0_vlacache.json"
  EV="--n-prune 0 --k-final 512 --cache-interval 3 --latency-from $LAT/pi0_efficientvla.json"
  AAC="--n 5 --move-th 10 --latency-from $LAT/pi0_aac.json"
else
  SP="--alpha 2.0 --global-from expert --capture-every 1"
  VC="--max-age 1"
  EV="--n-prune 0 --k-final 512 --cache-interval 3"
  AAC="--n 5 --move-th 5.0"
fi

cell () {
  local tag=$1 script=$2 cfg=$3 suite=$4; shift 4
  local f=$OUT/$suite/${tag}_s7.json
  mkdir -p "$OUT/$suite"
  if [ -s "$f" ]; then echo "  skip $MODEL $suite $tag"; return 0; fi
  echo "== $MODEL $suite $tag  [$(date +%H:%M)]"
  $PY -u $script --model $MODEL --suite $suite --trials 50 --seed 7 --configs $cfg --out "$f" "$@" \
    > $L/${MODEL}_${suite}_${tag}.log 2>&1 || { echo "   failed, see $L/${MODEL}_${suite}_${tag}.log"; rm -f "$f"; }
}

for b in ${@:-sp vc ev aac}; do
  for suite in $SUITES; do
    case $b in
      sp)  cell sp  specprune_eval.py specprune $suite $SP ;;
      vc)  cell vc  vlacache_eval.py  vlacache  $suite $VC ;;
      ev)  cell ev  effvla_eval.py    effvla    $suite $EV ;;
      aac) cell aac aac_eval.py       aac       $suite $AAC ;;
    esac
  done
done
