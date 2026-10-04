set -u
ROOT="${CORRECTOR_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${CPUS:+taskset -c $CPUS }$ROOT/.venv/bin/python"
R=$ROOT/ckpt/pi0
L=$ROOT/logs/pi0
LAT=${LAT:-$ROOT/data/latency/pi0_policy.json}
SUITES="libero_spatial libero_object libero_goal libero_10"
CORR="--configs corrector --adaptive --k-draws 4 --tau0 0.5 --grip snap --n-min 5 --c-safe 10 --agree-tau 1.0"
export JAX_PLATFORMS=cpu MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export TORCHINDUCTOR_COMPILE_THREADS=1
mkdir -p $L; for s in $SUITES; do mkdir -p $R/$s; done
cd $ROOT/scripts

cmax () { [ "$1" = libero_object ] && echo 10 || echo 25; }

evaluate () {
  local s=$1 n=$2; shift 2
  local f=$R/$s/${n}_s7.json
  if [ -s $f ]; then echo "  skip eval $s $n"; return 0; fi
  echo "== EVAL $s $n  [$(date +%H:%M)]"
  $PY -u eval_corrector.py --model pi0 --latency-from $LAT --suite $s --seed 7 --trials 50 \
    "$@" --out $f 2>&1 | tee $L/eval_${s}_$n.log | grep --line-buffered -aE "\[[0-9]+/10\]|success "
}
