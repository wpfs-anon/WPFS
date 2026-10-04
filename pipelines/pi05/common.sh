set -u
ROOT="${CORRECTOR_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${CPUS:+taskset -c $CPUS }$ROOT/.venv/bin/python"
D=$ROOT/distill/pi05
R=$ROOT/ckpt/pi05
L=$ROOT/logs/pi05
LAT=$R/latency_pi05.json
SUITES="libero_spatial libero_object libero_goal libero_10"
EARG="--configs corrector --adaptive --k-draws 4 --tau0 0.5 --grip snap --n-min 5 --c-max 10 --c-safe 5 --agree-tau 1.0 --lineage 50"
export JAX_PLATFORMS=cpu MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export TORCHINDUCTOR_COMPILE_THREADS=1
mkdir -p $D $L; for s in $SUITES; do mkdir -p $R/$s; done
cd $ROOT/scripts

list () {
  local out="" s; for s in ${3:-$SUITES}; do out="$out,$D/${1}_${s}_s${2}.pt"; done; echo "${out#,}"
}
lat_table () {
  [ -s $LAT ] || $PY -u latency_probe.py --model pi05 --out $LAT > $L/latency_pi05.log 2>&1
}
net_latency () {
  local j=$D/lat_$(basename $1 _g0.pt).json
  [ -s $j ] || $PY -u latency_probe.py --model pi05 --net $1 --out $j > $L/lat_$(basename $1 _g0.pt).log 2>&1
  python3 -c "import json; print(round(json.load(open('$j'))['latency_ms']['net'], 2))"
}
train () {
  local out=$1; shift
  if [ -s $D/${out}_g0.pt ] && grep -aq "best SEL rel" $L/train_$out.log 2>/dev/null; then
    echo "  skip train $out"; return 0; fi
  echo "== TRAIN $out  [$(date +%H:%M)]"
  $PY -u train_net_student.py --model pi05 --gemma-layers 0 --lambda-spread 1.0 "$@" \
    --cache $D/cache_$out --out $D/$out 2>&1 | tee $L/train_$out.log \
    | grep --line-buffered -aE "warm start|steps/epoch|step [0-9]+0000/|best SEL"
  grep -aq "best SEL rel" $L/train_$out.log || { echo "== TRAINING $out DID NOT FINISH"; exit 1; }
}
evaluate () {
  local s=$1 n=$2; shift 2
  local f=$R/$s/${n}_s7.json
  if [ -s $f ]; then echo "  skip eval $s $n"; return 0; fi
  echo "== EVAL $s $n  [$(date +%H:%M)]"
  $PY -u eval_corrector.py --model pi05 --latency-from $LAT --suite $s --seed 7 --trials 50 \
    "$@" --out $f 2>&1 | tee $L/eval_${s}_$n.log | grep --line-buffered -aE "\[[0-9]+/10\]|success "
}
