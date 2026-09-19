#!/bin/bash
# Rounds 0-2 of distillation.  Round 0 is harvested with the teacher driving; rounds 1 and 2
# with the previous student driving (DAgger), each on its own block of random scenes -- the
# evaluation's 50 fixed initial states are never used for training.  20 scenes a task per round.
source "$(dirname "$0")/common.sh"
HARG="--model pi05 --tasks 10 --trials 20 --random-scenes --c-max 10 --c-safe 5 --lineage 50"
harvest () {   # tag seed scene-seed [driver args]
  local tag=$1 seed=$2 sseed=$3; shift 3
  for s in $SUITES; do
    [ -s $D/${tag}_${s}_s${seed}.pt ] && { echo "  skip harvest $tag $s"; continue; }
    echo "== HARVEST $tag $s  [$(date +%H:%M)]"
    $PY -u harvest_distill.py $HARG --suite $s --seed $seed --scene-seed $sseed --out $D/${tag}_${s} "$@" \
      > $L/harvest_${tag}_${s}.log 2>&1 || { tail -3 $L/harvest_${tag}_${s}.log; exit 1; }
  done
}
harvest r0 301 100000                                         # teacher drives
train st_r0 --shard "$(list r0 301)" --shard-weight 0.25,0.25,0.25,0.25 --select-shards 0,1,2,3 \
  --val-obs 320 --epochs 100 --val-every 5000
harvest dg 302 200000 --net $D/st_r0_g0.pt                    # st_r0 drives
train st_r1 --shard "$(list r0 301),$(list dg 302)" --shard-weight $(python3 -c "print(','.join(['0.125'] * 8))") \
  --select-shards 0,1,2,3,4,5,6,7 --val-obs 320 --epochs 100 --val-every 5000
harvest dg2 303 300000 --net $D/st_r1_g0.pt                   # st_r1 drives
train st_r2 --shard "$(list r0 301),$(list dg 302),$(list dg2 303)" \
  --shard-weight $(python3 -c "print(','.join(['0.0833'] * 12))") --select-shards 0,1,2,3,4,5,6,7,8,9,10,11 \
  --val-obs 480 --epochs 60 --val-every 5000
