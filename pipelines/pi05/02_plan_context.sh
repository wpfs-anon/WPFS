#!/bin/bash
# The student that reaches the teacher.  st_r2 drives a new harvest that also stores, for every
# correction, the teacher's plan-time hidden state (10 tokens, 1024 wide, computed by the plan and
# otherwise thrown away) and the plan's age (actions executed since it was made).  st_r8 reads
# both: --use-plan adds the plan tokens to its memory, --use-age embeds the age.  --mem-noise
# 0.05 jitters the memory by what a JPEG round trip of the frames really does.
source "$(dirname "$0")/common.sh"
HARG="--model pi05 --tasks 10 --trials 60 --random-scenes --scene-seed 700000 --seed 701 --c-max 10 --c-safe 5 --lineage 50"
for s in $SUITES; do
  [ -s $D/pc_${s}_s701.pt ] && { echo "  skip harvest pc $s"; continue; }
  echo "== HARVEST pc $s, st_r2 driving  [$(date +%H:%M)]"
  $PY -u harvest_dagger.py $HARG --net $D/st_r2_g0.pt --suite $s --out $D/pc_${s} > $L/harvest_pc_${s}.log 2>&1 \
    || { tail -3 $L/harvest_pc_${s}.log; exit 1; }
done
train st_r8 --shard "$(list pc 701)" --shard-weight 0.25,0.25,0.25,0.25 --select-shards 0,1,2,3 \
  --mem-noise 0.05 --use-plan --use-age --val-obs 480 --epochs 60 --val-every 5000 --save-every 20000
