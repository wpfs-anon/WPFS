#!/bin/bash
# One more DAgger round, st_r8 driving: the states this student itself reaches.
source "$(dirname "$0")/common.sh"
HARG="--model pi05 --tasks 10 --trials 60 --random-scenes --scene-seed 800000 --seed 801 --c-max 10 --c-safe 5 --lineage 50"
for s in $SUITES; do
  [ -s $D/pd_${s}_s801.pt ] && { echo "  skip harvest pd $s"; continue; }
  echo "== HARVEST pd $s, st_r8 driving  [$(date +%H:%M)]"
  $PY -u harvest_dagger.py $HARG --net $D/st_r8_g0.pt --suite $s --out $D/pd_${s} > $L/harvest_pd_${s}.log 2>&1 \
    || { tail -3 $L/harvest_pd_${s}.log; exit 1; }
done
