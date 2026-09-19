#!/bin/bash
# LIBERO-Long.  The same fine-tune (st_long, here at lr 1e-4 for 30 epochs) scores 93.0 against the
# base's 94.0, and its losses concentrate on tasks 5 and 7 (the two white-mug-to-plate tasks).  So
# st_long drives one more DAgger round on fresh scenes -- 60 a task, 120 on tasks 5 and 7 -- and
# st_r8 is fine-tuned on all three Long rounds, every observation equally likely.
source "$(dirname "$0")/common.sh"
lat_table
S=libero_10
train st_long --shard "$D/pc_${S}_s701.pt,$D/pd_${S}_s801.pt" --shard-weight 0.5,0.5 --select-shards 0,1 \
  --init-from $D/st_r8_g0.pt --lr 1e-4 --mem-noise 0.05 --use-plan --use-age \
  --val-obs 480 --epochs 30 --val-every 5000 --save-every 20000
if [ ! -s $D/pe_${S}_s901.pt ]; then
  echo "== HARVEST pe $S, st_long driving, 60 scenes a task and 120 on tasks 5, 7  [$(date +%H:%M)]"
  $PY -u harvest_dagger.py --model pi05 --suite $S --tasks 10 --trials 60 --task-trials 5:120,7:120 \
    --random-scenes --scene-seed 900000 --seed 901 --c-max 10 --c-safe 5 --lineage 50 \
    --net $D/st_long_g0.pt --out $D/pe_${S} > $L/harvest_pe_${S}.log 2>&1 || { tail -3 $L/harvest_pe_${S}.log; exit 1; }
fi
train st_long_pe --shard "$D/pc_${S}_s701.pt,$D/pd_${S}_s801.pt,$D/pe_${S}_s901.pt" \
  --init-from $D/st_r8_g0.pt --lr 3e-4 --mem-noise 0.05 --use-plan --use-age \
  --val-obs 480 --epochs 20 --val-every 2000
NL=$(net_latency $D/st_long_pe_g0.pt)
evaluate $S student $EARG --net $D/st_long_pe_g0.pt --net-latency $NL
python3 paired_table.py $R/$S r5 student
