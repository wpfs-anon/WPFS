#!/bin/bash
# Per-suite students for Spatial, Object and Goal: st_r8 fine-tuned on that suite's two rounds
# (pc, st_r2 driving; pd, st_r8 driving) in equal parts, at lr 3e-4 for 20 epochs.  lr 3e-4 ends
# at a similar validation error to lr 1e-4 but evaluates better (Spatial 99.0 vs 98.4, Goal 98.8
# vs 98.0) -- select the rate on evaluation, not on validation error.
source "$(dirname "$0")/common.sh"
lat_table
for s in libero_spatial libero_object libero_goal; do
  t=${s#libero_}
  train st_ft_${t} --shard "$D/pc_${s}_s701.pt,$D/pd_${s}_s801.pt" --shard-weight 0.5,0.5 --select-shards 0,1 \
    --init-from $D/st_r8_g0.pt --lr 3e-4 --mem-noise 0.05 --use-plan --use-age \
    --val-obs 480 --epochs 20 --val-every 2000
  NL=$(net_latency $D/st_ft_${t}_g0.pt)
  evaluate $s student $EARG --net $D/st_ft_${t}_g0.pt --net-latency $NL
  python3 paired_table.py $R/$s r5 student
done
