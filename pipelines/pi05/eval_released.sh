#!/bin/bash
# Evaluate the released students (checkpoints/pi05/) without training anything.
#   bash pipelines/pi05/eval_released.sh [suite ...]
source "$(dirname "$0")/common.sh"
lat_table
declare -A CK=([libero_spatial]=st_ft_spatial [libero_object]=st_ft_object [libero_goal]=st_ft_goal [libero_10]=st_long_pe)
for s in ${@:-$SUITES}; do
  c=$ROOT/checkpoints/pi05/${CK[$s]}_g0.pt
  NL=$(net_latency $c)
  evaluate $s r5 --configs baseline --replan 5
  evaluate $s student $EARG --net $c --net-latency $NL
  python3 paired_table.py $R/$s r5 student
done
