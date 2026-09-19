#!/bin/bash
# pi0.5 alone (replan every 5 and every 10 actions) and the teacher corrector, 4 suites.
source "$(dirname "$0")/common.sh"
lat_table
for s in $SUITES; do
  evaluate $s r5 --configs baseline --replan 5          # openpi's LIBERO default
  evaluate $s r10 --configs baseline --replan 10        # the whole 10-action chunk open loop
  evaluate $s teacher_L50 $EARG                         # pi0.5 correcting its own stale chunk
done
for s in $SUITES; do echo "--- $s"; python3 paired_table.py $R/$s r5 r10 teacher_L50; done
