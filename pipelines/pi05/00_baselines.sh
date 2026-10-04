#!/bin/bash
source "$(dirname "$0")/common.sh"
lat_table
for s in $SUITES; do
  evaluate $s r5 --configs baseline --replan 5
  evaluate $s r10 --configs baseline --replan 10
  evaluate $s teacher_L50 $EARG
done
for s in $SUITES; do echo "--- $s"; python3 paired_table.py $R/$s r5 r10 teacher_L50; done
