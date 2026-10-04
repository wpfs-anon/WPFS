#!/bin/bash
source "$(dirname "$0")/common.sh"
NET=${NET:-$ROOT/checkpoints/pi0/student.pt}
for s in ${@:-$SUITES}; do
  evaluate $s base --configs baseline --replan 10
  evaluate $s teacher $CORR --c-max $(cmax $s)
  if [ -f "$NET" ]; then evaluate $s student $CORR --c-max $(cmax $s) --net $NET; fi
  python3 paired_table.py $R/$s base teacher $( [ -f "$NET" ] && echo student )
done
