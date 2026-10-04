#!/bin/bash
source "$(dirname "$0")/common.sh"
WHAT=${1:-all}
STUDENT=${2:-$CORRECTOR_HOME/checkpoints/dboft/student.pt}
[ -f "$SREF" ] || { log "no $SREF: run 00_calibrate_sref.sh"; exit 1; }
for RNG in $SEEDS; do
    export RNG
    case $WHAT in all|baseline) eval_run baseline_r$RNG --mode baseline ;; esac
    case $WHAT in all|teacher)  eval_run teacher_r$RNG --mode teacher $TC --sref-file "$SREF" ;; esac
    case $WHAT in all|student)  eval_run student_r$RNG --mode teacher $TC --sref-file "$SREF" --student "$STUDENT" ;; esac
done
"$PY_CLIENT" "$CORRECTOR_HOME/scripts/dboft_table.py" --seeds "${SEEDS// /,}" baseline=baseline teacher=teacher student=student
