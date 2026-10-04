#!/bin/bash
source "$(dirname "$0")/common.sh"
[ -f "$CKPT/dboft_student_r1.pt" ] && { log "r1 exists, skipped"; exit 0; }
( cd "$CORRECTOR_HOME/scripts" && "$PY_SERVER" "$TRAINER" --scenes-per-task 48 \
    --data "$HV/hv_r20,$HV/hv_r22,$HV/hv_r24" --out "$CKPT/dboft_student_r1.pt" --epochs 60 )
