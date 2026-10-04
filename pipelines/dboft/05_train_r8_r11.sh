#!/bin/bash
source "$(dirname "$0")/common.sh"
DATA="$HV/hv_r20,$HV/hv_r22,$HV/hv_r24,$HV/dg_r30,$HV/dg_r32,$HV/dg_r34"
cd "$CORRECTOR_HOME/scripts"
[ -f "$CKPT/dboft_student_r8.pt" ] || "$PY_SERVER" "$TRAINER" --scenes-per-task 48 --data "$DATA" \
    --dagger-tasks Eggplant --init "$CKPT/dboft_student_r1.pt" --lr 1e-4 --epochs 10 --warmup 100 \
    --out "$CKPT/dboft_student_r8.pt"
[ -f "$CKPT/dboft_student_r11.pt" ] || "$PY_SERVER" "$TRAINER" --scenes-per-task 48 --data "$DATA" \
    --dagger-tasks StackCube --init "$CKPT/dboft_student_r8.pt" --lr 5e-5 --epochs 3 --warmup 30 \
    --out "$CKPT/dboft_student_r11.pt"
log "evaluate with: bash pipelines/dboft/01_eval.sh student $CKPT/dboft_student_r11.pt"
