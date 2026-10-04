#!/bin/bash
source "$(dirname "$0")/common.sh"
[ -f "$CKPT/dboft_student_r1.pt" ] || { log "no r1: run 03_train_r1.sh"; exit 1; }
for RNG in 30 32 34; do
    export RNG
    if ls "$HV/dg_r$RNG"/shard_*.pt >/dev/null 2>&1; then log "dg_r$RNG exists, skipped"; continue; fi
    train_run dg_r$RNG --mode teacher $TC --sref-file "$SREF" --student "$CKPT/dboft_student_r1.pt" \
        --harvest "$HV/dg_r$RNG" --k-harvest 8
done
