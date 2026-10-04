#!/bin/bash
source "$(dirname "$0")/common.sh"
[ -f "$SREF" ] || { log "no $SREF: run 00_calibrate_sref.sh"; exit 1; }
[ -f "$TRAIN_SCENES" ] || { log "no $TRAIN_SCENES: run scripts/dboft_new_scenes.py"; exit 1; }
for RNG in 20 22 24; do
    export RNG
    if ls "$HV/hv_r$RNG"/shard_*.pt >/dev/null 2>&1; then log "hv_r$RNG exists, skipped"; continue; fi
    train_run hv_r$RNG --mode teacher $TC --sref-file "$SREF" --harvest "$HV/hv_r$RNG" --k-harvest 8
done
du -sh "$HV"
