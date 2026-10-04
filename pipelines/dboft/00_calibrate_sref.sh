#!/bin/bash
source "$(dirname "$0")/common.sh"
if [ -f "$SREF" ] && [ "${FORCE:-0}" != 1 ]; then
    log "$SREF exists (the one the paper used); FORCE=1 re-measures it"; exit 0
fi
rm -f "$SREF"
export RNG=0
CFG=evaluation/configs/simpler/dboft_calib.yaml eval_run calib --mode teacher $TC --cal-calls 48 --cal-first 12 --sref-file "$SREF"
python3 -c "import json, sys; d = json.load(open('$SREF')); print('SREF tasks:', len(d)); sys.exit(0 if len(d) == 4 else 1)" \
    || { log "SREF is missing tasks"; exit 1; }
