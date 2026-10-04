#!/bin/bash
: "${CORRECTOR_HOME:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
D=$CORRECTOR_HOME/dboft
set -u
export PATH=$D/bin:$PATH HF_HOME=$D/hf
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json XDG_RUNTIME_DIR=/tmp

VENV_DEX=${VENV_DEX:-$D/venv_dex}
VENV_CLIENT=${VENV_CLIENT:-$D/venv_client}
PY_SERVER=$VENV_DEX/bin/python
PY_CLIENT=$VENV_CLIENT/bin/python
SERVER=$CORRECTOR_HOME/scripts/dboft_server.py
TRAINER=$CORRECTOR_HOME/scripts/train_dboft_student.py
BENCH=$D/dexbotic-benchmark
OUT=$D/results/dboft
HV=$D/harvest/dboft
CKPT=$D/ckpt
SREF=$CORRECTOR_HOME/data/dboft/sref_dboft.json
TRAIN_SCENES=$CORRECTOR_HOME/data/dboft/train_scenes.json
SEEDS="${SEEDS:-0 2 6 8 12}"
TASKS="${TASKS:-StackCube Carrot Spoon Eggplant}"
CFG=${CFG:-evaluation/configs/simpler/dboft_local.yaml}

TC="--k 4 --start-step 8 --n-min 2 --lineage 15 --agree-tau 1.5 --grip-lock --grip-lock-mode both"

O1=simpler/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
O2=simpler/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png
mkdir -p "$OUT" "$HV" "$CKPT" "$D/logs"

log() { echo "[$(date +%H:%M:%S)] $*"; }

stop_server() {
    for p in $(ss -ltnp 2>/dev/null | grep ':7891' | grep -oE 'pid=[0-9]+' | cut -d= -f2); do kill "$p" 2>/dev/null; done
    for _ in $(seq 1 60); do
        m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        [ "$m" -lt 2000 ] && break
        sleep 3
    done
}

start_server() {
    local logfile=$1; shift
    stop_server
    ( cd "$D/dexbotic" && setsid nohup "$PY_SERVER" "$SERVER" --log "$logfile" "$@" \
        > "$D/logs/server_$(basename "$logfile" .jsonl).log" 2>&1 < /dev/null & )
    for _ in $(seq 1 100); do sleep 3; ss -ltn | grep -q ':7891' && break; done
    ss -ltn | grep -q ':7891' || { log "server did not start"; tail -20 "$D/logs/server_$(basename "$logfile" .jsonl).log"; exit 1; }
}

one_task() {
    local name=$1 task=$2 envname scene ov robot x y want
    case $task in
      StackCube) envname=StackGreenCubeOnYellowCubeBakedTexInScene-v0; scene=bridge_table_1_v1; ov=$O1; robot=widowx; x=0.147; y=0.028 ;;
      Carrot)    envname=PutCarrotOnPlateInScene-v0;                   scene=bridge_table_1_v1; ov=$O1; robot=widowx; x=0.147; y=0.028 ;;
      Spoon)     envname=PutSpoonOnTableClothInScene-v0;               scene=bridge_table_1_v1; ov=$O1; robot=widowx; x=0.147; y=0.028 ;;
      Eggplant)  envname=PutEggplantInBasketScene-v0;                  scene=bridge_table_1_v2; ov=$O2; robot=widowx_sink_camera_setup; x=0.127; y=0.06 ;;
    esac
    local extra=()
    [ -n "${EP_RANGE:-}" ] && extra=(--set obj-episode-range "$EP_RANGE")
    log ">>> $name / $task"
    ( cd "$BENCH" && env DBOFT_VARSEG=1 ${SCENES_FILE:+DBOFT_SCENES=$SCENES_FILE} "$PY_CLIENT" evaluation/run_simpler_evaluation.py \
        --config "$CFG" \
        --set octo-init-rng "$RNG" --set additional-env-save-tags octo_init_rng_"$RNG" \
        --set env-name $envname --set scene-name $scene --set rgb-overlay-path $ov --set robot $robot \
        --set robot-init-x-range "$x,$x,1" --set robot-init-y-range "$y,$y,1" ${extra[@]+"${extra[@]}"} \
        --set output-dir "$OUT/$name/env_$task" 2>&1 | grep -E "Success rate|Traceback|failed" | tail -3 )
    want=24
    [ -n "${EP_RANGE:-}" ] && want=$(( ${EP_RANGE#*,} - ${EP_RANGE%,*} ))
    local n; n=$(find "$OUT/$name/env_$task" -name "*.mp4" 2>/dev/null | wc -l)
    [ "$n" -lt "$want" ] && log "  WARNING: $name/$task has only $n/$want episodes"
    return 0
}

run() {
    local name=$1 tasks=$2; shift 2
    log "===== $name [$tasks] rng $RNG: $* ====="
    start_server "$OUT/calls_${name}.jsonl" "$@"
    for t in $tasks; do one_task "$name" "$t"; done
    stop_server
}

eval_run() {
    local name=$1; shift
    ( unset SCENES_FILE EP_RANGE DBOFT_SCENES; run "$name" "$TASKS" "$@" )
}

train_run() {
    local name=$1; shift
    SCENES_FILE=$TRAIN_SCENES EP_RANGE="0,48" run "$name" "$TASKS" "$@"
}
