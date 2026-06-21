#!/usr/bin/env bash
#
# fleet_big_retry.sh — launch + drive the BIG models (>16B) on H100_SXM in
# PARALLEL, as a STANDALONE driver with its own .fleet_big/ dir. It reuses the
# per-box helpers (sync_up / at_vast / sync_back / vast_destroy / fleet_model_run),
# every one pinned to its INSTANCE, so it can run ALONGSIDE an already-running
# cloud/fleet_run.sh without editing or colliding with it. Each box self-destructs.
#
# Usage:  HF_TOKEN=... bash cloud/fleet_big_retry.sh
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
FB="$HERE/.fleet_big"; mkdir -p "$FB"
IMAGE="${IMAGE:-vastai/pytorch:@vastai-automatic-tag}"
PORTAL_ENV='-p 1111:1111 -e OPEN_BUTTON_PORT="1111" -e OPEN_BUTTON_TOKEN="1"'
DISK="${DISK:-100}"          # GB: a 32B bf16 (~64GB) + the CUDA venv
GPU="${GPU:-H100_SXM}"       # 80GB SXM (H100_PCIE is usually sold out)
PRICE="${PRICE:-3.0}"
STUDIES="${STUDIES:-baseline}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MODELS=(
  "Qwen/Qwen3-32B"
  "google/gemma-2-27b-it"
  "mistralai/Mistral-Small-24B-Instruct-2501"
)

wait_running() {
  local iid="$1" i st
  for i in $(seq 1 120); do
    st="$(vastai show instances-v1 --raw 2>/dev/null | INSTANCE_ID="$iid" python3 -c '
import sys,json,os
iid=int(os.environ["INSTANCE_ID"])
for r in json.load(sys.stdin):
    if r.get("id")==iid: print(r.get("actual_status") or "unknown"); break
else: print("missing")')"
    [ "$st" = "running" ] && return 0
    sleep 15
  done
  return 1
}

drive() {  # iid model bare
  local iid="$1" model="$2" bare="$3" log="$FB/$3.log"
  { echo "[$bare] instance=$iid model=$model"
    wait_running "$iid" || { echo "[$bare] never became running"; return 1; }
    INSTANCE="$iid" bash "$HERE/sync_up.sh"
    INSTANCE="$iid" bash "$HERE/at_vast.sh" "bash cloud/at_setup.sh"
    INSTANCE="$iid" bash "$HERE/at_vast.sh" \
      "HF_TOKEN='$HF_TOKEN' MODEL='$model' SHARD_INDEX=0 SHARD_COUNT=1 STUDIES='$STUDIES' BATCH_SIZE=$BATCH_SIZE bash cloud/fleet_model_run.sh"
    INSTANCE="$iid" SYNC_SUBDIR="box-$bare" bash "$HERE/sync_back.sh"
    INSTANCE="$iid" bash "$HERE/vast_destroy.sh" --yes-i-am-really-sure
    echo "[$bare] DONE + destroyed"
  } >"$log" 2>&1
  echo "[$bare] finished (log: $log)"
}

for model in "${MODELS[@]}"; do
  bare="${model##*/}"
  q="gpu_name=$GPU num_gpus>=1 verified=true rentable=true direct_port_count>=1 disk_space>=$DISK dph_total<=$PRICE"
  oid="$(vastai search offers "$q" -o dph_total+ --raw 2>/dev/null | python3 -c 'import sys,json;o=json.load(sys.stdin);print(o[0]["id"] if o else "")')"
  if [ -z "$oid" ]; then echo "[$bare] NO OFFER ($GPU <= \$$PRICE, disk>=$DISK)"; continue; fi
  iid="$(vastai create instance "$oid" --image "$IMAGE" --env "$PORTAL_ENV" --onstart-cmd 'entrypoint.sh' --disk "$DISK" --ssh --direct --raw 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin).get("new_contract",""))')"
  if [ -z "$iid" ]; then echo "[$bare] CREATE FAILED"; continue; fi
  echo "[$bare] created instance $iid (offer $oid)"
  printf '%s\n' "$iid" > "$FB/$bare.id"
  drive "$iid" "$model" "$bare" &
done
wait
echo ">> fleet_big_retry: all big boxes done. Results under sync/box-*/"
