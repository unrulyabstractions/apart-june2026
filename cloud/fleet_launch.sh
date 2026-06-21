#!/usr/bin/env bash
#
# fleet_launch.sh — launch ALL per-(model, shard) boxes CONCURRENTLY (req. A2/A3).
#
# Reads the fleet plan from cloud/fleet_sizing.py (one TSV row per box), then fires
# every `vast create` IN PARALLEL (background jobs) and polls them all at once, so
# total launch wall-clock ≈ the SLOWEST single box, not the sum. Each box's
# instance id is recorded in cloud/.fleet/<bare>__shard<k>of<K>.id, consumed by
# fleet_run.sh / fleet_sync_back.sh / fleet_destroy.sh.
#
# ┌───────────────────────────────────────────────────────────────────────────┐
# │ DRAFT. A human runs it after review. It costs money once boxes are created.│
# │ Nothing else in cloud/ spends. Set FLEET_CONFIRM=1 to skip the y/N prompt.  │
# └───────────────────────────────────────────────────────────────────────────┘
#
# Usage:
#   bash cloud/fleet_launch.sh                 # launch the default fleet
#   FLEET_CONFIRM=1 bash cloud/fleet_launch.sh # no interactive confirm

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
FLEET_DIR="${FLEET_DIR:-$HERE/.fleet}"
DISK="${DISK:-60}"
IMAGE="${IMAGE:-vastai/pytorch:@vastai-automatic-tag}"
PORTAL_ENV='-p 1111:1111 -e OPEN_BUTTON_PORT="1111" -e OPEN_BUTTON_TOKEN="1"'

command -v vastai >/dev/null || { echo "vastai not found. pip install vastai" >&2; exit 1; }
mkdir -p "$FLEET_DIR"

PLAN="$(/usr/bin/env python3 "$HERE/fleet_sizing.py" plan)"
echo ">> Fleet plan (model / gpu / \$max / shard):"
printf '%s\n' "$PLAN" | awk -F'\t' '{printf "   %-40s %-10s $%-5s shard %s/%s\n",$1,$3,$4,$5+1,$6}'

if [ "${FLEET_CONFIRM:-0}" != "1" ]; then
  read -r -p ">> Create ALL of these boxes concurrently? [y/N] " ans
  [ "$ans" = "y" ] || [ "$ans" = "Y" ] || { echo "Aborted."; exit 0; }
fi

# Launch ONE box for a plan row; backgrounded by the caller so all run at once.
launch_one() {
  local model="$1" bare="$2" gpu="$3" price="$4" sidx="$5" scount="$6"
  local tag="${bare}__shard${sidx}of${scount}"
  local q="gpu_name=${gpu} num_gpus>=1 verified=true rentable=true direct_port_count>=1 disk_space>=${DISK} dph_total<=${price}"
  local offers oid create iid
  offers="$(vastai search offers "$q" -o dph_total+ --raw)"
  oid="$(printf '%s' "$offers" | python3 -c 'import sys,json;o=json.load(sys.stdin);print(o[0]["id"] if o else "")')"
  if [ -z "$oid" ]; then echo "[$tag] NO OFFER (loosen price/gpu)"; return 1; fi
  create="$(vastai create instance "$oid" --image "$IMAGE" --env "$PORTAL_ENV" \
    --onstart-cmd 'entrypoint.sh' --disk "$DISK" --ssh --direct --raw)"
  iid="$(printf '%s' "$create" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("new_contract",""))')"
  if [ -z "$iid" ]; then echo "[$tag] CREATE FAILED"; return 1; fi
  # Record id + the model/shard this box must run (read by fleet_run.sh).
  printf '%s\n' "$iid" > "$FLEET_DIR/$tag.id"
  printf '%s\t%s\t%s\n' "$model" "$sidx" "$scount" > "$FLEET_DIR/$tag.job"
  echo "[$tag] created instance $iid (offer $oid, $gpu)"
}

export -f launch_one
export FLEET_DIR DISK IMAGE PORTAL_ENV

# Fire every create in the BACKGROUND, then wait for all — concurrent launch.
echo ">> Launching all boxes concurrently..."
while IFS=$'\t' read -r model bare gpu price sidx scount; do
  launch_one "$model" "$bare" "$gpu" "$price" "$sidx" "$scount" &
done <<< "$PLAN"
wait

echo ">> All create calls returned. Instance ids under $FLEET_DIR/*.id"
echo ">> Next: bash cloud/fleet_run.sh   # setup+run+sync-back+self-destruct, all in parallel"
