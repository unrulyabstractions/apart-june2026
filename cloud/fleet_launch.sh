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
# Disk is per-model now (sized in fleet_sizing.py so big-model weight downloads
# don't run out of space); no single DISK knob here anymore.
IMAGE="${IMAGE:-vastai/pytorch:@vastai-automatic-tag}"
PORTAL_ENV='-p 1111:1111 -e OPEN_BUTTON_PORT="1111" -e OPEN_BUTTON_TOKEN="1"'

command -v vastai >/dev/null || { echo "vastai not found. pip install vastai" >&2; exit 1; }
mkdir -p "$FLEET_DIR"

# TSV columns: model  bare  gpu  max_price  disk_gb  shard_index  shard_count
# FLEET_PLAN_FILE lets a controller launch a SUBSET / sharded plan (e.g. only the
# models still missing, or one model split across K boxes) without editing the
# shared fleet_sizing.py another controller is reading. Unset == the full default
# plan, so default behavior is unchanged.
if [ -n "${FLEET_PLAN_FILE:-}" ]; then
  PLAN="$(cat "$FLEET_PLAN_FILE")"
else
  PLAN="$(/usr/bin/env python3 "$HERE/fleet_sizing.py" plan)"
fi
echo ">> Fleet plan (model / gpu / \$max / disk / shard):"
printf '%s\n' "$PLAN" | awk -F'\t' '{printf "   %-40s %-10s $%-5s %sG  shard %s/%s\n",$1,$3,$4,$5,$6+1,$7}'

if [ "${FLEET_CONFIRM:-0}" != "1" ]; then
  read -r -p ">> Create ALL of these boxes concurrently? [y/N] " ans
  [ "$ans" = "y" ] || [ "$ans" = "Y" ] || { echo "Aborted."; exit 0; }
fi

# Launch ONE box for a plan row; backgrounded by the caller so all run at once.
# The plan's per-row disk (cuda_max_good>=12.4 so the cu124 torch wheel runs;
# disk sized to the model so 24-32B weight downloads don't run out of space).
launch_one() {
  local model="$1" bare="$2" gpu="$3" price="$4" disk="$5" sidx="$6" scount="$7"
  local tag="${bare}__shard${sidx}of${scount}"
  # RELIABILITY-FIRST selection (money is not the constraint, uptime is): require a
  # high reliability score + a datacenter-grade host so the box actually boots and
  # is not reclaimed mid-run, and order by reliability DESC (not cheapest-first,
  # which kept drawing flaky hosts that never reached 'running').
  local q="gpu_name=${gpu} num_gpus>=1 verified=true rentable=true direct_port_count>=1 disk_space>=${disk} dph_total<=${price} cuda_max_good>=12.4 reliability2>=0.985"
  local offers oid create iid
  offers="$(vastai search offers "$q" -o reliability2- --raw)"
  # Fallback: if no host clears the strict reliability bar, relax it once rather
  # than silently giving up (still reliability-ordered, never cheapest-first).
  if [ "$(printf '%s' "$offers" | python3 -c 'import sys,json;print(len(json.load(sys.stdin)))' 2>/dev/null || echo 0)" -eq 0 ]; then
    q="gpu_name=${gpu} num_gpus>=1 verified=true rentable=true direct_port_count>=1 disk_space>=${disk} dph_total<=${price} cuda_max_good>=12.4 reliability2>=0.95"
    offers="$(vastai search offers "$q" -o reliability2- --raw)"
  fi
  # SPREAD ACROSS MACHINES: every concurrent launch_one searches at the same instant
  # and, if each took offers[0], they ALL land on the single top machine — co-locating
  # 2+ instances per host, where the 2nd instance stalls in 'loading' forever (it never
  # reaches 'running', so wait_running times out and the box is wasted). Instead we pick
  # UNIFORMLY AT RANDOM among the top reliable offers, ONE PER DISTINCT MACHINE, so
  # concurrent launches scatter across hosts while every pick still comes from the
  # high-reliability set (reliability-first preserved, co-location avoided).
  oid="$(printf '%s' "$offers" | python3 -c '
import sys, json, random
offers = json.load(sys.stdin)
if not offers:
    print(""); sys.exit()
offers.sort(key=lambda o: -o.get("reliability2", 0))  # reliability DESC
# keep the FIRST offer per machine (its most-reliable offer), among the top 24 hosts
seen, pool = set(), []
for o in offers:
    m = o.get("machine_id")
    if m in seen:
        continue
    seen.add(m); pool.append(o)
    if len(pool) >= 24:
        break
print(random.choice(pool)["id"])')"
  if [ -z "$oid" ]; then echo "[$tag] NO OFFER (loosen price/gpu/disk)"; return 1; fi
  create="$(vastai create instance "$oid" --image "$IMAGE" --env "$PORTAL_ENV" \
    --onstart-cmd 'entrypoint.sh' --disk "$disk" --ssh --direct --raw)"
  iid="$(printf '%s' "$create" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("new_contract",""))')"
  if [ -z "$iid" ]; then echo "[$tag] CREATE FAILED"; return 1; fi
  # Record id + the model/shard this box must run (read by fleet_run.sh).
  printf '%s\n' "$iid" > "$FLEET_DIR/$tag.id"
  printf '%s\t%s\t%s\n' "$model" "$sidx" "$scount" > "$FLEET_DIR/$tag.job"
  echo "[$tag] created instance $iid (offer $oid, $gpu, disk ${disk}G)"
}

export -f launch_one
export FLEET_DIR IMAGE PORTAL_ENV

# Fire every create in the BACKGROUND, then wait for all — concurrent launch.
echo ">> Launching all boxes concurrently..."
while IFS=$'\t' read -r model bare gpu price disk sidx scount; do
  launch_one "$model" "$bare" "$gpu" "$price" "$disk" "$sidx" "$scount" &
done <<< "$PLAN"
wait

echo ">> All create calls returned. Instance ids under $FLEET_DIR/*.id"
echo ">> Next: bash cloud/fleet_run.sh   # setup+run+sync-back+self-destruct, all in parallel"
