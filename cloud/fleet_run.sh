#!/usr/bin/env bash
#
# fleet_run.sh — drive EVERY launched box CONCURRENTLY through its full pipeline,
# then SELF-DESTROY it (req. A2/A3 + B). Total wall-clock ≈ the slowest box.
#
# For each cloud/.fleet/*.id (one per (model, shard) box), in PARALLEL:
#   1. wait until the box is 'running' and its SSH endpoint resolves,
#   2. sync_up   the repo + SESGO prompts to THAT box,
#   3. at_setup  (uv sync; CUDA backend auto-selected; vLLM available in image),
#   4. RUN the box's model+shard pipeline on the box (batched; writes ONLY to its
#      own out/sesgo/<study>/<bare-model>/[shard_k_of_K]/ slice),
#   5. sync_back from THAT box into its OWN sync/box-<tag>/ quarantine
#      (rsync --ignore-existing, no --delete — the existing safety model),
#   6. vast_destroy THAT box (billing stops the moment its work lands).
#
# Because each box writes to a DISJOINT output slice and lands in a DISJOINT
# sync/ subtree, concurrent boxes never target the same file — safe under
# concurrency by construction. A per-box log lives at cloud/.fleet/<tag>.log.
#
# Usage:
#   bash cloud/fleet_run.sh                  # all boxes; default batched studies
#   STUDIES="divergence" BATCH_SIZE=32 bash cloud/fleet_run.sh

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
FLEET_DIR="${FLEET_DIR:-$HERE/.fleet}"
STUDIES="${STUDIES:-baseline divergence}"   # which collects each box runs
BATCH_SIZE="${BATCH_SIZE:-32}"              # vLLM continuous-batching width
N_THINKING="${N_THINKING:-8}"

[ -d "$FLEET_DIR" ] || { echo "No fleet. Run cloud/fleet_launch.sh first." >&2; exit 1; }

# Wait (bounded) for a single instance to report 'running'.
wait_running() {
  local iid="$1" i st
  for i in $(seq 1 80); do
    st="$(vastai show instances-v1 --raw 2>/dev/null | INSTANCE_ID="$iid" python3 -c '
import sys,json,os
iid=int(os.environ["INSTANCE_ID"])
d=json.load(sys.stdin)
rows=d if isinstance(d,list) else d.get("instances", d.get("results", []))
for r in rows:
    if r.get("id")==iid: print(r.get("actual_status") or r.get("cur_state") or "unknown"); break
else: print("missing")')"
    [ "$st" = "running" ] && return 0
    sleep 15
  done
  return 1
}

# Run ONE box end-to-end, then destroy it. Backgrounded by the caller.
run_one() {
  local tag="$1"
  local iid model sidx scount log
  iid="$(cat "$FLEET_DIR/$tag.id")"
  IFS=$'\t' read -r model sidx scount < "$FLEET_DIR/$tag.job"
  log="$FLEET_DIR/$tag.log"
  { echo "[$tag] instance=$iid model=$model shard=$sidx/$scount"

    wait_running "$iid" || { echo "[$tag] never became running"; return 1; }

    # Steps 2-3: push code + env. INSTANCE pins every helper to THIS box.
    INSTANCE="$iid" bash "$HERE/sync_up.sh"
    INSTANCE="$iid" bash "$HERE/at_vast.sh" "bash cloud/at_setup.sh"

    # Step 4: run THIS box's model+shard pipeline (batched) on the box.
    INSTANCE="$iid" MODEL="$model" SHARD_INDEX="$sidx" SHARD_COUNT="$scount" \
      STUDIES="$STUDIES" BATCH_SIZE="$BATCH_SIZE" N_THINKING="$N_THINKING" \
      bash "$HERE/at_vast.sh" \
      "HF_TOKEN='$HF_TOKEN' MODEL='$model' SHARD_INDEX=$sidx SHARD_COUNT=$scount STUDIES='$STUDIES' BATCH_SIZE=$BATCH_SIZE N_THINKING=$N_THINKING bash cloud/fleet_model_run.sh"

    # Step 5: pull THIS box's results into ITS OWN sync/box-<tag>/ quarantine.
    INSTANCE="$iid" SYNC_SUBDIR="box-$tag" bash "$HERE/sync_back.sh"

    # Step 6: self-destroy — billing stops the moment this box's work lands.
    INSTANCE="$iid" bash "$HERE/vast_destroy.sh" --yes-i-am-really-sure
    echo "[$tag] DONE + destroyed"
  } >"$log" 2>&1
  echo "[$tag] finished (log: $log)"
}

export -f run_one wait_running
export FLEET_DIR STUDIES BATCH_SIZE N_THINKING HERE

echo ">> Driving all boxes concurrently (studies: $STUDIES, batch_size: $BATCH_SIZE)..."
for idf in "$FLEET_DIR"/*.id; do
  [ -e "$idf" ] || { echo "No .id files in $FLEET_DIR"; exit 1; }
  tag="$(basename "$idf" .id)"
  run_one "$tag" &
done
wait

echo ">> Fleet finished. Results quarantined under sync/box-*/ (disjoint per box)."
echo ">> Inspect:  find sync -type f"
echo ">> Promote:  bash cloud/merge_sync.sh   # --ignore-existing into out/, no clobber"
