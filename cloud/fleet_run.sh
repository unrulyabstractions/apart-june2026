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
# SUBSAMPLE / MAX_NEW_TOKENS pass straight through to fleet_model_run.sh on the
# box. Both default EMPTY here so an unset value lets fleet_model_run.sh apply its
# own default (full grid / 512 tokens) — backward-compatible no-op.
SUBSAMPLE="${SUBSAMPLE:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"

[ -d "$FLEET_DIR" ] || { echo "No fleet. Run cloud/fleet_launch.sh first." >&2; exit 1; }

# One instance's status, looked up across ALL pages of the account's instances.
# The v1 instances endpoint PAGINATES (25/page); a single `--raw` page only shows
# the first 25, so when the account already holds >25 instances (many concurrent
# fleets), a box on a later page reads as "missing" forever and wait_running times
# out. We therefore walk every page via next_token until the id is found.
instance_status() {
  INSTANCE_ID="$1" python3 -c '
import json, os, subprocess, sys
iid = int(os.environ["INSTANCE_ID"])
token = None
for _ in range(40):  # hard page cap so a runaway token loop can never hang
    cmd = ["vastai", "show", "instances-v1", "--raw"]
    if token:
        cmd += ["--next-token", token]
    try:
        d = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)
    except Exception:
        print("missing"); sys.exit()
    rows = d if isinstance(d, list) else d.get("instances", d.get("results", []))
    for r in rows:
        if r.get("id") == iid:
            print(r.get("actual_status") or r.get("cur_state") or "unknown"); sys.exit()
    token = d.get("next_token") if isinstance(d, dict) else None
    if not token or not rows:
        break
print("missing")'
}

# Wait (bounded) for a single instance to report 'running'.
wait_running() {
  local iid="$1" i st
  for i in $(seq 1 80); do
    st="$(instance_status "$iid")"
    [ "$st" = "running" ] && return 0
    sleep 15
  done
  return 1
}

# A box reports 'running' (API) well BEFORE its sshd / direct port is reachable.
# Pushing then races the network and rsync dies mid-transfer ("connection
# unexpectedly closed"), leaving the box with NO code / NO prompt xlsx -> generate
# produces 0 items -> collect writes an EMPTY response_samples.json. So after
# wait_running we PROBE actual SSH connectivity (a trivial remote `true`) and only
# proceed once the box really answers. Bounded; returns non-zero if it never does.
wait_ssh() {
  local iid="$1" i
  for i in $(seq 1 40); do
    if INSTANCE="$iid" bash "$HERE/at_vast.sh" "true" >/dev/null 2>&1; then
      return 0
    fi
    sleep 10
  done
  return 1
}

# Push code+prompts (sync_up) and build the env (at_setup), each RETRIED a few
# times: a transient SSH hiccup mid-rsync must not silently leave a half-synced
# box. Returns non-zero only if every attempt fails, so the caller can abort the
# box instead of running it on an empty/partial tree.
sync_and_setup() {
  local iid="$1" attempt
  for attempt in 1 2 3; do
    echo "[setup attempt $attempt] sync_up"
    if INSTANCE="$iid" bash "$HERE/sync_up.sh" && \
       INSTANCE="$iid" bash "$HERE/at_vast.sh" "bash cloud/at_setup.sh"; then
      return 0
    fi
    echo "[setup attempt $attempt] FAILED; retrying after 15s"
    sleep 15
  done
  return 1
}

# Run ONE box end-to-end, then destroy it. Backgrounded by the caller.
run_one() {
  local tag="$1"
  local iid model sidx scount ngpu hf_device_map log
  iid="$(cat "$FLEET_DIR/$tag.id")"
  IFS=$'\t' read -r model sidx scount ngpu < "$FLEET_DIR/$tag.job"
  # MULTI-GPU box (e.g. Llama-3.1-70B on 2× H100): tell the on-box HF backend to
  # shard the weights across every visible GPU via device_map="auto". 1-GPU boxes
  # (and older 3-field .job files where ngpu is empty) keep the single-device path.
  hf_device_map=""
  [ "${ngpu:-1}" -gt 1 ] 2>/dev/null && hf_device_map="auto"
  log="$FLEET_DIR/$tag.log"
  { echo "[$tag] instance=$iid model=$model shard=$sidx/$scount"

    wait_running "$iid" || { echo "[$tag] never became running"; return 1; }

    # Step 1b: wait until sshd is ACTUALLY reachable (API 'running' is too early).
    if ! wait_ssh "$iid"; then
      echo "[$tag] SSH never came up; destroying box (no work attempted)."
      INSTANCE="$iid" bash "$HERE/vast_destroy.sh" --yes-i-am-really-sure
      return 1
    fi

    # Steps 2-3: push code + env, with retries. If the box can't be fully set up,
    # ABORT before running so we never produce an EMPTY result on a half-synced
    # box -- and destroy it so it stops billing.
    if ! sync_and_setup "$iid"; then
      echo "[$tag] sync_up/at_setup failed after retries; destroying box (no empty run)."
      INSTANCE="$iid" bash "$HERE/vast_destroy.sh" --yes-i-am-really-sure
      return 1
    fi

    # Step 4: run THIS box's model+shard pipeline (batched) on the box. On a
    # multi-GPU box HF_DEVICE_MAP=auto makes the HF backend shard the model across
    # every GPU (empty == single-device path, unchanged for 1-GPU boxes).
    INSTANCE="$iid" MODEL="$model" SHARD_INDEX="$sidx" SHARD_COUNT="$scount" \
      STUDIES="$STUDIES" BATCH_SIZE="$BATCH_SIZE" N_THINKING="$N_THINKING" \
      bash "$HERE/at_vast.sh" \
      "HF_TOKEN='$HF_TOKEN' HF_DEVICE_MAP='$hf_device_map' MODEL='$model' SHARD_INDEX=$sidx SHARD_COUNT=$scount STUDIES='$STUDIES' BATCH_SIZE=$BATCH_SIZE N_THINKING=$N_THINKING SUBSAMPLE='$SUBSAMPLE' MAX_NEW_TOKENS='$MAX_NEW_TOKENS' bash cloud/fleet_model_run.sh"

    # Step 5: pull THIS box's results into ITS OWN sync/box-<tag>/ quarantine.
    INSTANCE="$iid" SYNC_SUBDIR="box-$tag" bash "$HERE/sync_back.sh"

    # Step 6: self-destroy — billing stops the moment this box's work lands.
    INSTANCE="$iid" bash "$HERE/vast_destroy.sh" --yes-i-am-really-sure
    echo "[$tag] DONE + destroyed"
  } >"$log" 2>&1
  echo "[$tag] finished (log: $log)"
}

export -f run_one wait_running instance_status wait_ssh sync_and_setup
export FLEET_DIR STUDIES BATCH_SIZE N_THINKING SUBSAMPLE MAX_NEW_TOKENS HERE HF_TOKEN

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
