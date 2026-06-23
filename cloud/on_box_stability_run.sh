#!/usr/bin/env bash
#
# on_box_stability_run.sh — runs ON the vast box, launched DETACHED (nohup setsid) by the
# local driver so a dropped ssh connection NEVER interrupts it. The box keeps working; the
# driver just reconnects and polls. Builds the datasets, then runs the resumable greedy
# readout for every mode x dataset, and writes a terminal marker the driver polls for:
#   out/.STAB_DONE   — every slice reached its full expected count
#   out/.STAB_FAILED — a slice failed all internal retries (real error, not a dropped link)
#
# Config via env: MODEL BARE_MODEL MODES MAX_REASONING SHARD_INDEX SHARD_COUNT LIMIT HF_TOKEN
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root on the box (/root/apart)
PY=".venv/bin/python"

: "${MODEL:?set MODEL}"; : "${BARE_MODEL:?set BARE_MODEL}"; : "${MODES:?set MODES}"
MAX_REASONING="${MAX_REASONING:-512}"
SHARD_INDEX="${SHARD_INDEX:-0}"; SHARD_COUNT="${SHARD_COUNT:-1}"
LIMIT_ARG=""; [ -n "${LIMIT:-}" ] && LIMIT_ARG="--limit ${LIMIT}"
export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"  # cut fragmentation OOM
SHARD_SUF=""; [ "${SHARD_COUNT}" -gt 1 ] && SHARD_SUF="/shard_${SHARD_INDEX}_of_${SHARD_COUNT}"

mkdir -p out
rm -f out/.STAB_DONE out/.STAB_FAILED
echo "[on_box] $(date) start model=$MODEL modes='$MODES' shard=$SHARD_INDEX/$SHARD_COUNT limit='${LIMIT:-full}'"

# Datasets (sync_up excludes data/, so build on-box). Retry a couple of times for any blip.
built=0
for a in 1 2 3; do
  "$PY" -m experiment.generate.build_stability_datasets --out-dir data && { built=1; break; }
  echo "[on_box] dataset build attempt $a failed; retrying"; sleep 10
done
[ "$built" = 1 ] || { echo "[on_box] dataset build FAILED"; touch out/.STAB_FAILED; exit 1; }

expected_count() {  # echo the expected slice size for a dataset path
  "$PY" - "$1" <<'PY'
import json, os, sys
recs = json.load(open(sys.argv[1]))
lim = os.environ.get("LIMIT")
if lim:
    recs = recs[: int(lim)]
n = len(recs); i = int(os.environ["SHARD_INDEX"]); c = int(os.environ["SHARD_COUNT"])
print((n * (i + 1) // c) - (n * i // c) if c > 1 else n)
PY
}
have_count() {  # echo how many samples are already in an output file (0 if missing)
  "$PY" -c "import json,sys
try: print(len(json.load(open(sys.argv[1]))['samples']))
except Exception: print(0)" "$1" 2>/dev/null || echo 0
}

for MODE in $MODES; do
  for SPEC in "full_prompt_dataset.json:stability" "forced_fork.json:forked"; do
    DS="${SPEC%%:*}"; STUDY="${SPEC##*:}"
    OUT="out/$STUDY/${BARE_MODEL}-${MODE}${SHARD_SUF}/response_samples.json"
    exp="$(SHARD_INDEX=$SHARD_INDEX SHARD_COUNT=$SHARD_COUNT expected_count "data/$DS")"
    # The readout is checkpointed + resumable; retry it (it resumes) until the slice is full.
    ok=0
    for a in 1 2 3 4 5 6; do
      echo "[on_box] readout study=$STUDY mode=$MODE attempt=$a (have $(have_count "$OUT")/$exp)"
      "$PY" -m experiment.stability.run_greedy_readout \
          --model "$MODEL" --mode "$MODE" --backend huggingface \
          --dataset "data/$DS" --study "$STUDY" --out-dir out \
          --max-reasoning "$MAX_REASONING" \
          --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT" $LIMIT_ARG || true
      [ "$(have_count "$OUT")" -ge "$exp" ] && { ok=1; break; }
      echo "[on_box] slice incomplete; resuming after 15s"; sleep 15
    done
    [ "$ok" = 1 ] || { echo "[on_box] readout FAILED study=$STUDY mode=$MODE"; touch out/.STAB_FAILED; exit 1; }
    echo "[on_box] slice DONE $OUT ($(have_count "$OUT")/$exp)"
  done
done

touch out/.STAB_DONE
echo "[on_box] $(date) ALL DONE"
