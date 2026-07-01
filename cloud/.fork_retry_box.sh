#!/usr/bin/env bash
# Retry wrapper for run_one_forking_box_32b.sh: a transient cloud fault (host never
# boots, rsync drop, at_setup flake) should be INVISIBLE -- each attempt re-searches a
# FRESH offer, so we only give up after exhausting retries. SUCCESS == driver exit 0.
set -u; cd "$(dirname "$0")/.."
MAX_ATTEMPTS="${MAX_ATTEMPTS:-4}"
LBL="${RUN_LABEL:-box}"
for a in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "[retry $LBL] attempt $a/$MAX_ATTEMPTS $(date +%H:%M:%S)"
  bash cloud/run_one_forking_box_32b.sh
  rc=$?
  if [ "$rc" -eq 0 ]; then echo "[retry $LBL] SUCCESS on attempt $a"; exit 0; fi
  if [ "$rc" -eq 21 ]; then echo "[retry $LBL] ABORT: vast API unreachable (rc=21) -- retrying is futile, fix network"; exit 21; fi
  echo "[retry $LBL] attempt $a failed rc=$rc -- fresh offer in 15s"
  sleep 15
done
echo "[retry $LBL] EXHAUSTED $MAX_ATTEMPTS attempts"; exit 1
