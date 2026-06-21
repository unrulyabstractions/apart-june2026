#!/usr/bin/env bash
#
# vast_launch.sh — search Vast.ai for a matching GPU offer, create an instance
# with the PyTorch/Jupyter portal template, poll until it's running, and record
# the instance id in cloud/.vast_instance_id (consumed by the other scripts).
#
# Adapted from the constellation_takehome reference for the `apart` repo. Its job
# is to stand up a single CUDA GPU box on which we run the FULL SESGO divergence
# + stability collections (Qwen3-0.6B auto-selects the HuggingFace/CUDA backend
# on a GPU host — no MLX).
#
# ┌───────────────────────────────────────────────────────────────────────────┐
# │ THIS SCRIPT IS A DRAFT. Read it, then a human runs it after review.        │
# │ It costs money once it creates an instance. Nothing else here spends.      │
# └───────────────────────────────────────────────────────────────────────────┘
#
# Prereqs:
#   pip install vastai
#   vastai set api-key <YOUR_KEY>          # from the Vast Keys page (shown once)
#   SSH public key registered on your Vast account (Keys page)
#
# Usage:
#   chmod +x cloud/vast_launch.sh
#   bash cloud/vast_launch.sh
#
# Override the CONFIG via env vars:
#   GPU_NAME=RTX_4090 NUM_GPUS=1 MAX_PRICE=0.80 bash cloud/vast_launch.sh

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# ----------------------------- CONFIG -----------------------------
# Single RTX 3090, 50 GB disk, <$0.50/hr ceiling. Qwen3-0.6B is tiny (~1.2 GB in
# fp16) so a 3090 is ample; bump GPU_NAME/MAX_PRICE only if you also scale models.
GPU_NAME="${GPU_NAME:-RTX_3090}"
NUM_GPUS="${NUM_GPUS:-1}"
DISK="${DISK:-50}"            # GB; PERMANENT after creation.
MAX_PRICE="${MAX_PRICE:-0.50}"
ORDER="${ORDER:-dph_total+}"  # cheapest first; use 'dlperf_usd-' for perf/$.
IMAGE="${IMAGE:-vastai/pytorch:@vastai-automatic-tag}"

# Portal/Jupyter env block (verbatim from the PyTorch template "copy CLI" button).
PORTAL_ENV='-p 1111:1111 -p 6006:6006 -p 8080:8080 -p 8384:8384 -p 72299:72299 -e OPEN_BUTTON_PORT="1111" -e OPEN_BUTTON_TOKEN="1" -e JUPYTER_DIR="/" -e DATA_DIRECTORY="/workspace/" -e PORTAL_CONFIG="localhost:1111:11111:/:Instance Portal|localhost:8080:18080:/:Jupyter|localhost:8080:8080:/terminals/1:Jupyter Terminal|localhost:8384:18384:/:Syncthing|localhost:6006:16006:/:Tensorboard"'
# ------------------------------------------------------------------

command -v vastai >/dev/null || { echo "vastai not found. Run: pip install vastai" >&2; exit 1; }

# ----------------------------- 1. SEARCH --------------------------
QUERY="gpu_name=${GPU_NAME} num_gpus>=${NUM_GPUS} verified=true rentable=true direct_port_count>=1 disk_space>=${DISK} dph_total<=${MAX_PRICE}"
echo ">> Searching offers: ${QUERY}"
OFFERS_JSON="$(vastai search offers "$QUERY" -o "$ORDER" --raw)"

OFFER_ID="$(printf '%s' "$OFFERS_JSON" | python3 -c 'import sys,json; o=json.load(sys.stdin); print(o[0]["id"] if o else "")')"

if [[ -z "$OFFER_ID" ]]; then
  echo "No matching offers. Loosen GPU_NAME / NUM_GPUS / MAX_PRICE and retry." >&2
  exit 1
fi

echo ">> Top offer:"
printf '%s' "$OFFERS_JSON" | python3 -c '
import sys, json
o = json.load(sys.stdin)[0]
print("   offer %s | %sx %s | $%.3f/hr | reliability=%.3f | host_disk_avail=%sGB | net_down=%sMbps" % (
    o.get("id"), o.get("num_gpus"), o.get("gpu_name"),
    o.get("dph_total", 0.0), o.get("reliability2", 0.0),
    o.get("disk_space"), o.get("inet_down")))
'

# Explicit confirmation before spending money.
read -r -p ">> Create this instance? [y/N] " ans
[[ "$ans" == "y" || "$ans" == "Y" ]] || { echo "Aborted."; exit 0; }

# ----------------------------- 2. CREATE --------------------------
CREATE_JSON="$(vastai create instance "$OFFER_ID" \
  --image "$IMAGE" \
  --env "$PORTAL_ENV" \
  --onstart-cmd 'entrypoint.sh' \
  --disk "$DISK" \
  --ssh --direct \
  --raw)"

INSTANCE_ID="$(printf '%s' "$CREATE_JSON" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("new_contract",""))')"

if [[ -z "$INSTANCE_ID" ]]; then
  echo ">> Create failed. Raw response:" >&2
  printf '%s\n' "$CREATE_JSON" >&2
  exit 1
fi
echo ">> Created instance ${INSTANCE_ID}. Waiting for it to come up (image pull can take a while)..."

# ----------------------------- 3. POLL ----------------------------
for i in $(seq 1 80); do
  ST_JSON="$(vastai show instances --raw)"
  STATUS="$(printf '%s' "$ST_JSON" | INSTANCE_ID="$INSTANCE_ID" python3 -c '
import sys, json, os
iid = int(os.environ["INSTANCE_ID"])
for r in json.load(sys.stdin):
    if r.get("id") == iid:
        print(r.get("actual_status") or r.get("cur_state") or "unknown")
        break
else:
    print("missing")
')"
  echo "   [$i] status: ${STATUS}"
  if [[ "$STATUS" == "running" ]]; then
    echo ">> Instance is running."
    break
  fi
  sleep 15
done

# ----------------------------- 4. CONNECT INFO --------------------
echo
echo ">> SSH:"
vastai ssh-url "$INSTANCE_ID" || echo "   (ssh-url not ready yet; rerun: vastai ssh-url ${INSTANCE_ID})"
echo
echo ">> Next: bash cloud/sync_up.sh    # push code + SESGO prompts to the box"
echo ">> Then: bash cloud/at_vast.sh \"bash cloud/at_setup.sh\"   # uv sync on the box"
echo ">> When finished:  bash cloud/vast_destroy.sh --yes-i-am-really-sure"

# Record the ID so the other cloud/ scripts can find it later.
echo "$INSTANCE_ID" > "$HERE/.vast_instance_id"
echo
echo ">> Wrote instance id to $HERE/.vast_instance_id"
