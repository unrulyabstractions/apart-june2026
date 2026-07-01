#!/usr/bin/env bash
# Re-run 2B then 9B SEQUENTIALLY (avoids the parallel-rsync broken-pipe that killed them).
set -u
cd "$(dirname "$0")/.."
QID=12f12e3218e086624b88c098137bbf28
C="THINKING=1 SELECT_FORCE_QID=$QID SELECT_CATEGORIES=racism SELECT_LANGUAGES=es \
GPU_NAME=RTX_4090 DISK=60 MIN_RELIABILITY=0.95 BASE_MAX_NEW_TOKENS=256 MAX_NEW_TOKENS=256 \
N_SAMPLES=20 N_PRIOR=32 TEMPERATURE=1.0"
echo "=== [seq] 2B start $(date +%H:%M:%S) ==="
MODEL=Qwen/Qwen3.5-2B GPU_RAM=20 MAX_PRICE=0.80 HF_GEN_MICRO_BATCH=48 IID_FILE=cloud/.fork_2b.iid \
  env $C bash cloud/run_one_forking_box_32b.sh > cloud/.fork_2b.log 2>&1
echo "=== [seq] 2B done rc=$? $(date +%H:%M:%S) ==="
echo "=== [seq] 9B start $(date +%H:%M:%S) ==="
MODEL=Qwen/Qwen3.5-9B GPU_RAM=23 MAX_PRICE=0.95 HF_GEN_MICRO_BATCH=10 IID_FILE=cloud/.fork_9b.iid \
  env $C bash cloud/run_one_forking_box_32b.sh > cloud/.fork_9b.log 2>&1
echo "=== [seq] 9B done rc=$? $(date +%H:%M:%S) ==="
