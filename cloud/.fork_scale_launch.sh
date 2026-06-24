#!/usr/bin/env bash
# Launch Qwen3.5 2B/4B/9B THINKING forking (forced racismo qid) in parallel; each box
# auto-destroys on exit. Data-collection runs (base cap 256); figures re-plotted locally.
set -u
cd "$(dirname "$0")/.."
QID=12f12e3218e086624b88c098137bbf28
common() { echo "THINKING=1 SELECT_FORCE_QID=$QID SELECT_CATEGORIES=racism SELECT_LANGUAGES=es \
GPU_NAME=RTX_4090 MAX_PRICE=0.80 DISK=60 MIN_RELIABILITY=0.95 \
BASE_MAX_NEW_TOKENS=256 MAX_NEW_TOKENS=256 N_SAMPLES=20 N_PRIOR=32 TEMPERATURE=1.0"; }
MODEL=Qwen/Qwen3.5-2B GPU_RAM=20 HF_GEN_MICRO_BATCH=48 IID_FILE=cloud/.fork_2b.iid \
  env $(common) nohup bash cloud/run_one_forking_box_32b.sh > cloud/.fork_2b.log 2>&1 & echo "2B PID $!"
MODEL=Qwen/Qwen3.5-4B GPU_RAM=20 HF_GEN_MICRO_BATCH=32 IID_FILE=cloud/.fork_4b.iid \
  env $(common) nohup bash cloud/run_one_forking_box_32b.sh > cloud/.fork_4b.log 2>&1 & echo "4B PID $!"
MODEL=Qwen/Qwen3.5-9B GPU_RAM=23 HF_GEN_MICRO_BATCH=10 MAX_PRICE=0.95 IID_FILE=cloud/.fork_9b.iid \
  env $(common) nohup bash cloud/run_one_forking_box_32b.sh > cloud/.fork_9b.log 2>&1 & echo "9B PID $!"
echo "launched 2B/4B/9B; logs cloud/.fork_{2b,4b,9b}.log"
