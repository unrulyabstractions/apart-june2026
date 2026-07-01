#!/usr/bin/env bash
# Forking on the NEW xenofobia item (scale-varying, high-uncertainty) for the 4 Qwen3.5
# thinking models, tagged -xeno so outputs are separate from the racismo runs.
set -u
cd "$(dirname "$0")/.."
QID=c4ed674d3758dadf3c91ea71fad9c230
C="THINKING=1 SELECT_FORCE_QID=$QID SELECT_CATEGORIES=xenophobia SELECT_LANGUAGES=es RUN_TAG=-xeno \
GPU_NAME=RTX_4090 DISK=60 MIN_RELIABILITY=0.97 BASE_MAX_NEW_TOKENS=256 MAX_NEW_TOKENS=256 \
N_SAMPLES=20 N_PRIOR=32 TEMPERATURE=1.0"
MODEL=Qwen/Qwen3.5-0.8B GPU_RAM=20 MAX_PRICE=0.80 HF_GEN_MICRO_BATCH=64 IID_FILE=cloud/.xeno_08b.iid \
  env $C nohup bash cloud/run_one_forking_box_32b.sh > cloud/.xeno_08b.log 2>&1 & echo "0.8B PID $!"
MODEL=Qwen/Qwen3.5-2B GPU_RAM=20 MAX_PRICE=0.80 HF_GEN_MICRO_BATCH=48 IID_FILE=cloud/.xeno_2b.iid \
  env $C nohup bash cloud/run_one_forking_box_32b.sh > cloud/.xeno_2b.log 2>&1 & echo "2B PID $!"
MODEL=Qwen/Qwen3.5-4B GPU_RAM=20 MAX_PRICE=0.80 HF_GEN_MICRO_BATCH=32 IID_FILE=cloud/.xeno_4b.iid \
  env $C nohup bash cloud/run_one_forking_box_32b.sh > cloud/.xeno_4b.log 2>&1 & echo "4B PID $!"
MODEL=Qwen/Qwen3.5-9B GPU_RAM=23 MAX_PRICE=0.95 HF_GEN_MICRO_BATCH=10 IID_FILE=cloud/.xeno_9b.iid \
  env $C nohup bash cloud/run_one_forking_box_32b.sh > cloud/.xeno_9b.log 2>&1 & echo "9B PID $!"
echo "launched xeno 0.8/2/4/9B; logs cloud/.xeno_*.log"
