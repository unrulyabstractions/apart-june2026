#!/usr/bin/env bash
set -u; cd "$(dirname "$0")/.."
qid(){ case $1 in racismo)echo fa413d87400592dd051c38c115f662bb;; xenofobia)echo f682248452559dbdecea7b4e7f4e8a22;; clasismo)echo bc16eaabf5abf62db57b15eabee088ac;; genero)echo 28fa1cf1878a21f096bd3c85128e9eff;; esac; }
alias_of(){ case $1 in racismo)echo racism;; xenofobia)echo xenophobia;; clasismo)echo classism;; genero)echo gender;; esac; }
mb(){ case $1 in 0.8B)echo 64;; 2B)echo 48;; 4B)echo 32;; 9B)echo 10;; esac; }
gr(){ case $1 in 9B)echo 23;; *)echo 20;; esac; }
pr(){ case $1 in 9B)echo 0.95;; *)echo 0.80;; esac; }
for cat in racismo xenofobia clasismo genero; do
  for m in 0.8B 2B 4B 9B; do
    tag="${cat:0:3}sh"
    MODEL=Qwen/Qwen3.5-$m THINKING=1 SHARED_BASE_FILE=cloud/shared_bases/$cat.json \
    SELECT_FORCE_QID=$(qid $cat) SELECT_CATEGORIES=$(alias_of $cat) SELECT_LANGUAGES=es RUN_TAG=-$tag \
    GPU_NAME=RTX_4090 GPU_RAM=$(gr $m) MAX_PRICE=$(pr $m) DISK=60 MIN_RELIABILITY=0.97 INET_DOWN=400 \
    MAX_NEW_TOKENS=256 N_SAMPLES=16 N_PRIOR=24 TEMPERATURE=1.0 POSITION_STRIDE=3 HF_GEN_MICRO_BATCH=$(mb $m) \
    IID_FILE=cloud/.sh_${cat:0:3}_$m.iid \
      nohup bash cloud/run_one_forking_box_32b.sh > cloud/.sh_${cat:0:3}_$m.log 2>&1 &
    echo "$cat $m PID $!"; sleep 2
  done
done
echo "launched 16 shared-base forking boxes"
