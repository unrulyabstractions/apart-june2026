#!/usr/bin/env bash
# Re-run the 6 smallest stability models (n=10) with the fixed reasoning detection +
# parser + label_prob/vocab_diversity schema. No output filtering (silent crashes hide
# behind greps). Each model's rc is asserted; a non-zero rc aborts loudly.
set -uo pipefail
cd "$(dirname "$0")"
rm -rf out/stability/*

run() {  # run <backend-flag> <mode> <model>
  local flag="$1" mode="$2" model="$3"
  echo "=================================================================="
  echo ">>> RUN model=$model mode=$mode $flag"
  echo "=================================================================="
  uv run python -m experiment.stability.run_greedy_readout \
      --model "$model" --mode "$mode" --limit 10 $flag
  local rc=$?
  echo ">>> rc=$rc for $model ($mode)"
  if [ $rc -ne 0 ]; then echo "!!! ABORT: $model rc=$rc"; exit $rc; fi
}

run "--backend huggingface" nonthinking google/gemma-4-E2B-it
run ""                      nonthinking meta-llama/Llama-3.2-1B-Instruct
run ""                      nonthinking mlx-community/Ministral-3-3B-Instruct-2512-4bit
run ""                      thinking    mlx-community/Ministral-3-3B-Reasoning-2512-4bit
run ""                      nonthinking Qwen/Qwen3.5-0.8B
run ""                      thinking    Qwen/Qwen3.5-0.8B
echo "ALL SIX DONE"
