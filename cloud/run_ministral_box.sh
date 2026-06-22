#!/usr/bin/env bash
#
# run_ministral_box.sh — autonomously get the two Ministral-3 models' data on a CUDA
# box (their FP8 weights load on the HF backend ONLY on a GPU; the Mac's CPU/MPS
# dequant path is bugged). Launch -> setup -> run BOTH modes -> pull results -> ALWAYS
# self-destroy (trap on EXIT, so a drop/failure can never leave the box billing).
#
# Results land in jobs/ministral_cloud/  (gitignored quarantine).

set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
export GPU_NAME="${GPU_NAME:-RTX_4090}" MAX_PRICE="${MAX_PRICE:-0.80}" DISK="${DISK:-60}"
LIMIT="${LIMIT:-200}"
REMOTE_ROOT="/root/apart"
DEST="$REPO_ROOT/jobs/ministral_cloud"
mkdir -p "$DEST"
LOG="$DEST/driver.log"
exec >>"$LOG" 2>&1
echo "==== $(date) launching Ministral box (GPU=$GPU_NAME limit=$LIMIT) ===="

# 1. Launch (auto-confirm the y/N) and capture the instance id.
echo y | bash "$HERE/vast_launch.sh"
IID="$(cat "$HERE/.vast_instance_id" 2>/dev/null || echo '')"
[ -n "$IID" ] || { echo "FATAL: no instance id (launch failed)"; exit 1; }
# GUARANTEED teardown no matter how we exit.
trap 'echo "[ministral] destroying $IID"; INSTANCE="$IID" bash "$HERE/vast_destroy.sh" --yes-i-am-really-sure || true' EXIT
echo "[ministral] instance=$IID"

# 2. Wait for sshd to actually answer.
ready=0
for i in $(seq 1 60); do
  INSTANCE="$IID" bash "$HERE/at_vast.sh" "true" >/dev/null 2>&1 && { ready=1; break; }
  sleep 10
done
[ "$ready" = 1 ] || { echo "FATAL: sshd never came up"; exit 1; }

# 3. Push code + SESGO prompts, build the env.
INSTANCE="$IID" bash "$HERE/sync_up.sh" || { echo "FATAL: sync_up"; exit 1; }
INSTANCE="$IID" bash "$HERE/at_vast.sh" "bash cloud/at_setup.sh" || { echo "FATAL: at_setup"; exit 1; }

# 4. Build the dataset on-box, then run BOTH Ministral modes on the HF backend.
# Ministral's FP8 weights need the `kernels` package for the on-GPU fp8 matmul.
# Ministral is UNGATED — no HF token needed (and never embed it in the command, which
# at_vast echoes to the log). uv venvs have no pip, so install kernels via `uv pip`.
RUN="cd $REMOTE_ROOT && uv pip install -q -U kernels && .venv/bin/python -m experiment.generate.build_stability_datasets --out-dir data"
for spec in "mistralai/Ministral-3-3B-Instruct-2512:nonthinking" "mistralai/Ministral-3-3B-Reasoning-2512:thinking"; do
  m="${spec%:*}"; mode="${spec#*:}"
  RUN="$RUN && .venv/bin/python -m experiment.stability.run_greedy_readout --model $m --mode $mode --dataset data/full_prompt_dataset.json --study stability --out-dir out --backend huggingface --limit $LIMIT"
done
INSTANCE="$IID" bash "$HERE/at_vast.sh" "$RUN" || { echo "FATAL: on-box run"; exit 1; }

# 5. Pull the new-layout output (out/<study>/...) DOWN into the quarantine (never out/).
. "$HERE/_ssh_target.sh"; INSTANCE="$IID" _resolve_ssh_target || { echo "FATAL: ssh resolve for pull"; exit 1; }
rsync -av -e "ssh $SSH_EPHEMERAL_OPTS -i ${SSH_KEY:-$HOME/.ssh/id_ed25519} -p $SSH_PORT" \
  "$SSH_USER@$SSH_HOST:$REMOTE_ROOT/out/stability/" "$DEST/" || echo "WARN: rsync pull issue"

echo "[ministral] DONE -> $DEST/ :"
find "$DEST" -name response_samples.json
# trap destroys the box on exit.
