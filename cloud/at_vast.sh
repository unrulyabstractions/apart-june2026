#!/usr/bin/env bash
#
# at_vast.sh — run a single command on the Vast.ai box over SSH.
#
# This is the thin "send a command to the remote and stream its output back"
# wrapper. It does NOT sync anything by itself — pushing code is sync_up.sh and
# pulling results is sync_back.sh, kept as SEPARATE scripts so the dangerous
# direction (cloud -> local) is always explicit and isolated.
#
# It resolves the SSH host/port FRESH from `vastai ssh-url` on every call,
# because Vast IPs/ports change after a stop/start.
#
# Usage:
#   bash cloud/at_vast.sh "nvidia-smi"
#   bash cloud/at_vast.sh "cd apart && uv run python -c 'import torch; print(torch.cuda.is_available())'"
#   INSTANCE=12345678 bash cloud/at_vast.sh "<cmd>"      # override instance id
#
# Requirements:
#   - cloud/.vast_instance_id (written by vast_launch.sh) OR INSTANCE env / arg.
#   - ~/.ssh/id_ed25519 — your SSH key, registered on the Vast account.
#   - vastai CLI installed and authenticated.
#
# The remote repo root is /root/apart (set once in REMOTE_ROOT below). Commands
# are run from there.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
INSTANCE_FILE="$HERE/.vast_instance_id"
REMOTE_ROOT="${REMOTE_ROOT:-/root/apart}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

# ── 1. Resolve the active instance ID (env > arg-file > recorded file) ──
INSTANCE="${INSTANCE:-}"
if [ -z "$INSTANCE" ]; then
  [ -f "$INSTANCE_FILE" ] || {
    echo "No instance id. Set INSTANCE=<id> or run: bash cloud/vast_launch.sh" >&2
    exit 1
  }
  INSTANCE="$(cat "$INSTANCE_FILE")"
fi

[ $# -gt 0 ] || { echo "usage: bash cloud/at_vast.sh <command>" >&2; exit 1; }
CMD="$*"

# ── 2. Resolve SSH host / port FRESH each call (v1 API; see _ssh_target.sh) ──
. "$HERE/_ssh_target.sh"
_resolve_ssh_target || exit 1

# ConnectTimeout bounds the wait_ssh readiness probe (a freshly-'running' box whose
# sshd is not up yet fails FAST instead of hanging on the full TCP timeout);
# ServerAlive* tears down a stalled session so long on-box steps don't wedge.
SSH="ssh -F /dev/null -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=8 -i $SSH_KEY -p $SSH_PORT $SSH_USER@$SSH_HOST"

# ── 3. Run the command on the remote box ───────────────────────────────
echo "[at_vast] run on $SSH_HOST:$SSH_PORT  ::  $CMD"
exec $SSH "cd $REMOTE_ROOT && $CMD"
