#!/usr/bin/env bash
#
# sync_up.sh — push the apart repo UP to the Vast.ai box (local -> cloud).
#
# This is the ONLY direction that writes to the remote. It is always safe for
# local files: rsync here reads from local and writes to the remote. (The
# dangerous direction — cloud writing onto local — lives in sync_back.sh and is
# deliberately defanged there.)
#
# What it pushes:
#   - the whole repo EXCEPT large/ephemeral/environment-specific dirs, so the box
#     gets the code + sesgo/ + src/ it needs to run the collections.
#   - PLUS a small, EXPLICIT, separate sync of datasets/SESGO/prompts/ (the
#     ~400 KB of .xlsx prompt sources). These are required input for
#     generate_prompt_dataset.py but live under the gitignored, normally-excluded
#     datasets/ tree, so they are synced as their own narrow step.
#
# Excludes (per task spec): out/ datasets/ sync/ .git .venv __pycache__ *.pyc
#                           paper/build  (+ a few obvious caches)
#
# Usage:
#   bash cloud/sync_up.sh
#   INSTANCE=12345678 bash cloud/sync_up.sh
#   DRY_RUN=1 bash cloud/sync_up.sh        # show what WOULD transfer, send nothing

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
INSTANCE_FILE="$HERE/.vast_instance_id"
REMOTE_ROOT="${REMOTE_ROOT:-/root/apart}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

# ── 1. Resolve instance + SSH endpoint (fresh) ─────────────────────────
INSTANCE="${INSTANCE:-}"
if [ -z "$INSTANCE" ]; then
  [ -f "$INSTANCE_FILE" ] || { echo "No instance id. Run cloud/vast_launch.sh first." >&2; exit 1; }
  INSTANCE="$(cat "$INSTANCE_FILE")"
fi

. "$HERE/_ssh_target.sh"
_resolve_ssh_target || exit 1

RSYNC_E="ssh $SSH_EPHEMERAL_OPTS -i $SSH_KEY -p $SSH_PORT"
DRY=""; [ "${DRY_RUN:-0}" = "1" ] && DRY="--dry-run"
# --timeout=120: if the stream stalls (a freshly-booted box whose net is flaky),
# fail fast with a non-zero exit so fleet_run's retry loop re-pushes, rather than
# leaving a half-synced tree behind.
RSYNC_TIMEOUT="--timeout=120"

# Retry rsync on transient broken-pipe / timeout (the Vast network drops mid-stream
# under load); --partial resumes the half-sent tree instead of restarting.
rsync_retry() {
  local n=0
  until rsync -ah $DRY $RSYNC_TIMEOUT --partial -e "$RSYNC_E" "$@"; do
    n=$((n+1)); [ "$n" -ge 6 ] && { echo "[sync_up] rsync FAILED after 6 tries" >&2; return 1; }
    echo "[sync_up] rsync retry $n/6 (transient drop); resuming ..." >&2; sleep 8
  done
}

# ── 2. Push the code tree (local -> remote) ────────────────────────────
# NO --delete here either: we never want to surprise-delete on the remote based
# on local state during an active run. Excludes match the task spec.
echo "[sync_up] code  $REPO_ROOT/  ->  $SSH_HOST:$REMOTE_ROOT/"
# IMPORTANT: anchor these with a leading slash so they match ONLY the top-level
# dirs. An unanchored 'datasets/' would also exclude src/datasets/ (the code!),
# which silently breaks `import src.datasets` on the box.
rsync_retry \
  --exclude='/out/'             \
  --exclude='/datasets/'        \
  --exclude='/sync/'            \
  --exclude='/cloud/.fleet*/'   \
  --exclude='/cloud/.plan*'     \
  --exclude='.git/'             \
  --exclude='.venv/'            \
  --exclude='__pycache__/'      \
  --exclude='*.pyc'             \
  --exclude='paper/build/'      \
  --exclude='.DS_Store'         \
  --exclude='.pytest_cache/'    \
  --exclude='.ruff_cache/'      \
  --exclude='.mypy_cache/'      \
  --exclude='*.egg-info/'       \
  "$REPO_ROOT/" "$SSH_USER@$SSH_HOST:$REMOTE_ROOT/"

# ── 3. Push ONLY the SESGO prompt sources (required generation input) ──
# datasets/ is excluded above (large vendored corpora), but generate_prompt_dataset.py
# reads datasets/SESGO/prompts/*.xlsx (load_items, ~400 KB). Sync just that subtree,
# explicitly and separately, so the remote can regenerate the five datasets.
echo "[sync_up] SESGO prompts  datasets/SESGO/prompts/  ->  $SSH_HOST:$REMOTE_ROOT/datasets/SESGO/prompts/"
# rsync does not create missing parent dirs; ensure the remote target exists.
ssh $SSH_EPHEMERAL_OPTS -i "$SSH_KEY" -p "$SSH_PORT" \
  "$SSH_USER@$SSH_HOST" "mkdir -p $REMOTE_ROOT/datasets/SESGO/prompts"
rsync_retry \
  --exclude='__pycache__/' --exclude='.DS_Store' \
  "$REPO_ROOT/datasets/SESGO/prompts/" \
  "$SSH_USER@$SSH_HOST:$REMOTE_ROOT/datasets/SESGO/prompts/"

echo "[sync_up] done."
echo "[sync_up] next: bash cloud/at_vast.sh \"bash cloud/at_setup.sh\""
