#!/usr/bin/env bash
#
# sync_back.sh — pull remote results DOWN to a LOCAL quarantine dir (cloud -> local).
#
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ SAFETY GUARANTEE (the whole point of this script):                        ║
# ║                                                                           ║
# ║   1. It writes ONLY to the local gitignored  sync/  directory.            ║
# ║      It NEVER targets out/ or any code path. The cloud cannot touch out/. ║
# ║                                                                           ║
# ║   2. It uses  rsync --ignore-existing  — so a file that already exists    ║
# ║      locally in sync/ is NEVER overwritten. Only brand-NEW files are      ║
# ║      copied down.                                                         ║
# ║                                                                           ║
# ║   3. There is NO --delete. Nothing local is ever removed.                 ║
# ║                                                                           ║
# ║ Net effect: the worst the cloud can do is add new files under sync/.      ║
# ║ Your out/, your code, and your git tree are untouchable from here.        ║
# ║                                                                           ║
# ║ After this runs, INSPECT sync/ by hand, then promote with merge_sync.sh.  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Usage:
#   bash cloud/sync_back.sh
#   INSTANCE=12345678 bash cloud/sync_back.sh
#   DRY_RUN=1 bash cloud/sync_back.sh        # show what WOULD copy, copy nothing

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
INSTANCE_FILE="$HERE/.vast_instance_id"
REMOTE_ROOT="${REMOTE_ROOT:-/root/apart}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

# Hard-coded local destination. Intentionally NOT configurable to out/ — the
# whole safety story is that the cloud writes only into sync/.
LOCAL_SYNC="$REPO_ROOT/sync"

# ── 1. Resolve instance + SSH endpoint (fresh) ─────────────────────────
INSTANCE="${INSTANCE:-}"
if [ -z "$INSTANCE" ]; then
  [ -f "$INSTANCE_FILE" ] || { echo "No instance id. Run cloud/vast_launch.sh first." >&2; exit 1; }
  INSTANCE="$(cat "$INSTANCE_FILE")"
fi

SSH_URL="$(vastai ssh-url "$INSTANCE" 2>/dev/null | tr -d '\n')"
if [[ "$SSH_URL" =~ ^ssh://([^@]+)@([^:]+):([0-9]+)$ ]]; then
  SSH_USER="${BASH_REMATCH[1]}"; SSH_HOST="${BASH_REMATCH[2]}"; SSH_PORT="${BASH_REMATCH[3]}"
else
  echo "Could not parse ssh-url: '$SSH_URL'. Is instance $INSTANCE running?" >&2
  exit 1
fi

RSYNC_E="ssh -F /dev/null -o StrictHostKeyChecking=accept-new -i $SSH_KEY -p $SSH_PORT"
DRY=""; [ "${DRY_RUN:-0}" = "1" ] && DRY="--dry-run"

# The two result trees we collected on the box.
STUDIES=(divergence stability)

for study in "${STUDIES[@]}"; do
  src="$SSH_USER@$SSH_HOST:$REMOTE_ROOT/out/sesgo/$study/"
  dst="$LOCAL_SYNC/sesgo/$study/"
  mkdir -p "$dst"
  echo "[sync_back] $study  $SSH_HOST:.../out/sesgo/$study/  ->  sync/sesgo/$study/  (--ignore-existing, no --delete)"
  # -av  : archive + verbose
  # --ignore-existing : NEVER overwrite a file already present locally
  # (NO --delete) : NEVER remove anything local
  rsync -av $DRY --ignore-existing -e "$RSYNC_E" "$src" "$dst"
done

echo
echo "[sync_back] done. New files (if any) are quarantined under: $LOCAL_SYNC/sesgo/"
echo "[sync_back] 1) INSPECT them:   find $LOCAL_SYNC -type f"
echo "[sync_back] 2) PROMOTE to out: bash cloud/merge_sync.sh"
