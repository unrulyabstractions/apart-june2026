#!/usr/bin/env bash
#
# merge_sync.sh — promote NEW results from the local quarantine sync/ into out/.
#
# This is a PURELY LOCAL step. The cloud is not involved. It exists so a human
# can first inspect everything sync_back.sh quarantined under sync/, and only
# then copy the new files into the canonical out/ tree.
#
# Safety, same spirit as sync_back.sh:
#   - rsync --ignore-existing : an existing out/ file is NEVER overwritten.
#   - NO --delete             : nothing in out/ is ever removed.
# So the worst this can do is ADD new files to out/. Existing results are safe.
#
# By default it COPIES (leaving sync/ intact as a record). Pass --move to remove
# the source files from sync/ after a successful copy.
#
# Usage:
#   bash cloud/merge_sync.sh             # copy new sync/ files into out/
#   DRY_RUN=1 bash cloud/merge_sync.sh   # show what WOULD copy, copy nothing
#   bash cloud/merge_sync.sh --move      # copy then clear the promoted files from sync/

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
LOCAL_SYNC="$REPO_ROOT/sync"
OUT="$REPO_ROOT/out"

MOVE=0
[ "${1:-}" = "--move" ] && MOVE=1
DRY=""; [ "${DRY_RUN:-0}" = "1" ] && DRY="--dry-run"

if [ ! -d "$LOCAL_SYNC" ]; then
  echo "Nothing to merge: $LOCAL_SYNC does not exist. Run cloud/sync_back.sh first." >&2
  exit 0
fi

REMOVE=""; [ $MOVE -eq 1 ] && REMOVE="--remove-source-files"

echo "[merge_sync] sync/  ->  out/   (--ignore-existing, no --delete${MOVE:+, --remove-source-files})"
# Mirror the whole sync/ tree into out/, adding only files out/ doesn't have.
rsync -av $DRY --ignore-existing $REMOVE "$LOCAL_SYNC/" "$OUT/"

if [ $MOVE -eq 1 ] && [ -z "$DRY" ]; then
  # --remove-source-files leaves empty dirs behind; prune them.
  find "$LOCAL_SYNC" -type d -empty -delete 2>/dev/null || true
  echo "[merge_sync] promoted files removed from sync/ (empty dirs pruned)."
fi

echo "[merge_sync] done. Canonical results now under: $OUT/sesgo/"
