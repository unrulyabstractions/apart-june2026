#!/usr/bin/env bash
#
# sync_partial.sh — keep a box's IN-PROGRESS response_samples.json mirrored to a
# LOCAL partial-quarantine, and push it back UP to a (possibly fresh, replacement)
# box so a relaunched collect RESUMES instead of restarting from 0.
#
# WHY THIS EXISTS (req. C — crash-graceful box REPLACEMENT resume):
#   collect already checkpoints response_samples.json every 25 samples and, on
#   restart, RESUMES from an existing response_samples.json at its out path (the
#   SesgoQuerier loads it, computes completed identities, and skips them). That
#   makes a box restarting its OWN collect resumable for free. But if a box is
#   DESTROYED and REPLACED by a fresh one, the partial died with its disk. This
#   helper closes that gap: the fleet PULLs each box's partial periodically (mode
#   pull) into sync/partial-box-<tag>/, and on (re)launch PUSHes the newest partial
#   UP to the fresh box (mode push) so its collect resumes from it.
#
# SAFETY / ISOLATION (deliberately separate from sync_back.sh / merge_sync.sh):
#   - pull writes ONLY into sync/partial-box-<tag>/ (its OWN tree). It is NEVER
#     promoted: merge_sync.sh globs box-*/ only, so partial-box-*/ is invisible to
#     promotion and can never reach out/. The authoritative, completed results
#     still flow through sync_back.sh -> sync/box-<tag>/ -> merge_sync at the end.
#   - pull mirrors WITH overwrite (the partial GROWS, so the newest copy must win)
#     but, again, only inside the throwaway partial- tree — never out/, never the
#     box-<tag> quarantine, never code.
#   - push writes ONLY this study/model/shard's response_samples.json onto the box,
#     under its own out/sesgo/<study>/<bare>/[shard_k_of_K]/ slice. collect then
#     resumes from it. No other remote path is touched.
#   - Every operation is a NO-OP when the relevant file is absent (fresh study, no
#     prior partial), so adding these calls is backward-compatible.
#
# Usage:
#   TAG=Qwen3-0.6B__shard0of3 MODEL=Qwen/Qwen3-0.6B STUDIES="divergence" \
#     SHARD_INDEX=0 SHARD_COUNT=3 INSTANCE=123 bash cloud/sync_partial.sh pull
#   TAG=...                                            INSTANCE=123 bash cloud/sync_partial.sh push
#
# Env:
#   MODE          : "pull" (remote->local partial) or "push" (local partial->remote).
#                   May also be passed as $1.
#   INSTANCE      : box id (required; resolved to SSH endpoint via _ssh_target.sh).
#   TAG           : box tag, e.g. Qwen3-0.6B__shard0of3 (names the partial subtree).
#   MODEL         : HF repo id; bare name scopes the out slice.
#   STUDIES       : space-separated studies this box runs (default "divergence stability").
#   SHARD_INDEX   : 0-based shard (default 0).
#   SHARD_COUNT   : total shards (default 1). >1 nests the shard_<k>_of_<K>/ slice.

set -uo pipefail   # NOT -e: a partial sync must never abort the caller's run loop.

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
REMOTE_ROOT="${REMOTE_ROOT:-/root/apart}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

MODE="${MODE:-${1:-pull}}"
INSTANCE="${INSTANCE:-}"
TAG="${TAG:-}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
read -r -a STUDIES <<< "${STUDIES:-divergence stability}"

[ -n "$INSTANCE" ] || { echo "[sync_partial] no INSTANCE; skip" >&2; exit 0; }
[ -n "$TAG" ]      || { echo "[sync_partial] no TAG; skip" >&2; exit 0; }

BARE="${MODEL##*/}"
# The on-box per-shard out slice mirrors shard_out_dir() in the collect scripts:
# multi-shard runs nest shard_<k>_of_<K>/, single-shard runs do not.
SHARD_SEG=""
[ "${SHARD_COUNT:-1}" -gt 1 ] 2>/dev/null && SHARD_SEG="/shard_${SHARD_INDEX}_of_${SHARD_COUNT}"

# Resolve the SSH endpoint fresh (Vast ports change after a stop/start/replace).
. "$HERE/_ssh_target.sh"
_resolve_ssh_target || { echo "[sync_partial] cannot resolve SSH for $INSTANCE; skip" >&2; exit 0; }

SSH_OPTS="-F /dev/null -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=6 -i $SSH_KEY -p $SSH_PORT"
RSYNC_E="ssh $SSH_OPTS"

# Local partial quarantine for THIS box — throwaway, never promoted (box-* only is).
LOCAL_PARTIAL="$REPO_ROOT/sync/partial-box-$TAG"

pull_one() {  # remote response_samples.json -> local partial (overwrite: newest wins)
  local study="$1"
  local rel="sesgo/$study/$BARE$SHARD_SEG/response_samples.json"
  local src="$SSH_USER@$SSH_HOST:$REMOTE_ROOT/out/$rel"
  local dst="$LOCAL_PARTIAL/$rel"
  mkdir -p "$(dirname "$dst")"
  # NO --ignore-existing here (we WANT the growing partial to overwrite the stale
  # local copy) and --timeout so a flaky box never wedges the loop. A missing remote
  # file just yields a non-zero rsync we swallow (the study may not have started).
  rsync -a --timeout=60 -e "$RSYNC_E" "$src" "$dst" 2>/dev/null \
    && echo "[sync_partial pull] $study  <- box  ($(wc -c <"$dst" 2>/dev/null || echo 0) B)" \
    || true
}

push_one() {  # local partial -> remote out slice (so collect resumes from it)
  local study="$1"
  local rel="sesgo/$study/$BARE$SHARD_SEG/response_samples.json"
  local src="$LOCAL_PARTIAL/$rel"
  [ -f "$src" ] || return 0   # no prior partial for this study -> nothing to resume
  local dst="$SSH_USER@$SSH_HOST:$REMOTE_ROOT/out/$rel"
  # rsync does not create remote parents; mkdir the slice dir first.
  ssh $SSH_OPTS "$SSH_USER@$SSH_HOST" "mkdir -p $REMOTE_ROOT/out/$(dirname "$rel")" 2>/dev/null || return 0
  rsync -a --timeout=60 -e "$RSYNC_E" "$src" "$dst" 2>/dev/null \
    && echo "[sync_partial push] $study  -> box  (resume from $(wc -c <"$src") B)" \
    || true
}

for study in "${STUDIES[@]}"; do
  case "$MODE" in
    pull) pull_one "$study" ;;
    push) push_one "$study" ;;
    *) echo "[sync_partial] unknown MODE '$MODE' (want pull|push)" >&2; exit 0 ;;
  esac
done
