#!/usr/bin/env bash
#
# vast_destroy.sh — stop billing on the Vast.ai instance.
#
# ╔══════════════════════════════════════════════════════════════════════╗
# ║ SAFETY GUARD: requires an explicit  --yes-i-am-really-sure  flag to    ║
# ║ actually act. Without it, the script just prints what it WOULD do and  ║
# ║ exits. This prevents accidentally tearing down a box mid-run.          ║
# ╚══════════════════════════════════════════════════════════════════════╝
#
# IMPORTANT: pull your results back FIRST (cloud/sync_back.sh). Destroying the
# instance deletes its disk, including out/sesgo/, permanently.
#
# Usage:
#   bash cloud/vast_destroy.sh                                # dry-run (default)
#   bash cloud/vast_destroy.sh --list                         # show instances
#   bash cloud/vast_destroy.sh --stop --yes-i-am-really-sure  # pause (disk still billed)
#   bash cloud/vast_destroy.sh --yes-i-am-really-sure         # destroy (ALL billing stops)
#   bash cloud/vast_destroy.sh 12345 --yes-i-am-really-sure   # destroy a specific id
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
STORE="$HERE/.vast_instance_id"

command -v vastai >/dev/null || { echo "vastai not found. pip install vastai" >&2; exit 1; }

# ── 1. Parse arguments ──────────────────────────────────────────────────
ACTION="destroy"
INSTANCE=""
CONFIRMED=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stop)                   ACTION="stop";    shift ;;
    --list)                   ACTION="list";    shift ;;
    --yes-i-am-really-sure)   CONFIRMED=1;      shift ;;
    -h|--help)                sed -n '1,30p' "$0"; exit 0 ;;
    *)                        INSTANCE="$1";    shift ;;
  esac
done

# ── 2. List mode ────────────────────────────────────────────────────────
if [ "$ACTION" = "list" ]; then
  vastai show instances-v1
  exit 0
fi

# ── 3. Resolve which instance to act on ─────────────────────────────────
if [ -z "$INSTANCE" ]; then
  if [ -f "$STORE" ]; then
    INSTANCE="$(cat "$STORE")"
    echo ">> Resolved instance ${INSTANCE} from ${STORE}"
  else
    echo "No instance id given and ${STORE} not found." >&2
    vastai show instances-v1 >&2 || true
    exit 1
  fi
fi

# ── 4. SAFETY: refuse to act without explicit confirmation flag ─────────
if [ $CONFIRMED -eq 0 ]; then
  cat <<EOF
[dry-run] would ${ACTION} instance ${INSTANCE}

Refusing to proceed without explicit confirmation. To actually ${ACTION},
re-run with the --yes-i-am-really-sure flag:

    bash cloud/vast_destroy.sh${ACTION:+ --${ACTION}} ${INSTANCE} --yes-i-am-really-sure

REMINDER: destroying deletes the box's disk (out/sesgo/ included). Make sure you
already ran cloud/sync_back.sh and have your results locally.
EOF
  exit 0
fi

# ── 5. Actually act ─────────────────────────────────────────────────────
if [ "$ACTION" = "stop" ]; then
  vastai stop instance "$INSTANCE"
  echo ">> Stopped instance ${INSTANCE}. Disk still being billed."
  echo ">> Restart with: vastai start instance ${INSTANCE}"
else
  # The script's own --yes-i-am-really-sure already confirmed; auto-answer the
  # CLI's inner [y/N] prompt so teardown is non-interactive.
  printf 'y\n' | vastai destroy instance "$INSTANCE"
  echo ">> Destroyed instance ${INSTANCE}. All billing stopped."
  rm -f "$STORE"
fi
