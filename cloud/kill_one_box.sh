#!/usr/bin/env bash
# Destroy ONE stability box by tag (uses its recorded instance id). Never touches others.
#   bash cloud/kill_one_box.sh <tag>
t="${1:?tag}"; cd "$(dirname "$0")/.."
iid="$(cat "cloud/.stab_${t}.iid" 2>/dev/null)"
[ -n "$iid" ] || { echo "no .iid for $t"; exit 1; }
echo "destroying $t instance $iid"
INSTANCE="$iid" bash cloud/vast_destroy.sh --yes-i-am-really-sure
