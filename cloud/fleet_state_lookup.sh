#!/usr/bin/env bash
#
# fleet_state_lookup.sh — read instance status / SSH endpoint from the SHARED cache
# written by fleet_state_daemon.sh, instead of each box walking the API itself.
#
# Sourced (not executed). Provides:
#   cached_instance_status <iid>     -> echoes running|stopped|...|missing
#   cached_ssh_target       <iid>    -> sets SSH_USER / SSH_HOST / SSH_PORT (rc!=0 if unknown)
#
# Both prefer the fresh cache ($FLEET_DIR/instances_state.json). If the cache is
# missing, older than STATE_MAX_AGE, or lacks this id, they fall back to ONE live
# resolve for that single id (the old behaviour) — so single-box usage with no
# daemon still works, while a running daemon makes every lookup a local file read.
#
# This is what lets the fleet scale: with the daemon up, 1 box or 200 boxes cost the
# SAME ~3 API calls/cycle total, because nobody polls the API per-box anymore.

# Resolve FLEET_DIR + the live single-id resolver, regardless of caller's CWD.
_LOOKUP_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLEET_DIR="${FLEET_DIR:-$_LOOKUP_HERE/.fleet}"
STATE_FILE="${STATE_FILE:-$FLEET_DIR/instances_state.json}"
STATE_MAX_AGE="${STATE_MAX_AGE:-70}"   # daemon polls ~15s; tolerate a few missed cycles
# shellcheck source=/dev/null
. "$_LOOKUP_HERE/_ssh_target.sh"   # provides _resolve_ssh_target (live, single id)

# Print one field for <iid> from the cache IFF the cache is present AND fresh AND has
# the id. rc!=0 (prints nothing) means "cache can't answer — fall back to live".
_state_field() {
  local iid="$1" field="$2"
  [ -f "$STATE_FILE" ] || return 1
  IID="$iid" FIELD="$field" MAXAGE="$STATE_MAX_AGE" STATE_FILE="$STATE_FILE" python3 -c '
import json, os, sys, time
try:
    with open(os.environ["STATE_FILE"]) as f:
        d = json.load(f)
except Exception:
    sys.exit(1)
if int(time.time()) - int(d.get("updated", 0)) > int(os.environ["MAXAGE"]):
    sys.exit(1)  # stale -> caller goes live
inst = d.get("instances", {}).get(os.environ["IID"])
if not inst:
    sys.exit(1)  # id not in cache -> caller goes live
val = inst.get(os.environ["FIELD"])
if val in (None, ""):
    sys.exit(1)
print(val)
'
}

# Status, cache-first. Falls back to a single live resolve only if the cache can't
# answer (no daemon / stale / id absent). Always prints SOMETHING (…|missing).
cached_instance_status() {
  local iid="$1" st
  if st="$(_state_field "$iid" status)"; then
    printf '%s\n' "$st"; return 0
  fi
  # Live fallback: resolve just this id's status (bounded paginated walk).
  INSTANCE_ID="$iid" python3 -c '
import json, os, subprocess, sys
iid = int(os.environ["INSTANCE_ID"]); token = None
for _ in range(60):
    cmd = ["vastai", "show", "instances-v1", "--raw"]
    if token: cmd += ["--next-token", token]
    try:
        d = json.loads(subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout)
    except Exception:
        print("missing"); sys.exit()
    rows = d if isinstance(d, list) else d.get("instances", d.get("results", []))
    for r in rows:
        if r.get("id") == iid:
            print(r.get("actual_status") or r.get("cur_state") or "unknown"); sys.exit()
    token = d.get("next_token") if isinstance(d, dict) else None
    if not token or not rows: break
print("missing")'
}

# SSH endpoint, cache-first. Sets SSH_USER/SSH_HOST/SSH_PORT. rc!=0 if unresolved.
cached_ssh_target() {
  local iid="$1" host port
  host="$(_state_field "$iid" ssh_host)" && port="$(_state_field "$iid" ssh_port)"
  if [ -n "${host:-}" ] && [ -n "${port:-}" ]; then
    SSH_USER="root"; SSH_HOST="$host"; SSH_PORT="$port"; return 0
  fi
  # Live fallback (no daemon / stale / absent): the original single-id resolver.
  INSTANCE="$iid" _resolve_ssh_target
}
