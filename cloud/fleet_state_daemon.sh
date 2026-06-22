#!/usr/bin/env bash
#
# fleet_state_daemon.sh — ONE shared poller for the WHOLE fleet's instance state.
#
# THE SCALING FIX. Before this, every box resolved its own status + SSH endpoint by
# walking EVERY page of `vastai show instances-v1` ON EVERY POLL. With N boxes that
# is O(N) full paginated API walks every 10-15s — at 100 boxes, hundreds-to-
# thousands of vastai calls per cycle, which the API throttles, so endpoints stop
# resolving and otherwise-healthy boxes get wasted. THIS daemon walks all pages ONCE
# per cycle for the whole account and writes a single cache file; every box then
# reads the cache (one local file read) instead of hitting the API. The API load is
# now ~3 calls/cycle REGARDLESS of fleet size — the thing that makes 100+ seamless.
#
# Output cache (atomic): $FLEET_DIR/instances_state.json
#   { "updated": <epoch>, "instances": { "<id>": {status, ssh_host, ssh_port,
#     gpu_util, intended} , ... } }
#
# FAIL-SAFE: if a cycle's API call errors or returns empty, the LAST GOOD cache is
# kept (never overwritten with nothing) — a transient API flap can't blank the cache
# and strand the fleet. Runs until $FLEET_DIR/.state_daemon.stop appears (or the
# fleet dir disappears). Singleton via a pidfile.
#
# Usage:
#   bash cloud/fleet_state_daemon.sh &              # background, default 15s cycle
#   STATE_INTERVAL=20 bash cloud/fleet_state_daemon.sh &
#   touch cloud/.fleet/.state_daemon.stop           # ask it to exit

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
FLEET_DIR="${FLEET_DIR:-$HERE/.fleet}"
INTERVAL="${STATE_INTERVAL:-15}"
STATE_FILE="$FLEET_DIR/instances_state.json"
STOP_FILE="$FLEET_DIR/.state_daemon.stop"
PID_FILE="$FLEET_DIR/.state_daemon.pid"

mkdir -p "$FLEET_DIR"

# Singleton: if a live daemon already owns the pidfile, do not start a second one
# (two daemons racing the same atomic write is harmless, but wasteful + confusing).
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null; then
  echo "[state-daemon] already running (pid $(cat "$PID_FILE")); not starting another." >&2
  exit 0
fi
echo "$$" >"$PID_FILE"
rm -f "$STOP_FILE"

# One full paginated walk -> the cache JSON on stdout. Exits non-zero (prints
# nothing) if the API gave nothing usable, so the caller keeps the last good cache.
_walk_all_pages() {
  python3 -c '
import json, subprocess, sys, time

instances = {}
token = None
got_any_page = False
for _ in range(60):  # hard page cap: 60*25 = 1500 instances, far past any real fleet
    cmd = ["vastai", "show", "instances-v1", "--raw"]
    if token:
        cmd += ["--next-token", token]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        d = json.loads(r.stdout)
    except Exception:
        # A failed page mid-walk: stop and emit what we have ONLY if we got >=1 page,
        # else signal total failure so the caller preserves the previous cache.
        break
    got_any_page = True
    rows = d if isinstance(d, list) else d.get("instances", d.get("results", []))
    for row in rows:
        iid = row.get("id")
        if iid is None:
            continue
        instances[str(iid)] = {
            "status": row.get("actual_status") or row.get("cur_state") or "unknown",
            "intended": row.get("intended_status") or "",
            "ssh_host": row.get("ssh_host") or "",
            "ssh_port": row.get("ssh_port") or "",
            "gpu_util": row.get("gpu_util"),
        }
    token = d.get("next_token") if isinstance(d, dict) else None
    if not token or not rows:
        break

if not got_any_page:
    sys.exit(1)  # total API failure -> caller keeps last good cache
print(json.dumps({"updated": int(time.time()), "instances": instances}))
'
}

# Atomic publish: write a tmp file in the same dir, then mv (rename is atomic on the
# same filesystem) so a reader never sees a half-written cache.
_publish() {
  local json="$1" tmp
  tmp="$(mktemp "$FLEET_DIR/.state.XXXXXX")"
  printf '%s\n' "$json" >"$tmp" && mv -f "$tmp" "$STATE_FILE"
}

echo "[state-daemon] pid $$ polling every ${INTERVAL}s -> $STATE_FILE"
cycles=0
while :; do
  [ -e "$STOP_FILE" ] && { echo "[state-daemon] stop file seen; exiting."; break; }
  [ -d "$FLEET_DIR" ] || { echo "[state-daemon] fleet dir gone; exiting."; break; }

  if json="$(_walk_all_pages)" && [ -n "$json" ]; then
    _publish "$json"
    cycles=$((cycles + 1))
  else
    echo "[state-daemon] API gave nothing this cycle; keeping last good cache." >&2
  fi
  sleep "$INTERVAL"
done

rm -f "$PID_FILE"
