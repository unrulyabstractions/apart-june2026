#!/usr/bin/env bash
#
# _ssh_target.sh — resolve SSH_USER / SSH_HOST / SSH_PORT for $INSTANCE.
#
# Sourced by at_vast.sh, sync_up.sh, sync_back.sh. NOT executable on its own.
#
# Why this exists: vastai CLI 1.0.13 deprecated BOTH `vastai ssh-url` and
# `vastai show instances` — they hit the v0 API and now return HTTP 410
# ("/api/v0/instances/ is deprecated. Use /api/v1/instances/ instead."). The
# instance's SSH endpoint is still available from the paginated v1 listing as the
# top-level `ssh_host` / `ssh_port` fields, so we read it from there. User is root.
#
# CRITICAL: instances-v1 PAGINATES (25/page). A single `--raw` page only shows the
# first 25, so when the account holds >25 instances (many concurrent fleets) a box
# on a LATER page resolves to an EMPTY ssh endpoint — and every wait_ssh / sync_up
# then fails forever, destroying otherwise-healthy boxes. So we WALK EVERY PAGE via
# next_token until the instance is found. (subprocess-driven so the loop can issue
# one CLI call per page from inside python.)
#
# Requires $INSTANCE to be set before calling _resolve_ssh_target.

_resolve_ssh_target() {
  local line
  line="$(INSTANCE="$INSTANCE" python3 -c '
import json, os, subprocess, sys
iid = int(os.environ["INSTANCE"])
token = None
for _ in range(40):  # hard page cap so a runaway token loop can never hang
    cmd = ["vastai", "show", "instances-v1", "--raw"]
    if token:
        cmd += ["--next-token", token]
    try:
        d = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)
    except Exception:
        break
    rows = d if isinstance(d, list) else d.get("instances", d.get("results", []))
    for r in rows:
        if r.get("id") == iid:
            print((r.get("ssh_host") or ""), (r.get("ssh_port") or ""))
            sys.exit()
    token = d.get("next_token") if isinstance(d, dict) else None
    if not token or not rows:
        break
')"
  SSH_USER="root"
  SSH_HOST="${line%% *}"
  SSH_PORT="${line##* }"
  if [ -z "$SSH_HOST" ] || [ -z "$SSH_PORT" ]; then
    echo "Could not resolve SSH endpoint for instance $INSTANCE via 'vastai show instances-v1'." >&2
    echo "Is it running?  vastai show instances-v1" >&2
    return 1
  fi
}
