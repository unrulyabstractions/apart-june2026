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

# Shared SSH options for EVERY connection to a Vast box (at_vast / sync_up / sync_back).
# Vast reuses the SAME proxy endpoints ([sshN.vast.ai]:PORT) across many different
# boxes, each with a DIFFERENT host key. A persistent known_hosts therefore guarantees
# "REMOTE HOST IDENTIFICATION HAS CHANGED" rejections that hang forever as
# "waiting for sshd" (gotcha #1) — a fleet-killer at scale, and previously fixed by
# hand-purging known_hosts. Discarding host keys (UserKnownHostsFile=/dev/null +
# StrictHostKeyChecking=no) removes the failure mode ENTIRELY, so 100+ boxes cycling
# through reused proxy hosts never collide. Acceptable for throwaway GPU boxes.
# ConnectTimeout fails a not-yet-up sshd fast; ServerAlive* tears down a wedged session.
SSH_EPHEMERAL_OPTS="-F /dev/null -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=8"

_resolve_ssh_target() {
  # Short-circuit: if the caller already exported SSH_HOST + SSH_PORT (e.g. the job
  # orchestrator resolved them ONCE from the shared cache and passed them down), reuse
  # them instead of doing another live paginated API walk. This is what keeps the
  # SETUP path (sync_up / at_vast) cheap at fleet scale — no per-call API hit. A box's
  # endpoint is stable for its lifetime, so reusing it within one run is safe.
  if [ -n "${SSH_HOST:-}" ] && [ -n "${SSH_PORT:-}" ]; then
    SSH_USER="${SSH_USER:-root}"; return 0
  fi
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
