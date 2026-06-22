#!/usr/bin/env bash
#
# job_registry.sh — every unit of cloud work is a JOB with a stable identifier and a
# manifest that is its single source of truth. Sourced (not executed).
#
# Layout (jobs/ is gitignored — the quarantine; the cloud NEVER writes to out/):
#   jobs/<job-id>/
#     manifest.json   id, model, study, shard, instance, status, verdict, reason, ts
#     box.log         the box's run log
#     VERIFY.txt      the verifier's evidence
#     output/         synced payload, mirrors out/ exactly (promoted from here)
#
# CORE INVARIANT — NOTHING SILENTLY FAILS. Every job MUST end in a TERMINAL state:
#     promoted | verified | broken | failed(reason)
# Anything still in created/launched/running/synced at reconciliation is a LOUD
# failure ("stalled — investigate"). jr_fail records a reason AND prints to stderr.
#
# Functions:
#   jr_new <study> <model> <sidx> <scount> [verify_subpath]  -> echoes job-id
#   jr_dir <job-id> ; jr_get <job-id> <key> ; jr_set <job-id> <key> <val>
#   jr_fail <job-id> <reason>   (terminal failure, loud)
#   jr_list                     (all job-ids)
#   jr_assert_arg <name> <val>  (abort LOUDLY on an empty required arg)

_JR_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$_JR_HERE/.." && pwd)}"
JOBS_DIR="${JOBS_DIR:-$REPO_ROOT/jobs}"

# Terminal states a job may legitimately END in. Used by reconciliation to find
# jobs that died without recording anything (the silent-failure case).
JR_TERMINAL_STATES="promoted verified broken failed"

# Abort LOUDLY if a required value is empty — a blank model/study would otherwise
# create a mislabelled job and silently collect the WRONG thing (gotcha #9 class).
jr_assert_arg() {
  if [ -z "${2:-}" ]; then
    echo "FATAL[job_registry]: required arg '$1' is empty — refusing to create a mislabelled job." >&2
    return 2
  fi
}

# Lowercase, strip org prefix, keep filesystem-safe chars: Qwen/Qwen3-14B -> qwen3-14b
_jr_slug() { printf '%s' "${1##*/}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9.-' '-' | sed 's/-\{2,\}/-/g; s/^-//; s/-$//'; }

# Create a new job and echo its id. Fails (rc!=0, prints nothing on stdout) if a
# required field is missing — so a caller can never spawn an unlabelled job.
jr_new() {
  local study="$1" model="$2" sidx="$3" scount="$4" vsub="${5:-}"
  jr_assert_arg study "$study" || return 2
  jr_assert_arg model "$model" || return 2
  jr_assert_arg shard_index "$sidx" || return 2
  jr_assert_arg shard_count "$scount" || return 2
  local now jid
  now="$(date +%s)"
  jid="$(_jr_slug "$model")__$(_jr_slug "$study")__s${sidx}of${scount}__${now}${RANDOM}"
  local dir="$JOBS_DIR/$jid"
  mkdir -p "$dir/output" || { echo "FATAL[job_registry]: cannot mkdir $dir" >&2; return 1; }
  # Default the verify subpath to the canonical out/ slice for this (study,model).
  [ -z "$vsub" ] && vsub="sesgo/$study/$(printf '%s' "${model##*/}")"
  JID="$jid" STUDY="$study" MODEL="$model" SIDX="$sidx" SCOUNT="$scount" \
    VSUB="$vsub" NOW="$now" MANIFEST="$dir/manifest.json" python3 -c '
import json, os
m = {
  "job_id": os.environ["JID"], "study": os.environ["STUDY"], "model": os.environ["MODEL"],
  "shard_index": int(os.environ["SIDX"]), "shard_count": int(os.environ["SCOUNT"]),
  "instance_id": "", "status": "created", "verdict": "", "reason": "",
  "verify_subpath": os.environ["VSUB"],
  "created_at": int(os.environ["NOW"]), "updated_at": int(os.environ["NOW"]),
}
json.dump(m, open(os.environ["MANIFEST"], "w"), indent=2)
' || { echo "FATAL[job_registry]: cannot write manifest for $jid" >&2; return 1; }
  printf '%s\n' "$jid"
}

jr_dir() { printf '%s\n' "$JOBS_DIR/$1"; }

jr_get() {
  local jid="$1" key="$2" mf="$JOBS_DIR/$1/manifest.json"
  [ -f "$mf" ] || { echo "FATAL[job_registry]: no manifest for job '$jid'" >&2; return 1; }
  KEY="$key" MF="$mf" python3 -c '
import json, os
print(json.load(open(os.environ["MF"])).get(os.environ["KEY"], ""))'
}

# Update one manifest key (bumps updated_at). Validates the key against the schema so
# a typo cannot silently write a field nobody reads.
jr_set() {
  local jid="$1" key="$2" val="$3" mf="$JOBS_DIR/$1/manifest.json"
  [ -f "$mf" ] || { echo "FATAL[job_registry]: no manifest for job '$jid'" >&2; return 1; }
  KEY="$key" VAL="$val" MF="$mf" python3 -c '
import json, os, sys, time
allowed = {"instance_id","status","verdict","reason"}
k = os.environ["KEY"]
if k not in allowed:
    sys.stderr.write(f"FATAL[job_registry]: refusing to set unknown manifest key {k!r}\n"); sys.exit(1)
m = json.load(open(os.environ["MF"]))
m[k] = os.environ["VAL"]; m["updated_at"] = int(time.time())
json.dump(m, open(os.environ["MF"], "w"), indent=2)
' || return 1
}

# Terminal failure: record status=failed + a reason, and ALWAYS announce it loudly so
# it can never pass unnoticed. This is the anti-silent-failure primitive.
jr_fail() {
  local jid="$1" reason="$2"
  jr_set "$jid" status failed || true
  jr_set "$jid" reason "$reason" || true
  echo "JOB-FAILED[$jid]: $reason" >&2
}

jr_list() {
  [ -d "$JOBS_DIR" ] || return 0
  local d
  for d in "$JOBS_DIR"/*/; do
    [ -f "$d/manifest.json" ] && basename "$d"
  done
}
