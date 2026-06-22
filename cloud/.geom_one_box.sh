#!/usr/bin/env bash
# Drive ONE geom_multi box (tag $1) through the same pipeline run_one uses, then destroy it.
set -uo pipefail
HERE="$PWD/cloud"
FLEET_DIR="$HERE/.fleet_geom_multi"
tag="$1"
export STUDIES="geometry" N_THINKING=0 SUBSAMPLE=0.07 BATCH_SIZE=16
iid="$(cat "$FLEET_DIR/$tag.id")"
IFS=$'\t' read -r model sidx scount ngpu < "$FLEET_DIR/$tag.job"
log="$FLEET_DIR/$tag.log"
{
  echo "[$tag] RELAUNCH instance=$iid model=$model shard=$sidx/$scount"
  # wait_running
  for i in $(seq 1 80); do
    st="$(INSTANCE_ID="$iid" python3 -c '
import json,os,subprocess,sys
iid=int(os.environ["INSTANCE_ID"]); token=None
for _ in range(40):
    cmd=["vastai","show","instances-v1","--raw"]
    if token: cmd+=["--next-token",token]
    try: d=json.loads(subprocess.run(cmd,capture_output=True,text=True).stdout)
    except Exception: print("missing"); sys.exit()
    rows=d if isinstance(d,list) else d.get("instances",d.get("results",[]))
    for r in rows:
        if r.get("id")==iid: print(r.get("actual_status") or r.get("cur_state") or "unknown"); sys.exit()
    token=d.get("next_token") if isinstance(d,dict) else None
    if not token or not rows: break
print("missing")')"
    [ "$st" = "running" ] && break
    sleep 15
  done
  # wait_ssh
  ok=0
  for i in $(seq 1 60); do
    if INSTANCE="$iid" bash "$HERE/at_vast.sh" "true" >/dev/null 2>&1; then ok=1; break; fi
    sleep 10
  done
  [ "$ok" = 1 ] || { echo "[$tag] SSH never came up; destroying."; INSTANCE="$iid" bash "$HERE/vast_destroy.sh" --yes-i-am-really-sure; exit 1; }
  # sync_up + at_setup with retries
  setup_ok=0
  for a in 1 2 3 4; do
    if INSTANCE="$iid" bash "$HERE/sync_up.sh" && INSTANCE="$iid" bash "$HERE/at_vast.sh" "bash cloud/at_setup.sh"; then setup_ok=1; break; fi
    echo "[$tag] setup attempt $a failed; retry"; sleep 15
  done
  [ "$setup_ok" = 1 ] || { echo "[$tag] setup failed; destroying."; INSTANCE="$iid" bash "$HERE/vast_destroy.sh" --yes-i-am-really-sure; exit 1; }
  # run geometry
  INSTANCE="$iid" bash "$HERE/at_vast.sh" \
    "HF_TOKEN='$HF_TOKEN' MODEL='$model' SHARD_INDEX=$sidx SHARD_COUNT=$scount STUDIES='geometry' BATCH_SIZE=16 N_THINKING=0 SUBSAMPLE='0.07' MAX_NEW_TOKENS='' bash cloud/fleet_model_run.sh"
  # sync_back into its own quarantine
  INSTANCE="$iid" SYNC_SUBDIR="box-$tag" STUDIES="geometry" bash "$HERE/sync_back.sh"
  # self-destroy
  INSTANCE="$iid" bash "$HERE/vast_destroy.sh" --yes-i-am-really-sure
  echo "[$tag] DONE + destroyed"
} >"$log" 2>&1
