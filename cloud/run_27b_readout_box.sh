#!/usr/bin/env bash
#
# run_one_forking_box_32b.sh — full lifecycle for ONE Qwen3-32B forking box, fast.
#
# Stands up EXACTLY ONE large-VRAM Vast.ai GPU box (H100 80GB), runs the FULL-CoT
# forking-paths collection for Qwen/Qwen3-32B end to end ON the box (select item ->
# collect {O_t} -> analyze -> plot), syncs results back into a DISJOINT local
# quarantine (sync/fork32/...), and DESTROYS the box. Destroy is wired to an EXIT
# trap so the box is torn down on success, failure, OR interruption — it can never
# be left billing.
#
# WHY THIS IS A SEPARATE DRIVER (vs run_one_forking_box.sh, the 0.6B one):
#   - Qwen3-32B needs an 80 GB GPU (H100), not a 4090; and ~64 GB of fp16 weights
#     means a bigger disk for the HF download cache.
#   - There is no pre-selected forking item for 32B (item selection is MODEL-
#     specific: which token flips the answer depends on the model), so we run
#     select_forking_item.py ON the box for Qwen3-32B instead of pushing a fixed
#     selected_item.json.
#   - It uses its OWN tracking file (.fork32.iid) and OWN quarantine subtree
#     (sync/fork32/) so it never resolves/destroys the separate 0.6B forking box
#     or the divergence boxes that may be running concurrently.
#
# BACKEND: HuggingFace batched. capture_forking_trajectory flattens EVERY
# (position, alternate, sample) branch into ONE continue_from_text_batch call, so
# all base-path positions decode together (not one at a time). The HF backend
# decodes that flat batch in GPU-saturating micro-batches (HF_GEN_MICRO_BATCH,
# tuned below for the 32B on 80 GB). vLLM is NOT used: the forking plan-build needs
# full-vocab logits + a greedy token trajectory, which the vLLM backend does not
# expose, so the pipeline is HF-only by design.
#
# Optional env overrides:
#   GPU_NAME (H100_SXM) NUM_GPUS (1) MAX_PRICE (3.0) DISK (160) MIN_RELIABILITY (0.985)
#   N_PRIOR (50) N_SAMPLES (50) MAX_NEW_TOKENS (768) BASE_MAX_NEW_TOKENS (768)
#   TEMPERATURE (1.0) HF_GEN_MICRO_BATCH (24)
#   SELECT_CATEGORIES (gender) SELECT_LANGUAGES (es) SELECT_N_PILOT (12) SELECT_MAX_ITEMS (10)
#   SELECT_SCAFFOLD ("")    debiasing scaffold_id to prepend (e.g. interpretive_direction)
#   SELECT_FORCE_QID ("")   force this question_id (skips the GPU pilot entirely)
#   RUN_TAG ("")            output-subdir suffix so scaffold/baseline conditions never collide

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_ROOT="/root/apart"

MODEL="${MODEL:-Qwen/Qwen3-32B}"
BARE_MODEL="${MODEL##*/}"
GPU_NAME="${GPU_NAME:-H100_SXM}"
NUM_GPUS="${NUM_GPUS:-1}"
MAX_PRICE="${MAX_PRICE:-3.0}"
DISK="${DISK:-160}"
MIN_RELIABILITY="${MIN_RELIABILITY:-0.985}"
# Min GPU VRAM (GB) for the offer filter. Default 79 (the 32B needs an 80 GB card);
# override DOWN for small models on a 24 GB RTX_4090 (e.g. GPU_RAM=20).
GPU_RAM="${GPU_RAM:-79}"

# FULL-CoT run knobs (task spec): branch EVERY base-path position (max-positions 0),
# 768-token base decode + 768-token continuations, n-prior 50, n-samples 50, T=1.0.
N_PRIOR="${N_PRIOR:-50}"
N_SAMPLES="${N_SAMPLES:-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
BASE_MAX_NEW_TOKENS="${BASE_MAX_NEW_TOKENS:-768}"
TEMPERATURE="${TEMPERATURE:-1.0}"
# Continuation decode micro-batch: 32B fp16 weights ~64 GB leave ~16 GB for the KV
# cache, so cap the batch at 24 prompts × 768 tokens (GPU-saturating, OOM-safe).
# (Left at 24 — unlike the <=14B paths, 32B has no headroom to raise this.)
HF_GEN_MICRO_BATCH="${HF_GEN_MICRO_BATCH:-24}"
# PILOT cost knob, default OFF here: this single-box run feeds the per-token O_t
# dynamics figure, which needs every position, so stride defaults to 1. Set
# POSITION_STRIDE>1 only for a cheap coverage pilot that doesn't need the figure.
POSITION_STRIDE="${POSITION_STRIDE:-1}"

# Item-selection knobs (runs ON the box for this model).
# SELECT_SESGO_DIR: the prompt-source root. The select script's own default
# (datasets/datasets/SESGO) is a doubled path that does NOT exist; sync_up.sh
# pushes the .xlsx sources to datasets/SESGO/prompts/, so we point select there.
SELECT_SESGO_DIR="${SELECT_SESGO_DIR:-datasets/SESGO}"
SELECT_CATEGORIES="${SELECT_CATEGORIES:-gender}"
SELECT_LANGUAGES="${SELECT_LANGUAGES:-es}"
SELECT_N_PILOT="${SELECT_N_PILOT:-12}"
SELECT_MAX_ITEMS="${SELECT_MAX_ITEMS:-10}"

# Scaffold-vs-baseline forking-comparison knobs. SELECT_SCAFFOLD prepends a
# debiasing preamble; SELECT_FORCE_QID forces the SAME item across conditions
# (skipping the pilot); RUN_TAG suffixes the out subdir so the two conditions
# write to DISJOINT paths and sync back separately. All empty => identical to
# the original full-pilot single-box behaviour.
SELECT_SCAFFOLD="${SELECT_SCAFFOLD:-}"
SELECT_FORCE_QID="${SELECT_FORCE_QID:-}"
RUN_TAG="${RUN_TAG:-}"
# THINKING=1 decodes the base path + prior in reasoning mode (enable_thinking for
# Qwen3.5-family). Required when forking a *-thinking model so the captured base
# path is the chain of thought, not the suppressed-thinking answer.
THINKING="${THINKING:-}"
# Build the optional select-only flags (only appended when their env is set).
SCAFFOLD_FLAG=""; [ -n "$SELECT_SCAFFOLD" ] && SCAFFOLD_FLAG="--scaffold $SELECT_SCAFFOLD"
FORCE_QID_FLAG=""; [ -n "$SELECT_FORCE_QID" ] && FORCE_QID_FLAG="--force-question-id $SELECT_FORCE_QID"
THINKING_FLAG=""; [ -n "$THINKING" ] && THINKING_FLAG="--thinking"
# RUN_TAG is ALWAYS passed (empty suffix == current path) to every stage. The
# --run-tag=VALUE form (not a space) is REQUIRED because a leading '-' suffix (e.g.
# "-interpretive_direction") would otherwise be mis-parsed by argparse as a flag.
RUN_TAG_FLAG="--run-tag=$RUN_TAG"

IID_FILE="${IID_FILE:-$HERE/.fork32.iid}"  # override per-run so parallel boxes never share a tracking file
INSTANCE=""

log() { echo "[fork32 $(date +%H:%M:%S)] $*"; }

# ── DESTROY-ON-EXIT: tear the box down no matter how we leave this script ──
destroy_box() {
  local rc=$?
  if [ -n "$INSTANCE" ]; then
    log "DESTROY instance $INSTANCE (exit rc=$rc)"
    INSTANCE="$INSTANCE" bash "$HERE/vast_destroy.sh" --yes-i-am-really-sure 2>&1 | sed "s/^/[fork32 destroy] /"
    sleep 5
    local still
    still="$(INSTANCE_ID="$INSTANCE" python3 -c '
import json, os, subprocess, sys
iid=int(os.environ["INSTANCE_ID"]); token=None
for _ in range(40):
    cmd=["vastai","show","instances-v1","--raw"]
    if token: cmd+=["--next-token",token]
    try: d=json.loads(subprocess.run(cmd,capture_output=True,text=True).stdout)
    except Exception: break
    rows=d if isinstance(d,list) else d.get("instances",d.get("results",[]))
    if any(isinstance(r,dict) and r.get("id")==iid for r in rows): print("yes"); sys.exit()
    token=d.get("next_token") if isinstance(d,dict) else None
    if not token or not rows: break
print("no")' 2>/dev/null)"
    if [ "$still" = "yes" ]; then
      log "WARN: instance $INSTANCE still listed; retrying destroy"
      printf 'y\n' | vastai destroy instance "$INSTANCE" 2>&1 | sed "s/^/[fork32 destroy2] /"
    else
      log "confirmed instance $INSTANCE no longer listed"
    fi
    rm -f "$IID_FILE"
  fi
  return $rc
}
trap destroy_box EXIT

run_on_box() {  # run a command on this box via direct ssh (fresh endpoint each call)
  local cmd="$1"
  . "$HERE/_ssh_target.sh"
  _resolve_ssh_target || return 1
  ssh -F /dev/null -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 \
      -o ServerAliveInterval=15 -o ServerAliveCountMax=240 \
      -i "$SSH_KEY" -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" \
      "cd $REMOTE_ROOT 2>/dev/null || true; $cmd"
}

# run_detached_and_wait — run a LONG command DETACHED on the box, then poll for
# completion over RECONNECTING SSH so a dropped session (the previous attempt died
# exactly here: "Connection closed by remote host" mid-decode) never loses the run.
#
# The command is launched under setsid+nohup with its stdout/stderr to a remote
# log; on exit its status is written to a marker file. We poll the marker every
# POLL secs (fresh SSH each poll, tolerant of transient drops) and stream new log
# lines, returning the remote command's real exit code. A dead box (SSH refuses
# for MAX_DROP consecutive polls) aborts so the EXIT trap can destroy + stop bill.
#   $1 = step tag (also the remote log/marker basename)   $2 = command string
run_detached_and_wait() {
  local tag="$1" cmd="$2"
  local rlog="/root/apart/.${tag}.out" rdone="/root/apart/.${tag}.done"
  run_on_box "rm -f $rlog $rdone; \
    setsid bash -c 'cd $REMOTE_ROOT; { $cmd; } > $rlog 2>&1; echo \$? > $rdone' \
    </dev/null >/dev/null 2>&1 &" || { log "FATAL: could not launch $tag detached"; return 99; }
  log "launched '$tag' DETACHED on box; polling $rdone (log: $rlog)"
  local poll=20 drops=0 max_drops=45 seen=0
  while true; do
    # ONE ssh call returns BOTH the done marker and the log line-count, so they can
    # never desync (the old split cat/wc calls could livelock: a flaky link returns
    # empty for `cat .done` while `wc -l` succeeds, resetting the drop counter and
    # never detecting completion -> infinite hang). Sentinels bracket the payload so a
    # partial/garbled read is treated as a transient drop, never as "done".
    local probe done_val nlines
    probe="$(run_on_box "printf 'P:%s:%s:Q' \"\$(cat $rdone 2>/dev/null)\" \"\$(wc -l < $rlog 2>/dev/null | tr -dc 0-9)\"" 2>/dev/null)"
    if [[ "$probe" != P:*:Q ]]; then           # no/garbled response == one transient drop
      drops=$((drops+1))
      log "'$tag' poll: no usable response from box ($drops/$max_drops)"
      [ "$drops" -ge "$max_drops" ] && { log "FATAL: box unreachable for $max_drops polls during '$tag' -- aborting"; return 98; }
      sleep "$poll"; continue
    fi
    drops=0
    probe="${probe#P:}"; done_val="${probe%%:*}"; nlines="${probe#*:}"; nlines="${nlines%:Q}"
    if [ -n "$done_val" ]; then                 # marker present => command finished
      run_on_box "tail -n 25 $rlog 2>/dev/null" 2>/dev/null | sed "s/^/[fork32 $tag] /"
      log "'$tag' finished with exit code $(printf '%s' "$done_val" | tr -dc '0-9' || echo '?')"
      return "$(printf '%s' "$done_val" | tr -dc '0-9')"
    fi
    if [ -n "$nlines" ] && [ "$nlines" -gt "$seen" ] 2>/dev/null; then
      run_on_box "sed -n '$((seen+1)),\$p' $rlog 2>/dev/null" 2>/dev/null | sed "s/^/[fork32 $tag] /"
      seen="$nlines"
    fi
    sleep "$poll"
  done
}

# ── 1. SEARCH for a matching verified offer (reliability floor enforced) ──
INET_DOWN="${INET_DOWN:-0}"  # min download Mbps — set high so the ~54GB model pulls fast (slow links stall on HF)
QUERY="gpu_name=${GPU_NAME} num_gpus=${NUM_GPUS} verified=true rentable=true direct_port_count>=1 gpu_ram>=${GPU_RAM} disk_space>=${DISK} inet_down>=${INET_DOWN} dph_total<=${MAX_PRICE}"
log "search offers: $QUERY (rel2 floor $MIN_RELIABILITY enforced post-hoc)"
OFFERS_JSON="$(vastai search offers "$QUERY" -o 'dph_total+' --raw 2>/dev/null)"
OFFER_ID="$(printf '%s' "$OFFERS_JSON" | MIN_REL="$MIN_RELIABILITY" python3 -c '
import sys, json, os
floor=float(os.environ["MIN_REL"])
o=json.load(sys.stdin)
flt=[r for r in o if r.get("reliability2",0)>=floor]
print(flt[0]["id"] if flt else "")')"
if [ -z "$OFFER_ID" ]; then
  log "FATAL: no matching offers for $GPU_NAME (reliability2>=$MIN_RELIABILITY, <=\$$MAX_PRICE/hr)"
  exit 2
fi
printf '%s' "$OFFERS_JSON" | OFFER_ID="$OFFER_ID" python3 -c '
import sys, json, os
oid=int(os.environ["OFFER_ID"])
o=next(r for r in json.load(sys.stdin) if r["id"]==oid)
print("[offer] id=%s %sx %s vram=%sMB $%.3f/hr reliability2=%.4f disk=%sGB net=%sMbps" % (
    o.get("id"), o.get("num_gpus"), o.get("gpu_name"),
    o.get("gpu_total_ram", o.get("gpu_ram",0)), o.get("dph_total",0.0),
    o.get("reliability2",0.0), o.get("disk_space"), o.get("inet_down")))'

# ── 2. CREATE the instance (PyTorch image, SSH, direct) ──
PORTAL_ENV='-p 1111:1111 -e OPEN_BUTTON_PORT="1111" -e OPEN_BUTTON_TOKEN="1" -e JUPYTER_DIR="/" -e DATA_DIRECTORY="/workspace/"'
CREATE_JSON="$(vastai create instance "$OFFER_ID" \
  --image 'vastai/pytorch:@vastai-automatic-tag' \
  --env "$PORTAL_ENV" --onstart-cmd 'entrypoint.sh' \
  --disk "$DISK" --ssh --direct --raw)"
INSTANCE="$(printf '%s' "$CREATE_JSON" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("new_contract",""))')"
if [ -z "$INSTANCE" ]; then
  log "FATAL: create failed: $CREATE_JSON"
  exit 3
fi
echo "$INSTANCE" > "$IID_FILE"
log "created instance $INSTANCE; waiting for 'running'"

# ── 3. POLL until running (walk paginated instances-v1) ──
instance_status() {
  INSTANCE_ID="$INSTANCE" python3 -c '
import json, os, subprocess, sys
iid=int(os.environ["INSTANCE_ID"]); token=None
for _ in range(40):
    cmd=["vastai","show","instances-v1","--raw"]
    if token: cmd+=["--next-token",token]
    try: d=json.loads(subprocess.run(cmd,capture_output=True,text=True).stdout)
    except Exception: break
    rows=d if isinstance(d,list) else d.get("instances",d.get("results",[]))
    for r in rows:
        if isinstance(r,dict) and r.get("id")==iid:
            print(r.get("actual_status") or r.get("cur_state") or "unknown"); sys.exit()
    token=d.get("next_token") if isinstance(d,dict) else None
    if not token or not rows: break
print("missing")'
}
for i in $(seq 1 120); do
  ST="$(instance_status)"
  log "  [$i] status: $ST"
  [ "$ST" = "running" ] && break
  sleep 15
done
[ "$ST" = "running" ] || { log "FATAL: instance never reached running"; exit 4; }

# ── 4. Wait for sshd to actually accept connections ──
log "waiting for sshd"
for i in $(seq 1 50); do
  if run_on_box "echo ssh_ok" 2>/dev/null | grep -q ssh_ok; then log "sshd up"; break; fi
  sleep 10
done

# ── 5. SYNC code UP (reuse sync_up.sh) ──
log "sync_up (code + SESGO prompts)"
INSTANCE="$INSTANCE" bash "$HERE/sync_up.sh" 2>&1 | sed "s/^/[fork32 up] /"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "FATAL: sync_up failed"; exit 5; }

# ── 6. uv sync + cu124 torch pin + device check (reuse at_setup.sh) ──
log "at_setup (uv sync, cu124 pin, device check)"
# DETACHED + polled: at_setup downloads ~500 MB of torch wheels over a single long SSH
# session; a mid-download proxy drop ("Can't assign requested address" / broken pipe)
# would otherwise crash it. Running detached makes it survive SSH drops like collect.
run_detached_and_wait setup "bash cloud/at_setup.sh"
[ $? -eq 0 ] || { log "FATAL: at_setup failed"; exit 6; }


# ── 7. Push the candidate dataset (sync_up excludes /data/, so push it explicitly) ──
log "push candidate dataset to box"
. "$HERE/_ssh_target.sh"; _resolve_ssh_target || { log "FATAL: ssh endpoint"; exit 5; }
RSYNC_E="ssh $SSH_EPHEMERAL_OPTS -i $SSH_KEY -p $SSH_PORT"
ssh $SSH_EPHEMERAL_OPTS -i "$SSH_KEY" -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" "mkdir -p $REMOTE_ROOT/data" 2>/dev/null
rsync -ah -e "$RSYNC_E" "$REPO_ROOT/data/shared_base_candidates_dataset.json" \
  "$SSH_USER@$SSH_HOST:$REMOTE_ROOT/data/" || { log "FATAL: dataset push failed"; exit 5; }

# ── 8. 27B THINKING greedy readout over the candidates (detached + polled) ──
log "27B thinking readout over candidate items ($MODEL)"
run_detached_and_wait readout \
  ".venv/bin/python -m experiment.stability.run_greedy_readout --model $MODEL --mode thinking \
     --backend huggingface --dataset data/shared_base_candidates_dataset.json --study sharedcand \
     --out-dir out --max-reasoning 1024"
[ $? -eq 0 ] || { log "FATAL: readout failed"; exit 8; }

# ── 9. Verify non-empty + sync back ──
OUT="out/sharedcand/${BARE_MODEL}-thinking/response_samples.json"
N="$(run_on_box ".venv/bin/python -c \"import json;print(len(json.load(open('$OUT'))['samples']))\" 2>/dev/null || echo 0")"
N="$(printf '%s' "$N" | tr -dc '0-9')"
log "remote $OUT has ${N:-0} samples"
[ "${N:-0}" -ge 1 ] || { log "FATAL: empty readout (no sync, will destroy)"; exit 9; }

log "sync_back -> sync/27b/sharedcand/"
SYNC_SUBDIR="27b" STUDIES="sharedcand" INSTANCE="$INSTANCE" bash "$HERE/sync_back.sh" 2>&1 | sed "s/^/[27b back] /"
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "FATAL: sync_back failed (box will still be destroyed)"; exit 11; }

log "SUCCESS: 27B thinking readout over ${N} candidates; quarantined under sync/27b/"
exit 0
