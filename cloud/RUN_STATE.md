# LIVE CLOUD RUN STATE  (update every monitoring cycle — DO NOT re-derive)

Last updated: 2026-06-22 ~01:02. Deadline (2026-06-21 AoE) PASSED; finishing for completeness.

## Active runs (poller log → box → output)

| run         | MODEL                         | log                  | sync/out target                              | purpose |
|-------------|-------------------------------|----------------------|----------------------------------------------|---------|
| fork-single | Qwen/Qwen3-14B (single-box)   | /tmp/fork14.log      | sync/fork32/ → out/sesgo/forking/Qwen3-14B/  | FALLBACK complete forking figure (box 42062055) |
| fork-fleet2 | Qwen/Qwen3-14B (5 shards)      | /tmp/fleet2.log + /tmp/forkshard_{0..4}.log | out/sesgo/forking/Qwen3-14B/ | CANONICAL ≥5-parallel; reuses sync/forkbase base_path (332 pos) |
| sel-qwen32b | Qwen/Qwen3-32B selection       | /tmp/sel_qwen32b.log | sync/sel_qwen32b/ → out/sesgo/selection/Qwen3-32B/ | the "empty selection-32B" backfill |
| sel-llama1b | meta-llama/Llama-3.2-1B sel    | /tmp/sel_llama1b.log | sync/sel_llama1b/ → out/sesgo/selection/Llama-3.2-1B-Instruct/ | scaffold-across-scale |
| sel-llama70b| meta-llama/Llama-3.1-70B sel(2GPU)| /tmp/sel_llama70b.log| sync/sel_llama70b/ → out/sesgo/selection/Llama-3.1-70B-Instruct/ | scaffold-across-scale |

DONE+promoted: div-0.6B, div-32B (out/sesgo/divergence/{Qwen3-0.6B,Qwen3-32B}, vocab-entropy verified).
HELD (document or later): geometry backfill Llama-3.2-1B/3B (needs activation-extraction runner; no run_one_geometry_box.sh).

## Remaining finish-line critical path
1. forking 14B lands (fleet2 canonical OR single-box fallback) → promote → analyze_forking_dynamics + plot_forking_dynamics → out/sesgo/forking/Qwen3-14B/forking_dynamics.png. Prefer the COMPLETE one (single-box full 332 pos, or fleet if all 5 shards landed; merge pads gaps).
2. selection backfills land → promote sync/sel_*/ → out/sesgo/selection/<model>/ → verify non-degenerate + image-verify plots.
3. FINAL PAPER PASS: dynamics.tex → Qwen3-14B (currently 0.6B); confirm bias captions match segment figure; rebuild PDF; image-verify all 15 \plotfig resolve + minimal.
4. out/ final audit + HF re-upload EXCLUDING projections.json (3.4G regenerable UI intermediate) + **/*.log + activations.
5. Main-body stale refs are the USER's to fix: limitations.tex:13 ("43 scored items"), limitations.tex:8/23/61/77 + methods.tex:7 ("only Qwen3-0.6B" coverage).

## HARD-WON GOTCHAS (never re-learn)
1. **Stuck "waiting for sshd" = STALE PROXY KEY** (Vast reuses [sshN.vast.ai]:PORT; accept-new refuses a CHANGED key). **NOW FIXED AUTOMATICALLY:** all SSH/rsync use `SSH_EPHEMERAL_OPTS` (in `_ssh_target.sh`) = `UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no`, so host keys are discarded and a reused proxy endpoint never collides. The old manual purge is no longer needed: `for hp in $(grep -oE "\[ssh[0-9]+\.vast\.ai\]:[0-9]+" ~/.ssh/known_hosts|sort -u); do ssh-keygen -R "$hp"; done`.
2. **NEVER broad `pkill` a poller** — each has `trap destroy EXIT`; SIGTERM destroys its box. (Lost div-32B box 42061044 mid-collect this way.) To tear down a fleet ON PURPOSE, pkill the shard pollers (traps destroy boxes) + explicit `vastai destroy`.
3. RTX 4090 boxes flaky sshd → `MIN_RELIABILITY=0.99`. H100 div/sel needs `MAX_PRICE>=3.0`.
4. macOS has NO `timeout`; ssh needs `-F /dev/null` (bad usekeychain in ~/.ssh/config). Box list: `vastai show instances-v1 --raw`. Destroy: `printf 'y\n' | vastai destroy instance <id>`.
5. Shell scripts (run_one_*.sh) are IMMUNE to Anthropic 529; agents die on it.
6. Fleet orchestrator now SKIPS base box if sync/forkbase/.../base_path.json exists (reuse cached base CoT).
7. Predicted answer field is `non_thinking.greedy_label` (NOT predicted_non_thinking, which is None in raw JSON). question_polarity ∈ {neg,nonneg}.
8. graphicspath registers only Qwen3-0.6B divergence dir → pass FULL relative path as BOTH \plotsub args for 32B figures.
9. **`;`-EXPORT BUG (cost 4 wasted boxes):** in run_one_selection/baseline_box.sh the RUN_ENV was `export <HF vars>; MODEL='...' STUDIES='...'` — the `;` made MODEL/STUDIES/N_THINKING/etc. NON-exported shell vars, so the child `bash fleet_model_run.sh` fell back to its defaults (MODEL=Qwen3-0.6B, STUDIES="baseline divergence"). Every non-0.6B run silently collected the WRONG model into the WRONG dir → verify on the intended dir = empty → FATAL → destroy → data lost. FIX: fold the vars into ONE `export` (remove the `;`). Forking scripts are immune (they pass `--model $MODEL` directly). When launching a NEW cloud collect, VERIFY the box runs the intended model (ssh + nvidia-smi/n_layers) before trusting it.

## Monitor one-liner
for f in fork14 fleet2 sel_qwen32b sel_llama1b sel_llama70b; do echo "== $f =="; tail -2 /tmp/$f.log; done
vastai show instances-v1 --raw | python3 -c 'import sys,json;[print(i["id"],i.get("actual_status"),i.get("gpu_util")) for i in json.load(sys.stdin)["instances"]]'
