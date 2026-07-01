# Lessons

## Cloud / shell
- The local shell is initialized from the user's zsh profile, where `status` is a
  READ-ONLY reserved variable. NEVER name a shell variable `status` (also avoid
  other zsh specials). It silently breaks background poll loops with
  "read-only variable: status". Use `poll_out`/`run_state` etc.
- `vast_launch.sh`'s inline polling parser assumes `show instances-v1 --raw` is a
  top-level list, but this vastai version returns `{"instances":[...]}`. The loop
  crashes AFTER the instance is created but BEFORE writing `.vast_instance_id`.
  Recovery: parse the create output's `new_contract` id, write it to
  `cloud/.vast_instance_id` manually. `_ssh_target.sh` already handles the dict
  shape, so all other cloud scripts work fine.

## Forking-paths at large model scale (32B)
- The HF backend `generate_batch` did ONE `model.generate` over the ENTIRE flat
  prompt list (no chunking). For a 32B model the forking branch set
  (positions x alternates x n_samples) is tens of thousands of sequences and
  OOMs an 80GB GPU on the KV cache. Fix: micro-batch chunking inside
  `generate_batch` (HF_GEN_MICRO_BATCH env, default 64), concatenated in order =
  identical output, bounded memory. This is what makes "saturate the GPU via
  batched forks" actually run at 32B.
- HF backend loads via `.from_pretrained(dtype).to(cuda)` with NO device_map;
  on CUDA dtype defaults to fp16. 32B fp16 ~= 66 GB on one H100 80GB, leaving
  ~15 GB for KV cache -> micro-batch must stay bounded.

## Teacher-forced (non-thinking) batch OOM on small GPUs
- `compute_trajectories_batch` runs ONE forward over the whole batch and the
  3-way `choose3_batch` builds 3x the logical batch of FULL-VOCAB logits at once.
  On a 24 GB RTX 4090, BATCH_SIZE=64 over long scaffolded SESGO prompts tried to
  alloc 12 GB and OOM'd — every box died after the first checkpoint (64 samples),
  yet still self-destructed, leaving truncated shards. `HF_GEN_MICRO_BATCH` only
  chunks GENERATION, not this teacher-forced forward. Fix: added
  `HF_FORWARD_MICRO_BATCH` to `compute_trajectories_batch` (chunk + concat,
  unset == one pass = unchanged) AND run the 4090 fleet at BATCH_SIZE=16 +
  HF_FORWARD_MICRO_BATCH=8. Lesson: for the non-thinking/teacher-forced path on a
  4090, keep BATCH_SIZE <= 16; the micro-batch is defense-in-depth.
- Truncated shards look "successful" (box reports DONE+destroyed). ALWAYS verify
  a shard's sample COUNT and sample_idx RANGE, not just that a file landed.

## Vast.ai fleet reliability + cost (full-data baseline run)
- A Vast SSH-PROXY-WIDE outage (kex_exchange_identification reset across many
  ssh*.vast.ai hosts + proxy IPs at once) can kill an ENTIRE in-flight fleet:
  boxes mid-collect lose their sync_back and self-destruct, AND not-yet-started
  boxes fail wait_ssh and self-destruct. More shards = MORE SSH endpoints = bigger
  failure surface. Mitigation: fewer shards (8 not 16) shrink the surface; the
  partial-sync mirror (sync/partial-box-*) preserves checkpoints up to the outage.
- Local MPS is NOT a viable fallback for the 24480-prompt grid: contended MPS ran
  ~4 samples/min (teacher-forced 3-way forward is heavy) => ~100h. Cloud RTX 4090
  is ~25x faster. Keep the run on cloud.
- A subsample STRIDE that is a multiple of the scaffold-block size (4 = none+3
  scaffolds) collapses onto ONE scaffold. Use a stride COPRIME to 4 (e.g. 17) for
  a subsample that still spans all scaffold conditions.
- Account balance can read transiently NEGATIVE mid-run (settlement lag) then
  recover; re-check `vastai show user` before concluding "out of credit".

## 2026-06-21 — Data loss: cloud budget drain stopped all boxes mid-run

ROOT CAUSE: an over-parallel push ran ~11 reliability-first Vast fleets at once
(~$22-44/hr aggregate). The burn drained the shared account $106 -> $0; when credit
crossed Vast's ~$5 threshold, Vast STOPPED EVERY INSTANCE ACCOUNT-WIDE, mid-run. At
that moment the study collectors synced results back only at the END of a box's run,
so every box stopped mid-grid lost ALL in-progress data (it lived on the box's local
disk, never reached sync/ quarantine). Lost: Qwen3-8B baseline; selection
Llama-70B/Llama-1B/Qwen-32B; full-grid 0.6B + 32B divergence; geometry Llama-1B/3B.

WHY NOT CAUGHT: no burn-vs-credit guard, no incremental sync-back, and "money is no
constraint" was misread as "unlimited concurrency" instead of "spend with rails".

PREVENTION (enforce):
1. INCREMENTAL sync-back: cloud/sync_partial.sh (commit e5f53da) mirrors each box's
   growing response_samples.json to quarantine every 180s and pushes the partial UP
   before any relaunch -> a stopped box resumes, never loses >3min. Collection also
   checkpoints every 25 samples.
2. CONCURRENCY CAP + BURN GUARD: cap concurrent boxes (~<=6); check `vastai show user`
   credit before launching; abort/pause as credit nears the threshold. Hourly loop is
   credit-gated (no launches under $50).
3. "money no constraint" != "infinite concurrency" -- a few checkpointed boxes beat a
   swarm that drains and dies.
4. VERIFY locally after each run (data present + plots render) before moving on.

## Concurrent agents share the working tree — your files can land in another agent's commit
- When multiple agents run in one repo, they share ONE working directory. A
  concurrent agent's `git add -A` / broad commit will sweep up YOUR new untracked
  files too. On this task, files I created (render_branching_tree.py,
  build_divergence_tree.py, forking_tree_model.py) showed up as ALREADY TRACKED in
  HEAD — committed by another agent's "plots overhaul" commit d3b499e, not me.
- Implication: don't assume `git status` untracked == "my uncommitted work". Before
  committing, diff working tree vs HEAD for YOUR files; commit only the residual
  modifications. Stage files by EXPLICIT path (never `git add -A`/`.`) so you never
  capture another agent's untracked files (I saw 2 stray files from another agent:
  plot_forking_commit_dynamics.py, thinking_belief_agreement_scatter.py — left them
  alone).
- out/ is gitignored here: produced figures are LOCAL artifacts, not committed.
  Deliverable = the code + the local PNGs, not PNGs in git.

## Forking-paths trunk selection on small models: use branch-divergence, not Δ_t
- The consecutive-barycenter "most-forking" index Δ_t = ||O_t - O_{t-1}|| is
  dominated by the noisy o_0->o_1 jump at t=0 (the <think> boilerplate token) on a
  tiny model. That token has a single alternate -> a DEGENERATE one-branch tree.
- The forking token is properly the position where re-sampling a different next
  token most diverts the outcome: max pairwise L2 between the alternates' o_{t,w}.
  Added most_divergent_branch_index(); use it as the tree trunk whenever the
  Bayesian CPD finds no SIGNIFICANT change point (BF<9).
- Token budget: Qwen3-0.6B needs >=384 (ideally 512) new tokens for thinking draws
  to close </think> and emit a parseable role; at 256 most draws land in the
  unparseable bucket and the "fork" is an artifact of truncation, not a real flip.

## Reasoning/chat-model detection must read the chat TEMPLATE, not the name
- A "Reasoning" checkpoint (Mistral Ministral-3-Reasoning) produced ZERO reasoning:
  response was 7 tokens, "Respuesta final: c)". Root cause was NOT the model — it was
  `_detect_chat_model`, which keyed off the NAME ("instruct"/"chat"/"-it"). "Reasoning"
  matched none -> treated as a BASE model -> `apply_chat_template` returned the bare
  prompt -> the `[SYSTEM_PROMPT]...[THINK]` reasoning scaffold baked into the template
  was never applied -> the model just answered.
- Fix: a non-empty `tokenizer.chat_template` IS the definitive chat signal (every
  instruct/reasoning ckpt ships one; base models don't). Name heuristics are a fallback.
- Reasoning families do NOT all use Qwen's `<think>`/`</think>`. Mistral uses
  `[THINK]`/`[/THINK]` driven by a template system-prompt. Detection, answer-segment
  splitting, and the runaway-CoT force-close must ALL use the model's OWN close token
  (added `ModelRunner.reasoning_close_marker`, read from the template), never a hardcoded
  `</think>`.
- Verify reasoning models by INSPECTING the generated text for the think block, not by
  trusting `is_reasoning_model` or the mode flag. "thinking mode" does nothing if the
  template that triggers it never reaches the model.

## Answer parser: search the whole answer segment, not just the tail
- A correct answer ("Respuesta final: z) <300-char explanation>") was marked invalid
  because the parser searched only the last 240 chars for the answer cue — the model
  stated the answer FIRST then explained, so the cue fell outside the tail window.
- Fix: search the WHOLE answer segment (everything after the last </think> / [/THINK])
  for the cue. Mid-reasoning mentions are already excluded by the think-close split, so
  the tail restriction bought nothing and broke answer-then-explain responses.

## label_prob and vocab_diversity must be read at the SAME trajectory index
- Symptom (user caught it): label_prob=0.94 yet vocab_diversity=3.84 for one sample —
  impossible, since 0.94 mass on the top token caps achievable diversity ~2.5.
- Root cause: GeneratedTrajectory.from_inference aligns BOTH views at the same index k:
  logprobs[k]=log P(t_k|prefix) AND full_logits[k]=the distribution that produced t_k,
  with logprobs[k]==log_softmax(full_logits[k])[t_k]. The readout took label_prob from
  logprobs[pos] (correct) but diversity from full_logits[pos-1] (the PRIOR token's
  distribution) — an off-by-one. A STALE compute_trajectory docstring claiming
  logprobs=[P(t1|t0),...] (length N-1, no leading 0) is what misled the indexing.
- Fix: use full_logits[pos] for the diversity. Verified by recomputing: softmax(full_logits[pos])[token]
  == label_prob exactly, and the invariant exp(entropy) <= max-given-p holds for all samples.
- General rule: when a probability and an entropy describe the SAME decision, compute BOTH
  from ONE distribution object, never index them separately. And sanity-check pairs with a
  cheap physical invariant (a p≈1 spike must force diversity≈1) — it instantly flags
  off-by-ones that look plausible in isolation.

## Forking: read the base path from existing response data — never re-decode it
- The stability/readout sweep already produced every model's full greedy response_text. The
  forking base path IS that response. Re-greedy-decoding it (the old decode_forking_base_path
  PHASE 1) is wasted compute AND decouples forking from the real data.
- Fix: build_branch_plan_from_text(runner, templated_prompt, base_path_text, ...) encodes the
  stored prompt_text + response_text and teacher-forces ONCE for the per-position fork logits —
  no generation. Every stored response is an independent base path -> forking is item-parallel
  by construction (a whole sweep forks without re-decoding anything).
- Remaining forking-parallelism work (prep): (1) a driver that reads out/stability/<model>/
  response_samples.json and forks selected items via build_branch_plan_from_text; (2) make
  fork_plan_positions resumable (skip positions already in dump_dir); (3) item-level fan-out
  (many items x position-shards); (4) fix old out/sesgo/forking paths -> out/forking; (5) reuse
  the robust stability box lifecycle (detached+resume+retries, kernels/torch FP8 fixes).

## ALWAYS collect/sync a box's data BEFORE killing or re-configuring it
- Killing a box SIGTERMs its driver -> EXIT trap runs vast_destroy -> the on-box checkpoint is
  GONE within seconds (destroy is fast; a post-kill rsync race loses). Lost ~1175 q9b prompts
  by killing before snapshotting when switching it to nonthinking-only.
- RULE: before ANY kill/re-shard/mode-change of a running box, FIRST rsync its /root/apart/out/
  to sync/partials/<tag>/ (and for shards, pull every shard + merge). Only then SIGTERM.
- The driver only syncs at completion, so mid-run progress lives solely on the box until then.

## 2026-06-24 — Never edit a script while it is executing
Edited cloud/run_one_forking_box_32b.sh (IID_FILE line) WHILE a launched copy was mid-run.
bash reads scripts by byte offset; inserting lines above the live execution point can
corrupt the running process and (worst case) skip the destroy-trap → runaway GPU billing.
Got lucky (execution was already past the edited region). RULE: finish/needed edits to a
cloud driver BEFORE launching it; for changes mid-flight, copy the script to a new path and
launch that, or wait for the run to finish.

## 2026-06-25 — Cloud billing-safety incident (NetAlerts TLS interception)

Root cause of a multi-hour billing leak + 20 phantom "failures":
1. **A network TLS-intercepting filter ("NetAlerts") blocked the vast.ai API.** github worked; only console.vast.ai was intercepted (cert issuer=NetAlerts Services, unverifiable). It rode the VPN egress, so switching the underlying network didn't help — disconnecting the filtering VPN did.
2. **`destroy_box` FALSE-confirmed destroys during the outage.** Its verify-loop did `except: break` then `print("no")`, so an unreachable API printed "no longer listed" → two 9B boxes billed for hours while logs claimed them destroyed. FIX: on parse failure print "unknown", never "no"; record unverified ids to `cloud/.pending_destroy`; added `cloud/reap_pending.sh` to drain it. ALWAYS re-verify billing via the live API after any network disruption — a destroy "confirmation" logged during an outage is worthless.
3. **The search step masked an API outage as "no matching offers"** (json.load crashed on empty stdin). FIX: distinguish empty/non-JSON (exit 21, "API unreachable") from a valid `[]` zero-offer result; retry wrapper aborts immediately on rc=21 instead of burning retries.
4. **Relaunching with `> log` OVERWROTE the original failure logs**, destroying the destroy-confirmation evidence I needed for the billing audit. NEVER overwrite a run log; use timestamped log names (`.sh_<box>_<HHMMSS>.log`).
5. **A Bash-tool call that exceeds the 120s timeout kills its whole process group — including `nohup`'d children.** The first 7-box launch (6×20s foreground) timed out and killed 6 boxes mid-lifecycle (one orphaned a billing instance). `nohup` survives ONLY if the launching call returns normally. Keep launch calls < 120s; stagger ≤10s × N.
6. **macOS has no `setsid`.** Don't use it for detachment; plain `nohup … &` + a short-enough foreground call is the portable path.
