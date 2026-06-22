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
