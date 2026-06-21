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
