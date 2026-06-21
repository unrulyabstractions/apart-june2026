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
