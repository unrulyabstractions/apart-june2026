# Cloud Commander Mission — todo

FLEET_DIR = $PWD/cloud/.fleet_cmd (separate from other controllers' .fleet*)

## PART A — Baseline size sweep (4 missing/sparse models -> full 2310)
- [ ] Qwen/Qwen3-8B          (missing)          -> RTX 4090 (8B)
- [ ] Qwen/Qwen3-14B         (missing)          -> A100 (<=16)
- [ ] google/gemma-2-9b-it   (never completed)  -> A100 (9.2B). HF_TOKEN set -> unblocked.
- [ ] meta-llama/Llama-3.1-8B-Instruct (96/2310) -> RTX 4090 (8B)
- [ ] Do NOT relaunch the 8 complete models. Verify each lands 2310.

## PART B — Geometry sharded (FULL 4620 grid = scaffold+none)
- [ ] CONFIRM --shard-index/--shard-count -> YES (already present, contiguous via apply_shard). No code change.
- [ ] Qwen/Qwen3-0.6B across K>=6 boxes (RTX 4090), --n-thinking 0, batched.
- [ ] Qwen/Qwen3-32B across K>=4 H100 boxes, --n-thinking 0, batched.

## PART C — Safe shard-combine merge (NEW CODE)
- [ ] sesgo/geometry/combine_geometry_shards.py (run-by-path, house style, <=150 lines, BaseSchema).
- [ ] dedup by sample_identity, union activations into one tree, write one response_samples.json.
- [ ] Verify combined count == sum of shard counts; every referenced .pt exists.
- [ ] analyze + visualize on the combined set.

## KEY DESIGN FINDING
- sample_idx is GLOBAL (0..4619) and PRESERVED through apply_shard.
- Activation filenames = sample_<global_idx>_<pos>_L<layer>.pt -> already DISJOINT across shards.
- Combine is clobber-safe: concat samples + union activations, no path rewriting.
- merge_sync.sh --ignore-existing keeps only first shard's response_samples.json -> hence dedicated combine.

## VERIFY FIRST (tiny local pilot)
- [ ] K=2, --subsample 0.01 local geometry run for Qwen3-0.6B (HF backend on MPS/CPU).
- [ ] Combine; assert combined count == shard0 + shard1.

## PROGRESS (live)
- [x] PART C combine script written + LOCAL PILOT PASSED (K=2, subsample 0.01: 24+24=48, all .pt present, idempotent).
- [x] Committed: 84d54ba (combine + README), 2aeda37 (FLEET_PLAN_FILE override). Both pushed.
- [x] PART B sharding CONFIRMED present (no code change to collect script).
- [x] Launched 4 baseline boxes (FLEET_DIR=.fleet_cmd_baseline) + 10 geometry boxes (.fleet_cmd_geo).
- [x] INCIDENT: 7 of 14 boxes co-located 2-per-machine; the 2nd instance on each shared
      machine stalled in 'loading' and hit wait_running's 20min timeout. Root cause:
      concurrent reliability-first searches all returned the SAME top offer/machine.
      RECOVERY: destroyed the 7 stalled boxes (stopped idle billing), relaunched all 7
      on DISTINCT machines via /tmp/relaunch_distinct.sh, fixed one further collision
      (Llama+0.6B-s4 both on 14826 -> moved 0.6B-s4 to 1831). 7 orig-running + 7 retry.
- [ ] Boxes running (monitor ban9y6jfj armed: completion + OOM across all 4 fleet dirs).
- [ ] Baseline: merge_sync -> verify 2310 each.
- [ ] Geometry: combine_geometry_shards Qwen3-0.6B + Qwen3-32B -> verify counts.
- [ ] analyze + visualize combined geometry.

## REVIEW
(filled at end)
