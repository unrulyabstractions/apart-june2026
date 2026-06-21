# Cross-family GEOMETRY study — todo (geom_multi controller)

FLEET_DIR = cloud/.fleet_geom_multi  (ISOLATED from .fleet_cmd*, .fleet_div32, etc.)

Goal: extend the scaffold-direction geometry finding ACROSS families. Qwen has
geometry already; add Llama-3.2-1B, Llama-3.2-3B, gemma-2-2b, Mistral-7B-Instruct-v0.3.
Question: is the scaffold the dominant separable axis in Llama/Gemma/Mistral too?

Constraints: money not a constraint; reliability-first; background; self-destruct;
parallelize hard. STUDIES=geometry, N_THINKING=0, SUBSAMPLE~0.07.

## Plan
- [x] Explore fleet + geometry scaffolding
- [x] Confirm cross-family structural_markers (Qwen/Llama/Gemma/Mistral) + n_thinking 0 path
- [x] HF_TOKEN present (gated Llama/Gemma/Mistral); vast credit ~$79
- [x] 25 OTHER live instances from concurrent fleets -> NEVER touch them; isolated FLEET_DIR
- [x] Plan TSV: 4 models x 4 shards = 16 RTX_4090 boxes (cloud/.plan_geom_multi.tsv)
- [x] Launch 16 boxes concurrently; fixed 3 co-locations (relaunched on distinct hosts)
- [~] Drive geometry shards (fleet_run.sh bg PID 70067), self-destruct each on completion
      - Llama-3.2-3B__shard0of4 SSH never resolved -> driver destroyed it -> RELAUNCHED
        (instance 41985383) + driven by one-box bg driver. All 16 shards accounted for.
- [ ] combine_geometry_shards.py per model; merge to out/ scoped
- [ ] analyze_geometry.py + visualize_geometry_samples.py per model
- [ ] Extract per-model best-layer scaffold silhouette
- [ ] fleet_destroy backstop; confirm 0 of MY boxes billing
- [ ] Commit only changed cloud/sesgo files (never paper/)

## Models -> repo ids -> GPU (RTX_4090, <=8B tier)
- Llama-3.2-1B  -> meta-llama/Llama-3.2-1B-Instruct
- Llama-3.2-3B  -> meta-llama/Llama-3.2-3B-Instruct
- gemma-2-2b    -> google/gemma-2-2b-it
- Mistral-7B-v3 -> mistralai/Mistral-7B-Instruct-v0.3

## Sharding: geometry grid=4620; subsample 0.07 -> ~323/model; 4 shards -> ~80/box

## Cross-family baseline (existing Qwen, label position, best layer)
- Qwen3-0.6B: layer 14/28 (0.50 depth) scaffold_silhouette = 0.635
- Qwen3-1.7B: layer 14/28 (0.50 depth) scaffold_silhouette = 0.341
- Qwen3-4B:   layer 23/36 (0.64 depth) scaffold_silhouette = 0.340

## Review
(to fill in)
