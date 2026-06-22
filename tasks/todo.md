# FINAL PUSH (hackathon, hours left) — parallel work tracker

## RUNNING — cloud (all credit-gated, destroy-on-done; ~$57 credit)
- [ ] forking Qwen3-0.6B full-CoT  (af84145d) — O_t + forking_positions/
- [ ] forking Qwen3-32B dynamics   (a8e10be1) — 50/pos, parallel, fastest backend, token excerpts; in uv-sync
- [ ] divergence Qwen3-0.6B + 32B  (a6bbef4f) — 1 prompt x ~100 samples + per-gen vocab entropy
- [ ] selection Qwen3-32B          (a0b107cf) — the empty cell (re-launched after a crash; no orphan)

## RUNNING — local polish workflow wlpewfk4m (5 agents)
- [ ] dynamics 5-panel restored+polished (O_t+token strip+pull/drift/potential+single diversity+changepoint), minimal text, fork-token highlights
- [ ] minimal-text baseline/cross_model/full_data + gemma-FREE re-render + promote bias VARIANT B to canonical bias_alignment_accuracy.png
- [ ] minimal-text divergence/selection/steer (kill all how-to-read/verbose legends)
- [ ] F3 stability split by format axis (label vs role order), minimal
- [ ] geometry minimal-text + depth_scatters ALL colour axes in per-axis subfolders

## WHEN WORKFLOW DONE — MY verification + integration pass (do NOT skip)
1. [ ] RENDER-PASS: actually RUN every figure script; catch crashes. KNOWN BUG:
       sesgo/baseline/cross_model_disambig_scaling_figure.py — plot_disambig_scaling() def takes
       (cells,out_path) but main() calls it with (cells,out_path,n_models). FIX signature mismatch.
2. [ ] IMAGE-VERIFY every figure (minimal text? clean? gemma gone? legends outside?). Iterate until polished.
3. [ ] Integrate updated/new figures into appendix/*.tex (bias B already at canonical path; depth-scatters; F3; dynamics).
4. [ ] Rebuild paper; verify PDF with image tokens (0 placeholders, 0 errors).
5. [ ] git add + commit + push (paper appendix + sesgo/ figure code).

## WHEN CLOUD DONE
6. [ ] Sync + VERIFY data quality (NOT uniform/NaN like gemma) for each: forking-32B, divergence, selection-32B.
7. [ ] Re-render the affected figures (dynamics-32B, divergence re-scoped, selection-32B) + integrate.

## FINAL
8. [ ] HF re-upload out/ (now 3.7G, clean) EXCLUDING **/*.log and **/activations/** ; confirm commit lands.
9. [ ] Final out/ audit (no shards/activations/gemma/degenerate cells); honest gaps note.

## DONE this push
- out/ cleaned 68G->3.7G (0 shards, 0 activations, gemma-2-27b-it removed — was degenerate uniform 1/3)
- paper modularized (homo-style: main.tex thin + paperdefs + sections/ + appendix/ + figures/)
- stability grid completed on cloud (Qwen 0.6B/32B, Llama 1B/70B); F3 within-family
- bias variant B chosen (clean: realizable triangle + ideal point, family-hue x size-shade, single legend, no how-to-read)
