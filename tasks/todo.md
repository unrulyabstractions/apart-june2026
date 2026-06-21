# Overnight autonomous finish (deadline 2026-06-21 11:59pm AoE)

User went to bed: "get this done, finish, check all data and plots." Work locally + safely
(NO unattended cloud fleet — that caused the data-loss drain). Document truly-lost cells.

## In flight (will re-invoke me on completion)
- [ ] Workflow `wbskr4421` — plain-language overhaul of ALL plot families (baseline, cross_model,
      divergence, geometry, stability, selection, steer) + divergence thinking/non-thinking
      distributions from EXISTING data. Agents leave edits UNCOMMITTED in main tree for me to merge.
      Ends with an independent image-token reviewer punch-list.
- [ ] Dynamics agent `abc175e9` — local forking rollouts + branching-tree viz in apart-forking worktree.

## When workflow completes (DO IN ORDER)
1. [ ] Read workflow result: apply the reviewer punch_list (fix residual jargon/clutter myself).
2. [ ] Review ALL uncommitted changes in main tree (git diff), sanity-check each family's generators.
3. [ ] Render geometry Qwen3-32B static PNGs (data present, 0 plots): analyze done by perf agent;
       run the geometry plot driver on Qwen3-32B with the NEW plain-language code.
4. [ ] Re-render every family so all PNGs are consistent with the merged code.
5. [ ] Open a sample of EACH family's PNGs with image tokens myself — confirm crisp + no jargon.
6. [ ] Commit + push all plot-overhaul changes to main (I own all merges; agents never git).
7. [ ] Merge dynamics/branching-tree work from apart-forking into main when that agent is done.
8. [ ] Integrate hero abstention_breakdown.png + divergence distributions + branching-tree into
       paper appendices; rebuild paper; VERIFY the PDF with image tokens (no overfull/missing figs).
9. [ ] Final data+plot audit; write honest GAPS section (truly-lost cells below).
10.[ ] Commit + push paper + final state.

## Truly-lost cells (need a cloud re-run — DEFERRED, document, do not drain budget unattended)
- baseline Qwen3-8B (no data)
- selection Llama-3.1-70B / Llama-3.2-1B / Qwen3-32B (no data)
- geometry Llama-3.2-1B / Llama-3.2-3B (no data)
- full divergence run (only n=47 pilot + n=16164 full_data exist; user said DON'T sample more)
- partial-but-usable: baseline Qwen3-32B (n=1376), gemma-2-27b-it (n=1408, possibly degenerate readout)

## Done this session
- geometry UI default -> Qwen3-0.6B (was alphabetical Llama-70B)
- data-loss root-cause + prevention -> tasks/lessons.md
- sesgo/common/plain_language_labels.py (shared label vocabulary)
- full_data hero plot: abstention_breakdown.png (thinking x scaffold x wording per category)
- geometry O(n^2) fix + Qwen3-32B analyzed + thinking_outcome axis (perf agent, main @ 86b105c)

## *** NARRATIVE KEY FIGURES — ONE PER STUDY (user spec 2026-06-21, TOP PRIORITY) ***
Build AFTER the polish workflow merges (on the stable plain-language base). New self-contained
modules; image-token verify each. Use sesgo/common/plain_language_labels.py. Categorize EXISTING
data only (no new sampling).
- F1 BASELINE-a: ambiguous-setting target/other/UNKNOWN (t/o/u) distribution AS MODEL SCALES UP,
  faceted per family (Qwen/Llama/gemma/Mistral) x bias type x neutral/negative. Stacked mass vs size.
- F2 BASELINE-b: disambiguated ACCURACY as model scales up, per family x bias type; PLUS the
  disambiguated NEGATIVE-vs-NEUTRAL question gap per family/size x bias type.
- F3 STABILITY: how often are choices independent of FORMATTING — smallest (Qwen3-0.6B) vs biggest
  (Llama-3.1-70B). Format-invariance / consistency rate.
- F4 DIVERGENCE: does the non-thinking t/o/u logprob distribution MATCH the thinking-CoT outcome
  t/o/u? Agreement/calibration of pre-think P(role) vs post-think outcome freq, per item.
  EXISTING n=47 rollouts only (user: do NOT sample more, just categorize).
- F5 SELECTION: how does the BASELINE change WITH scaffolding? Paired baseline->scaffold delta on
  abstention/accuracy. Anchor power on full_data (16164); show selection n=35 honestly.
- F6 DYNAMICS: arXiv:2601.06116-style TWO-panel — top: outcome distribution O_t (stacked area of
  candidate answers over CoT token index); bottom: change-point prob p(tau=t|y) w/ commit token
  highlighted. On a SESGO CoT rollout. (dynamics agent abc175e9 is building this.)
- F7 GEOMETRY: the single layer x position with the MOST structure — 2D/3D PCA scatter whose colour
  (scaffold/outcome) shows a clear gradient or decision boundary. Pick by max silhouette, CONFIRM
  by viewing candidates with IMAGE TOKENS.
