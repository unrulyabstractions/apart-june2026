# SESGO paper — state at 2026-06-21 (deadline tonight 11:59pm AoE)

## DONE + on main (pushed)
- Plot overhaul: ALL families plain-language (no 3-opt/non-thinking/p(UNKNOWN)/chance), shared
  sesgo/common/plain_language_labels.py. Reviewer blocker fixed (full_data_axis jargon + stale PNG).
- Geometry Qwen3-32B rendered (0->37 plots); O(n^2) analyze fixed; thinking_outcome axis.
- Dynamics: full 60-position forking O_t + change-point + both branching trees. HONEST result =
  no significant single change point on 0.6B (Bayes factor 0.023); strongest fork t=42.
- 7 NARRATIVE KEY FIGURES (one per study), each image-verified by me in the PDF:
  F1 outcome-mass-vs-scale mosaic, F2 disambig accuracy + wording gap, F3 format-invariance
  small-vs-big, F4 pre/post-think agreement, F5 scaffold->abstention delta, F6 O_t/change-point,
  F7 scaffold-boundary PCA (separation 0.71 @ layer 14). Reviewer punch-list fixed (F4 declutter,
  F6 CVD colours, F2 plain ticks, full_data blocker, geometry caption ranges).
- Paper: all 7 + hero breakdown + branching tree integrated into per-study appendices with
  finding-based captions. Builds clean: 18pp, 0 missing figures, 0 errors. Geometry n reconciled
  to the full 4620-sample run.
- Data-loss root-cause + prevention in tasks/lessons.md; agents-stay-in-worktrees rule in CLAUDE.md.

## REMAINING (small)
- [ ] Cosmetic nits (reviewer LOW): selection title/subtitle order (gloss above title on 3 figs);
      geometry pca_scatter_label annotation box overlaps the orange cloud. Both optional polish.
- [ ] §Results in the MAIN body is a deliberate STUB — the AUTHOR writes it (per request).

## TRULY-LOST cells (need a cloud re-run — DEFERRED, documented, NOT run unattended)
- baseline Qwen3-8B; selection Llama-70B/Llama-1B/Qwen-32B; geometry Llama-1B/3B.
- partial-but-usable + shown honestly: baseline Qwen3-32B (n=1376), gemma-2-27b-it (degenerate,
  flagged in F2). Full divergence beyond n=47 not run (user: do NOT sample more).
- To fill in the morning (watch the spend): a small credit-gated, checkpointed fleet via cloud/.
