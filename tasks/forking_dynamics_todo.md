# DYNAMICS (forking-paths) deliverable — LOCAL run + branching-tree viz

## Goal
Produce a rich forking-paths O_t figure locally (Qwen3-0.6B, MPS) + add a
left-to-right branching-TREE plotter (arXiv:2601.06116 style) and wire it into
BOTH the forking study and the divergence study.

## Plan
- [x] 1. Select forking item -> selected_item.json (idx=6, gender gossip, ent=0.703)
- [~] 2. Collect rollouts LOCALLY: 60 positions, N=40, max_new=384 (RUNNING ~100min)
- [ ] 3. Analyze (change-point, pull/drift/potential, diversity, survival)
- [ ] 4. Plot O_t (existing plot_forking_dynamics.py)
- [x] 5. NEW plotter: render_branching_tree.py (Okabe-Ito, horizontal-bar NODES, curved edges)
- [x] 6. Wire branching-tree into forking driver (plot_forking_dynamics.py emits both)
- [x] 7. Wire branching-tree into divergence viz (build_divergence_tree.py + viz call)
- [x] 8a. VIEWED divergence tree + synthetic forking tree — clean, matches Fig.11
- [ ] 8b. VIEW real O_t + real forking tree once collection done
- [x] 9. Updated forking README + EXPLANATION (divergence has no .md docs)
- [ ] 10. Commit + push (fetch+rebase; touch only sesgo/forking + sesgo/divergence + src/dynamics/forking_paths tree files)

## Reference style (arXiv:2601.06116 Fig.11/22, viewed from PDF)
- left-to-right root->trunk->2-3 labelled branches; NODES = horizontal-bar outcome
  distributions; curved colored edges (width ∝ prob mass); edge token labels;
  Okabe-Ito palette; outcome-category legend. MATCHED.

## Constraints
- NO cloud. uv run only. MPS HuggingFace backend.
- AVOID geometry/, cross_model_*.py, baseline_sample_plots.py, paper/
- Checkpoint/resume-safe; out/sesgo/forking/Qwen3-0.6B/ in MAIN repo

## Review (DONE)
- Local forking run on Qwen3-0.6B (MPS, HF): 60 positions, S=40, N=60 prior,
  384-tok rollouts, ~60 min, 92 branches. selected item idx=6 (gender gossip).
- o_0=[0,.32,.13,.55] -> o_T=[.03,.65,.30,.03]; target excursion ~t40, unknown swing ~t50.
- CPD: NO significant change point (BF=0.023, p(m=0)=0.98) — honest negative, noisy
  small-model o_t. Did NOT fabricate a change point.
- Branching-tree trunk = most-divergent-branch t=42 (","); alt " and" (p=0.12) swings
  outcome to "other" — a real fork. pull/drift/potential @ t42 = .284/.341/.285.
- 3 figures render clean + match Fig.11 reference: forking_dynamics.png,
  forking_branching_tree.png, divergence/plots/branching_tree.png.
- Branching-tree renderer (render_branching_tree.py) shared by BOTH studies.
- Committed 535390c, pushed to origin/main. (My new files render_branching_tree.py,
  build_divergence_tree.py, forking_tree_model.py + early README/viz edits were
  swept into a CONCURRENT agent's commit d3b499e; my later trunk-selector fix is 535390c.)
