# mental_risk experiment — mirror SESGO 5-study surface

## Mapping (SESGO -> mental_risk)
- SESGO scaffold (debiasing preamble, intervention axis) -> mental_risk FRAMING
  (at_risk_of/suffering/safe/intervene). No no-op baseline framing exists, so
  "selection" picks the best FRAMING by gold-correlation, not scaffold-vs-baseline.
- SESGO format axes (label_style x permutation) -> mental_risk format axes
  (label_style x order_flip x scale_high) for stability.
- SESGO gold = always UNKNOWN, accuracy = abstention -> mental_risk gold =
  continuous gold_risk; "fit" = Pearson r / MAE vs gold (no abstention notion).
- SESGO divergence (thinking role-dist entropy/JS) -> mental_risk divergence
  (thinking score-cloud entropy/std/dispersion via ScoreSummary).
- SESGO geometry (residual capture, force HF) -> mental_risk geometry (residual
  capture along the binary CATEGORIZE answer path, force HF via BinaryChoiceRunner).

## Library code (src/datasets/*_risk) — reuse existing, add geometry eval
- [x] src/datasets/mental_risk/ loader (exists, reuse)
- [x] src/datasets/risk/ querier + schemas (exists, reuse)
- [x] src/datasets/prompt/ risk_* generator (exists, reuse)
- [ ] src/datasets/risk_geometry/ : geometry schemas (RiskGeometryActivation,
      RiskGeometrySample, RiskGeometryDataset) + RiskGeometryQuerier capture.
      (suffix unique vs sesgo's geometry_*.py)

## Drivers under mental_risk/<pipeline>/ (run-by-path, parents[2] bootstrap)
- [ ] mental_risk/scaffolds_risk.py  — the framing set accessor (get_risk_framings)
- [ ] mental_risk/generate/generate_risk_prompt_datasets.py — emit 5 per-study grids
- [ ] mental_risk/baseline/  (keep generate/collect/visualize, repoint to baseline dir)
- [ ] mental_risk/stability/ collect_stability_risk.py + visualize_stability_risk.py
- [ ] mental_risk/selection/ collect_selection_risk.py + visualize_selection_risk.py
- [ ] mental_risk/divergence/ collect_divergence_risk.py + visualize_divergence_risk.py
- [ ] mental_risk/geometry/ collect_geometry_risk.py + visualize_geometry_risk.py
      + analyze_geometry_risk.py + geometry_viz_server_risk.py + visualize_geometry_risk.sh

## Verify
- [ ] py_compile every new file with main .venv python
- [ ] import-verify each driver via sys.path bootstrap
- [ ] run generate on the synthetic fixture if data present; else flag blocker
- [ ] docs: README/EXPLANATION in each new folder
- [ ] commit in worktree; confirm disjoint from sesgo/

## Blockers
- MentalRiskES corpus is encrypted+absent in worktree (needs MENTALRISK_ZIP_PASSWORD)
  and fixture data/ dirs are empty -> real model run must happen post-merge.

## Review (done)
- Library: added src/datasets/risk_geometry/ (5 modules + README + EXPLANATION):
  RiskGeometryActivation/Sample/Dataset, capture engine, PCA/projection engine.
  All auto-export; 13 public symbols import cleanly; added to tests/test_imports.py.
- Drivers under mental_risk/ (run-by-path, parents[2] bootstrap), all 5 studies:
  generate (1 multi-study generator), baseline/stability/selection/divergence
  collect+visualize, geometry collect+analyze+visualize+server(+page)+sh.
  Shared: scaffolds_risk, subject_resolution, risk_sample_io, risk_prediction,
  framing_ranking. mental_risk/__init__.py is empty (mirrors sesgo/__init__.py).
- Removed 3 old baseline drivers; this also fixed a PRE-EXISTING rule-5 clash
  (mental_risk/baseline/generate_prompt_dataset.py vs sesgo's same name).
- VERIFIED: py_compile all; run-by-path --help on all 12 py drivers; bash -n the
  .sh; generator run on a synthetic fixture -> correct per-study grids; geometry
  analyze+visualize+FastAPI server end-to-end on synthetic activations; all 4
  study visualizers on synthetic RiskDatasets; tests/test_imports.py 2 passed.
- DISJOINT from sesgo/ (git status confirms zero sesgo changes). No single-word
  filenames; all my filenames globally unique.
- BLOCKER stands: live model+data run needs the decrypted corpus (post-merge).
