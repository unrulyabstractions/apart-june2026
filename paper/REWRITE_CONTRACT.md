# PAPER REWRITE CONTRACT (single source of truth — both writer agents obey this)

Radical restructure of the SESGO hackathon paper to the author's new outline. Two
writers work disjoint file sets but MUST agree on: section/appendix labels, the 3
claims, notation, figure paths, and the placement decisions below. Build with
`cd paper && bash build.sh` (or latexmk). Macros live in `paperdefs-common.tex`:
`\plotfig{<full ../out path>}{<full ../out path OR bare name>}{<width>}{<caption>}{<label>}`
(guarded — missing file → placeholder, never breaks build) and
`\plotsub{<full path>}{<full path>}{<width>}` for subfigures. `\Cref{}`, `\LLM`,
`\citep{}`. KEEP the existing title block, author block, footnote, and the LLM
Usage Statement in main.tex. Length target: 4 pages main body (excl. refs+appendix).

## TITLE (unchanged)
"The Shapes of Bias in Spanish-Prompted LLMs and the Debiasing Prompt Scaffolds"

## ABSTRACT (rewrite per the advice below)
Order, one idea per sentence, accessible, concrete metrics, no overclaim:
1. context (LLMs encode/​amplify human social bias, harming minoritized groups);
2. gap (interventions must be context-aware; until recently no Spanish bias-eval data);
3. main claim (we characterize bias in Spanish-prompted open-weight LLMs across
   MODEL SCALE and PROMPT SCAFFOLDING via behavioral, distributional, and geometric
   studies on the new SESGO benchmark);
4. clarifying detail (SESGO: ambiguous items → correct answer is "unknown"
   (abstain); disambiguated → a specific group; a debiasing scaffold is a
   one-sentence preamble);
5. evidence — the 3 claims with numbers (smaller models more biased; scale lowers
   reasoning uncertainty 0.58→0.44 nats from 0.6B→32B; the best scaffold
   interpretive_direction lifts abstention 85.6%→99.9% and is the dominant
   separable axis in representation space, +0.48 silhouette);
6. impact (preliminary; gives future LATAM-context interventions a behavioral +
   geometric map; we release the harness). Signal "preliminary."

## THE 3 CLAIMS (results = CLAIM/EVIDENCE(+Figure)/LIMITATION/NOTES per claim)
CLAIM 1 — Smaller models are more biased.
  EVIDENCE: cross-model sweep (12 models, 4 families, n=770/cell ambiguous). On
  AMBIGUOUS items abstention-accuracy RISES with scale and the error split
  F(target)−F(other) narrows: e.g. Llama-3.2-1B ambiguous acc 0.13, bias_score
  0.87 (widest, most biased); large Qwen (14B 0.93, 32B 0.95) sit high+narrow near
  0. Disambiguated accuracy also rises with scale.
  FIGURES: ../out/sesgo/baseline/cross_model/plots/bias_alignment_accuracy.png
  (segments per model, Eq.1 bias score), baseline_scaleup_outcome_mosaic.png,
  baseline_size_sweep.png.
CLAIM 2 — Scale lowers (reasoning) uncertainty.
  EVIDENCE: divergence study — one ambiguous item, 100 thinking draws, per-draw
  next-token (vocab) entropy. Qwen3-0.6B 0.577±0.049 nats vs Qwen3-32B 0.443±0.041
  (~23% lower, tighter, disjoint ranges); both abstain where "unknown" reachable
  (94% vs 96%); only the forced 2-option readout splits them, in opposite
  directions across scale.
  FIGURES: ../out/sesgo/divergence/Qwen3-0.6B/plots/thinking_belief_agreement_scatter.png
  + ../out/sesgo/divergence/Qwen3-32B/plots/thinking_belief_agreement_scatter.png
  (per-draw vocab-entropy distributions, side by side).
CLAIM 3 — Simple scaffolding could help de-bias models.
  EVIDENCE: full-data run on Qwen3-0.6B selects s*=interpretive_direction →
  abstention 99.9% vs 85.6% no-scaffold baseline; NOT all scaffolds help (one is
  below baseline). Aligns with geometry (scaffold = dominant separable axis, +0.48
  silhouette). NO steering results in the paper (steering = Future Work only).
  FIGURES: ../out/sesgo/full_data/Qwen3-0.6B/plots/bias_alignment_accuracy.png,
  scaffold_abstention_delta.png, abstention_breakdown.png; geometry
  ../out/sesgo/geometry/Qwen3-0.6B/plots/pca_by_scaffold.png, scaffold_boundary_pca.png.

## THE 5 STUDIES (methods 3.1) + APPENDIX MAPPING
Methods 3.1 = ONE succinct paragraph + a TABLE; all technicality to appendix.
TABLE columns: Study | Models | Purpose | Method / Reference.
  - Model Scale  → App. A. (replicate SESGO baseline across 12 models; compare
    thinking vs non-thinking readouts). Reuse appendix/baseline.tex + cross_model.tex.
  - Stability    → App. B. (answer stability over label style + option order).
    Reuse appendix/stability.tex.
  - Scaffold effect → App. C. (does scaffolding improve SESGO abstention).
    Reuse appendix/selection.tex + full_data.tex + DIVERGENCE
    (appendix/divergence.tex: per-draw vocab entropy 0.6B vs 32B + thinking-vs-
    non-thinking). DIVERGENCE LIVES IN C. **NO STEERING ANYWHERE** — steering
    appears ONLY as one line in Future Work 5.2; DELETE appendix/steering.tex and
    remove every steering figure/claim from the body.
  - Dynamics     → App. D. = **FORKING ONLY** (appendix/dynamics.tex, forking-paths,
    Qwen3-14B, significant commit token t=270). Divergence does NOT go here.
  - Geometry     → App. E. (PCA per layer at change-of-turn token positions).
    Reuse appendix/geometry.tex.
EACH appendix X = "\subsection*{X.1 Extended Methodology}" + "\subsection*{X.2
Extended Results}" (bulleted facts + figures). Keep existing \label{app:...}
anchors where possible; new canonical labels: app:scale, app:stability,
app:scaffold, app:dynamics, app:geometry. Update appendix/index.tex to input A–E
in order (A baseline+cross_model, B stability, C selection+full_data+DIVERGENCE,
D dynamics-FORKING-ONLY, E geometry). NO steering appendix.

## CONTRIBUTIONS (intro end, bullet list)
- Extension of SESGO to evaluate MODEL SCALE effects (open-weight Qwen/Llama/Mistral).
- Behavioral + geometric analysis of debiasing prompt-scaffold effects.
- A chain-of-thought DYNAMICAL case study of bias commitment.

## RELATED WORK (sec 2) — 3 short subsections, use notation, be succinct
2.1 SESGO: cite \citep{robles2025sesgo} (arXiv:2509.03329). Reproduce/adapt its
  Figure 1 (the racism example: popular saying → stereotype → ambiguous|disambiguated
  context × non-negative|negative question → answers {Unknown, Other group, Target
  group}). Define notation: item has context condition c∈{ambig,disambig}, polarity
  ∈{neg,nonneg}; on ambiguous gold=Unknown (abstain); F(Target),F(Other)=share of
  INCORRECT answers harming the marginalized Target / the Other group; bias
  alignment = F(Target)−F(Other); bias_score = σ·√((1−acc)²+(F(Target)−F(Other))²)
  (Eq.1). Our extension = model scale. [SESGO Figure 1: include
  ../out/figs/sesgo_framework.png if present, else \TODO a re-drawn schematic +
  cite — FLAG to author that their Fig 1 should be re-created/attributed, do NOT
  copy the copyrighted image verbatim.]
2.2 CoT Dynamics: cite forking-paths \citep{forkingpaths} (arXiv:2412.07961) and
  \citep{forkingpaths2} (arXiv:2601.06116). We fork the greedy CoT at each token
  position t, sample continuations, parse to an outcome, and track O_t (the outcome
  distribution); a Bayesian change-point posterior tests for a single commitment
  token. Succinct, with notation.
2.3 Representational Geometry: PCA of residual-stream activations; the change-of-turn
  token as an interesting probe site, cite \citep{turntoken1} (arXiv:2604.07729) and
  \citep{turntoken2} (arXiv:2606.05194).

## INTRODUCTION (sec 1) — follow the author's bullets + abstract advice arc
Context (LLMs amplify pretraining bias, harm minoritized) → need for many
interventions incl. mech-interp / rep-engineering → interventions must be
context-aware → we tackle adapting interventions to the LATAM context → until
recently no Spanish bias-eval data → we leverage SESGO to characterize bias in
Spanish via behavioral/geometric/distributional studies → goal: inform future
LATAM interventions. Two LATAM-specific framings: (1) MODEL SIZE on OPEN-WEIGHT
LLMs — open models for sovereignty \citep{sovereignty} (arXiv:2412.12004) +
adaptation happens via open models; size matters because resource-poor adoption
favors smaller (possibly more harmful) models; (2) SCAFFOLDING — regular users
don't prompt-engineer, so we test how a scaffold shifts bias behavior + geometry.
Preliminary results: smaller models more biased; scale lowers uncertainty; simple
scaffolding could help. End with the 3 contributions (above).

## REFERENCES to add to references.bib (natbib keys)
robles2025sesgo (exists; arXiv:2509.03329), forkingpaths (2412.07961),
forkingpaths2 (2601.06116), sovereignty (2412.12004), turntoken1 (2604.07729),
turntoken2 (2606.05194). Minimal @article/@misc entries with arXiv eprint; if a
key already exists in references.bib, reuse it.

## DISCUSSION (sec 5)
5.1 Limitations: stability study implies other studies should filter unstable
  answers; small number of scaffolds; no clean resolution between small/large model
  on every axis; dynamics is a single-item case study.
5.2 Future Work: more geometric visualization methods; intervene/inspect in
  activation space; deepen scaffold analysis with many more scaffolds; steering.

## CONCLUSION (sec 6): short (3–5 sentences).

## REPRODUCIBILITY (3.2): code repo + HuggingFace dataset
(reuse sections/code_and_data.tex content / out/ + the HF dataset
unrulyabstractions/apart_global_south).

## HARD RULES
- HONESTY: preliminary framing; never overclaim; report n; the 0.6B selection /
  geometry / steering are single-model. Keep the LLM Usage Statement.
- Minimal-text figures already exist; reference by the exact paths above.
- Do NOT invent numbers — pull from the existing appendix .tex (already verified)
  or the data; if unsure, reuse the existing appendix wording.
- Forking figure is Qwen3-14B (significant commit token t=270, Bayes factor ≈1e9);
  0.6B forking accretes gradually (BF 0.023) — size contrast.

## AUTHOR'S WRITING VOICE — MIRROR IT AS CLOSELY AS POSSIBLE (top priority)
The author wants THEIR OWN wording preserved. Expand the author's bullets below
into flowing sentences with MINIMAL rewording — keep their exact terms, phrasings,
and word choices. Do NOT paraphrase into a generic academic voice; stay as close
to the author's language and register (direct, plain, slightly informal) as
possible while making it read as prose. Keep their words verbatim where you can:
"amplify the human biases in the pre-training data", "harm to minoritized",
"mech interp and rep engr", "context aware", "LATAM context", "Sovereignty",
"Open Weight", "Adoption of AI is limited in resource-poor settings", "potentially
more harmful", "regular people do not engineer their prompts", "Scaffolding",
"change-of-turn token", "preliminary". Do not add claims the author did not make.

## AUTHOR'S VERBATIM OUTLINE (mirror this wording when turning it into prose)
1. Introduction
- LLMs amplify the human biases in the pre-training data, causing harm to minoritized.
- There is a need for a wide-range of interventions, including mech interp and rep engr ones.
- The interventions need to be aware of context, otherwise not effective
- We tackle the problem of adapting intervention for LATAM context
- Until recently, there was no evaluation data for biases specific to Spanish, the language of Latam.
- This papers leverages the new available data (ref: SESGO) to characterize bias in Spanish concept through behavioral, geometric and distributional studies.
- The goal is to give insight so future interventions can be informed on how to operate in LATAM context.
- Our work attends to Latam context in two ways:
  * Focus on Model Size effect on Bias of Open Weight LLMs. Why Open Models?
    Sovereignty (ref: arXiv 2412.12004). Models need to be adapted to work well in
    Latam context, which usually happens thru open models. Why Model Size? Adoption
    of AI is limited in resource-poor settings. We need to understand if using a
    smaller model will mean a potentially more harmful one.
  * Focus on effect of Scaffolding. Recent work has shown that LLMs often need
    prompt scaffolding to properly adapt to context. Many regular people do not
    engineer their prompts, so we would like to see how the scaffold changes
    bias-related behavior and representation geometry.
- Our results are preliminary but suggest: Smaller models are more biased; Scale
  lowers uncertainty; Simple scaffolding could help de-bias models.
- Contributions: (1) Extension of SESGO evaluation to evaluate effects of model
  scale; (2) Behavioral and geometric analysis of prompt scaffold effects for
  debiasing; (3) CoT Dynamical case study of bias.
2. Related Work: "We extend this past work but applying in conjunction and to a new
  set of open weight models." 2.1 SESGO (what is SESGO; Figure 1 from SESGO;
  concisely explain with notation what the data task is; our extension: model
  scale). 2.2 CoT Dynamics (based on refs, we estimate the statistic and dynamics
  of CoT; use notation, be succinct). 2.3 Representational Geometry (PCA
  visualization; change-of-turn token as interesting site, refs).
3. Methods. 3.1 Parallel studies [SUCCINCT, move all technicality to appendix]:
  Model Scale (replicate SESGO baseline with multiple models; compare thinking vs
  non-thinking behavior); Stability (how stable answers are over labels and order);
  Scaffold effect (does Scaffolding improve SESGO performance); Dynamics (compare
  uncertainty in CoT between large and small model — NOTE: forking is the dynamics
  study; the cross-scale uncertainty is the divergence study which is reported in
  App. C); Geometry (PCA per layer through change-of-turn token position). + a
  TABLE [Study vs Models vs Purpose vs Methods References]. 3.2 Reproducibility
  [code and huggingface links].
4. Results — for EACH claim: "CLAIM #: ..." / "EVIDENCE: - ... + Figure" /
  "LIMITATION: - ..." / "NOTES: - ..". The three claims: Smaller models are more
  biased; Scale lowers uncertainty; Simple scaffolding could help de-bias models.
5. Discussion. 5.1 Limitations: Stability study indicated other studies should
  filter for unstable answers; Small number of scaffolds; Not resolution between
  small and large model; Dynamics only of one sample. 5.2 Future Work: Use more
  geometric visualization methods; See how interventions look in activation-space;
  Deepen scaffold analysis with many more; Steering.
6. Conclusion [WRITE SHORT].
