# Apart — Global South AI Safety Hackathon (June 2026)

This repository merges two lines of work on **homogenization and bias in LLMs**:

1. **Normativity / bias estimation** (from `queering-nlp-bias`) — characterizing
   what a model treats as "normal" and measuring bias as a *loss of diversity*
   around normativity, via LLM-induced statistics of scoring functions.
2. **Geometric & interpretability analysis** (from `temporal-manifolds`) — the
   activation-extraction, intervention/patching, and residual-stream machinery
   used to probe the internal structure behind those behaviors.

The written hackathon report lives in [`paper/`](paper/) (single-column arXiv-style
template); submission guidelines are in [`submission_info/`](submission_info/).

> **Note**: The `.md` documentation files in this repo were largely LLM-generated.
> Take them with a grain of salt. If something seems wrong, unclear, or contradicts
> the code, trust the code and report it.

## Conceptual foundation

**Bias is low deviance in a dimension where we would expect more diversity** — an
overly high concentration around normativity. An LLM defines a probability
distribution over all possible text continuations. **Normativity** is what the
model treats as the default path — the expected value of compliance across
structures. **Bias** manifests when this distribution concentrates too heavily
around certain outcomes, erasing diversity in dimensions that matter.

We operationalize this through:

- **Structures**: questions about text (e.g. "Does this mention women?") that encode what we care about
- **Compliance**: how well a trajectory satisfies a structure (0 = no, 1 = yes)
- **Core** `⟨Λ_n⟩`: the expected compliance — what the model treats as "normal"
- **Orientation** `θ_n(x)`: how a trajectory differs from the core
- **Deviance** `∂_n(x)`: magnitude of deviation from normal (low `E[∂]` = homogenized, high = diverse)

## Quick demo

```bash
uv sync                                  # install deps (Python 3.12+)
cp .env.example .env                     # fill in API keys / HF_TOKEN, etc.

# Run the full normativity pipeline (generate → score → estimate)
uv run python scripts/run_full_experiment.py \
    trials/generation/example.json trials/scoring/example.json
```

This runs three stages:

1. **Generate** — sample branching text continuations from a small model (e.g. `Qwen/Qwen3-0.6B`)
2. **Score** — judge each trajectory against structures with a larger judge model
3. **Estimate** — compute normativity metrics (core, orientation, deviance)

Generation methods: **simple sampling**, **forking paths** (deviations from the
greedy path), and **seeking entropy** (expand at high-uncertainty positions).
Add `--method forking-paths` / `--method seeking-entropy`, or `--all` to compare.

Outputs go to `out/<method>/<gen_name>/...` (gitignored).

## Directory structure

```
apart/
├── paper/                      # Hackathon report (LaTeX, arXiv-style); build.sh -> build/main.pdf
├── submission_info/            # Hackathon template + guidelines (PDF)
│
├── scripts/                    # Pipeline + analysis scripts
│   ├── run_full_experiment.py  # Full pipeline orchestrator
│   ├── generate_trajectories.py
│   ├── score_trajectories.py
│   ├── estimate_normativity.py
│   └── schemas/                # Config schemas (generation, scoring, estimation)
│
├── trials/                     # Experiment configs (generation/, scoring/)
├── configs/                    # Scenario YAMLs + prompt-dataset JSON configs
│
├── src/                        # Core library (union of both forks)
│   ├── common/                 # Data structures (token tree/trajectory/forks), math, analysis
│   │   └── math/entropy_diversity/   # entropy, diversity, power means, divergences
│   ├── inference/              # Model runner + backends (HF, MLX, nnsight, TransformerLens,
│   │   └── interventions/      #   Anthropic/OpenAI/Gemini APIs) + intervention/patching support
│   ├── datasets/               # DataManager + prompt/preference/parametric datasets
│   ├── generation/             # Trajectory generation methods
│   ├── estimation/             # Normativity estimation pipeline
│   ├── scoring/                # Trajectory scoring / judging
│   ├── geometry/               # Activation extraction (config, data, utils)
│   ├── binary_choice/          # Binary-choice running + parsing
│   ├── dynamics/               # Trajectory-dynamics analysis
│   └── viz/                    # Tree / dynamics visualization
│
├── streams/                    # Research streams (stability, prediction, ...)
├── webapp/                     # Streamlit/FastAPI playground for investigation
├── submodules/                 # External datasets (winoqueer, more-of-the-same)
├── wanderings/                 # Exploratory notes / scratch
├── utils/                      # Thin CLI wrappers
└── tests/                      # pytest
```

## Further reading

- [GUIDE_TO_EXPERIMENT.md](GUIDE_TO_EXPERIMENT.md) — step-by-step guide to running experiments
- [MOTIVATION.md](MOTIVATION.md) — why diversity matters (from critical theory)
- [GENERATION.md](GENERATION.md) · [SCORING.md](SCORING.md) · [ESTIMATION.md](ESTIMATION.md) — per-stage methodology
- [TERMINOLOGY.md](TERMINOLOGY.md) — glossary
- [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) — research questions

## Provenance

This repo is a thoughtful merge of two predecessor repos by the same author:
`queering-nlp-bias` (the normativity/bias pipeline) and `temporal-manifolds`
(geometric activation analysis with intervention support). Where the two forks
had diverged copies of the same module, they were merged taking the best of
both; see the git history for per-file merge notes.
