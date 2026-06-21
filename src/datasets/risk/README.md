# Risk-Assessment Querying

Runs `RiskPromptSample`s (from `src/datasets/prompt/`) through a local LLM and
records a `RiskAssessmentSample` per prompt with up to two analysis levels.

## The two analysis levels

| Level | Applies to | How | Output |
|-------|-----------|-----|--------|
| **Non-thinking** | CATEGORIZE only | Teacher-force both answer options past an empty `<think></think>` block (`BinaryChoiceRunner.choose`) — no reasoning. | A calibrated `P(at risk)` from a 2-way softmax over the two divergent logprobs. |
| **Thinking** | CATEGORIZE + SCORE | Sample `n_thinking_samples` free-form generations (`temperature > 0`), parse a risk score from each, summarize the cloud. | Mean / std / entropy / diversity / min / max over the parsed scores. |

Levels run per prompt according to `RiskQueryConfig` (`do_non_thinking`,
`do_thinking`), but non-thinking is skipped for SCORE prompts — a free numeric
answer has no binary anchor — so SCORE prompts carry the thinking level only.

## Recipes

**Non-thinking (CATEGORIZE only).**
`choice = runner.choose(text, choice_prefix or "Answer: ", labels)`;
`p = softmax(lp_pos, lp_neg)` ordered so `pos` is the at-risk option. The labels
and at-risk index come straight from the prompt (`labels`, `positive_idx`).

**Thinking.** For each generation we strip the thinking block, then: SCORE
extracts the first number in `[0,1]` (inverting when `scale_high is False`);
CATEGORIZE detects which label/phrase the model chose and maps the at-risk
option to `1.0`, the other to `0.0`. Unparseable draws return `None` and are
dropped before summarizing.

## Math reused (`src.common.math` — not reimplemented)

- mean: `aggregate(scores, AggregationMethod.MEAN)`
- std: `sqrt(max(E[x²] − E[x]², 0))` via `aggregate` on squared scores (mirrors `deviance_variance`)
- distribution spread: `normalize` → `probs_to_logprobs` → `shannon_entropy` + `q_diversity`

## Public API

| Symbol | Purpose |
|--------|---------|
| `RiskQueryConfig` | Query knobs (samples, temperature, tokens, which levels, subsample). |
| `RiskQuerier` | `query_sample(prompt_sample, runner)` and `query_dataset(prompt_dataset, model_name)`. |
| `RiskAssessmentSample` | Per-prompt record: provenance + `non_thinking` + `thinking`. |
| `NonThinkingResult` | Calibrated `predicted_risk` plus the raw logprob evidence. |
| `ScoreSummary` / `summarize_scores` | Aggregate of sampled thinking scores. |
| `parse_risk_score` | Parse one generation into an oriented `[0,1]` score (or `None`). |
| `RiskDataset` | `model` + `config` + `samples`; `save_as_json` / inherited `from_json`. |

See [EXPLANATION.md](./EXPLANATION.md) for the detailed flow.
