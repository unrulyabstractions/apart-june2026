# Risk-Assessment Querying — Detailed Specification

## Purpose

Given a `RiskPromptDataset` of self-describing `RiskPromptSample`s, elicit a risk
judgement from a local model **two ways** and store both, so downstream analysis
can compare a model's fast, reasoning-free instinct against its deliberated
estimate — and measure how much the latter wobbles across samples.

## Inputs

Each `RiskPromptSample` already carries everything needed to interpret a reply:
`text`, `task_type` (`SCORE`/`CATEGORIZE`), `labels`, `positive_idx`,
`choice_prefix`, `scale_high`, plus provenance (`subject_id`, `disorder`,
`gold_risk`, `framing`, `language`, `sample_idx`). The querier never re-derives
any of this.

## Level 1 — Non-thinking (calibrated probability) — CATEGORIZE only

`BinaryChoiceRunner.choose` teacher-forces two continuations (one per answer
option) past an auto-inserted empty `<think></think>` block, so the model never
reasons. From the divergent logprobs `lp_pos, lp_neg` (ordered so `pos` is the
at-risk option) we form a numerically-stable two-way softmax:

```
m = max(lp_pos, lp_neg)
p = exp(lp_pos - m) / (exp(lp_pos - m) + exp(lp_neg - m))   # predicted_risk
```

This level applies **only to CATEGORIZE** prompts, whose `labels` and at-risk
index (`sample.positive_idx`) are well-defined binary anchors. SCORE prompts ask
for a free number whose first token (e.g. the `0` of `0.75`) is uninformative as
a binary anchor, so `query_sample` skips non-thinking for them — SCORE is covered
by the thinking level alone.

We also keep `choice.choice_idx` (0/1/-1) and the raw labels/logprobs in
`NonThinkingResult` for auditing.

## Level 2 — Thinking (sampled reasoning)

Draw `n_thinking_samples` generations with `temperature > 0` (natural thinking).
Parse each via `parse_risk_score`:

1. Strip the thinking block (`strip_thinking_content`; the answer follows the
   last `</think>`).
2. **SCORE**: first number in `[0,1]` (regex); invert (`1 - x`) when
   `scale_high is False`. `None` if no parseable number.
3. **CATEGORIZE**: find the earliest-mentioned label token/phrase
   (case-insensitive); the at-risk option → `1.0`, the other → `0.0`; `None` if
   neither appears.

Drop `None`s, then `summarize_scores` collapses the survivors into a
`ScoreSummary`.

## Summary math (all from `src.common.math`)

- **mean** = `aggregate(scores, AggregationMethod.MEAN)`.
- **std** = `sqrt(max(E[x²] − E[x]², 0))`, computing `E[x²]` by reusing
  `aggregate` on the squared scores (the same identity `deviance_variance` uses).
- **entropy / diversity** over the score *distribution*:
  `probs = normalize([max(s,0) for s in scores])` (needs non-negative, ≥1
  element), `lp = probs_to_logprobs(probs)`, then `shannon_entropy(lp)` and
  `q_diversity(lp, 1.0)`. Singleton/empty inputs short-circuit to degenerate
  values so the maths never sees a malformed distribution.

## Output

`RiskQuerier.query_dataset` iterates samples behind a `ProgressTracker`,
clearing accelerator memory periodically, and returns a `RiskDataset`
(`prompt_dataset_id`, `model`, `config`, `samples`). `RiskAssessmentSample`
exposes convenience props `predicted_risk_non_thinking` and
`predicted_risk_thinking` (= `thinking.mean`). The raw generations live in the
private `_thinking_completions` field, excluded from the id hash and `to_dict`,
so persisted datasets stay compact. Save with `RiskDataset.save_as_json`; load
with the inherited `BaseSchema.from_json`.
