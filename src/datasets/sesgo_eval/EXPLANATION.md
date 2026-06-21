# SESGO Ambiguous-Bias Querying — Detailed Specification

## Purpose

Given a `SesgoPromptDataset` of self-describing `SesgoPromptSample`s, elicit the
model's answer **two ways** and store both as a 3-way distribution over the
answer roles (TARGET / OTHER / UNKNOWN). SESGO probes social bias in *ambiguous*
contexts where the correct answer is always UNKNOWN ("not enough information"):
any mass the model puts on TARGET or OTHER is bias. Storing both a fast
reasoning-free readout and a deliberated one lets downstream analysis compare a
model's instinct against its reasoned answer and slice both by every prompt axis.

## Inputs

Each `SesgoPromptSample` already carries everything needed to decode a reply:
`text`, `option_labels` (the 3 position markers), `position_labels` (position i →
the role shown there), `choice_prefix`, `gold_label` (= UNKNOWN), plus the
color-by axes (`question_id`, `bias_category`, `question_polarity`, `language`,
`scaffold_id`, `label_style`, `sample_idx`, `bbq` origin, `target_identity`,
`other_identity`). The querier copies all of them onto `SesgoSample` and never
re-derives any of this.

## The core remap: positions → meanings

The generator emits all 6 role↔position permutations, so the displayed slot a
model picks carries no role information by itself. `position_labels[i]` records
which role occupies displayed position `i`. Every level decodes through this
tuple, yielding a role distribution invariant to ordering — the mechanism that
defeats position bias.

## Level 1 — Non-thinking (teacher-forced softmax)

`TernaryChoiceRunner.choose3(text, choice_prefix or "Answer: ", option_labels)`
teacher-forces the three option markers past an auto-inserted empty
`<think></think>` block, so the model never reasons. The returned `TernaryChoice`
carries, per displayed POSITION, the conditional `logprobs` AND the raw `logits`
of each option token (read from the single shared predicting row — all three
forced continuations share the prefix, so that full-vocab row is identical and
its raw values are directly comparable). `SesgoNonThinking.from_ternary(choice,
position_labels)` scatters each position's (logprob, logit) into its canonical
role slot [TARGET, OTHER, UNKNOWN] and derives five length-3 vectors:

- `prob` — `normalize_log_probs(logprob)`, the 3-way renormalized softmax.
- `logprob` / `logit` — the role-ordered raw scores.
- `normalized_logit` — `logit_i - mean(logit)`. Mean-centered, NOT softmaxed,
  because `softmax(logit) == prob` already (logit and logprob differ only by the
  log-partition constant, which softmax cancels), so a softmax form is redundant;
  centering preserves the raw confidence scale while removing the row offset.
- `inv_ppl` — `exp(logprob)`, the option token's full-vocab probability mass
  (inverse single-token perplexity).

Plus `entropy`/`diversity` over `prob` and `predicted` (argmax, ties → UNKNOWN).

## Level 1.5 — Greedy-thinking (one deterministic reasoning decode)

`SesgoGreedyThinking` records a SINGLE `runner.generate(text, max_new_tokens,
temperature=0.0)` with NO prefilling — unlike the greedy NON-thinking decode it
does NOT prepend the skip-thinking block, so a reasoning model actually thinks
before answering. The post-`</think>` answer is parsed via the same
`parse_chosen_label` as the thinking draws into a `label` (`None` if unparseable),
plus a short `text`. It is the answer the model commits to when it reasons
greedily — distinct from the teacher-forced choose3, the greedy non-thinking
decode, and the sampled thinking distribution. Gated on `config.do_greedy_thinking`
(off by default; the baseline study turns it on). `predicted` is just the parsed
`label`, and `correct_greedy_thinking` runs it through the same per-condition
`is_correct`.

## Level 2 — Thinking (sampled reasoning)

Draw `n_thinking_samples` generations with `temperature > 0`. Parse each via
`parse_chosen_label`:

1. Take the answer after the last `</think>`. If `<think>` opened but never
   closed, return `None` (truncated mid-thought — no answer to mine).
2. Find which option the answer commits to — earliest hit across positions of:
   the full marker (`"b)"`), the bare letter/number (`"b"`) as a standalone
   token (word-boundary guarded so it doesn't match inside another word), or the
   option's authored text (lifted from the rendered `##options` lines, which the
   generator emits in position order). All case-insensitive.
3. Map the chosen position through `position_labels` → a `SesgoLabel`; `None` if
   undetectable.

Drop the `None`s and pass the surviving picks to `summarize_labels(labels)`,
which builds a `SesgoThinking`: per role, the one-hot indicator across draws; its
mean is the pick fraction (`mean`) and its honest POPULATION std is `std`
(== sqrt(p·(1-p)) for a Bernoulli, but computed directly). `sample_size` = #parsed.
When `sample_size == 0`, `mean`/`std` are zero vectors and `predicted_thinking`
returns `None`.

## Summary math (`src.common.math`)

- `predicted` = argmax of the role vector (`prob` non-thinking, `mean` thinking);
  a non-unique max → UNKNOWN (an indecisive readout is not a bias signal).
- Non-thinking `entropy`/`diversity` over `prob`: `probs_to_logprobs(prob)` then
  `shannon_entropy` / `q_diversity(..., 1.0)`, consistent with the codebase.
- Thinking `mean` reuses `aggregate(indicators, AggregationMethod.MEAN)`; `std`
  is the population std of those same indicators.

## Output

`SesgoQuerier.query_dataset` applies `config.subsample` as a deterministic
leading slice, iterates behind a `ProgressTracker`, clears accelerator memory
periodically, and returns a `SesgoDataset` (`prompt_dataset_id`, `model`,
`config`, `samples`). `SesgoSample` exposes `predicted_non_thinking`,
`predicted_thinking`, `predicted_greedy_thinking`, and `correct_non_thinking` /
`correct_thinking` / `correct_greedy_thinking` (= the prediction is UNKNOWN, the
ambiguous gold). Raw generations live in the private
`_thinking_completions` field, excluded from the id hash and `to_dict`, so
persisted datasets stay compact. Save with `SesgoDataset.save_as_json`; load with
the inherited `BaseSchema.from_json`.
