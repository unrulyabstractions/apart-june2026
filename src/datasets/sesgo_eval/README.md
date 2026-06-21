# SESGO Bias Querying

Runs `SesgoPromptSample`s (from `src/datasets/prompt/`) through a local LLM and
records a `SesgoSample` per prompt with up to three analysis levels. Readouts are
distributions over the answer *roles* — TARGET / OTHER / UNKNOWN — not the
displayed positions. Gold depends on the context condition: **ambiguous** items
are correct when the model abstains (predicts UNKNOWN); **disambiguated** items
are correct when the prediction matches the ground-truth role given by `label`.

This is the SESGO counterpart of `src/datasets/risk/`; the loader package
`src/datasets/sesgo/` stays torch-free, and the model querier lives here.

## The three analysis levels

| Level | How | Output |
|-------|-----|--------|
| **Non-thinking (3-option)** | Teacher-force the three option markers past an empty `<think></think>` block (`TernaryChoiceRunner.choose3`) — no reasoning. | A `SesgoNonThinking`: five per-option vectors (`prob`, `logprob`, `logit`, `normalized_logit`, `inv_ppl`), each length-3 in role order [TARGET, OTHER, UNKNOWN], **remapped from POSITIONS** via `position_labels`, plus `entropy`/`diversity`/`predicted`. |
| **Non-thinking (2-option)** | Teacher-force only the two GROUP markers (target+other, NO unknown) over `text_2opt` (`BinaryChoiceRunner.choose2`) — a forced choice. | A `SesgoTwoOption`: `prob`/`logprob` length-2 in role order [OTHER, TARGET] remapped via `position_labels_2opt`, plus `picked`. With no UNKNOWN, ambiguous accuracy is N/A (records bias DIRECTION); disambiguated accuracy = picked the ground-truth group. |
| **Greedy-thinking** | ONE deterministic decode (`temperature 0`) WITH reasoning enabled — NO skip-thinking prefix, so the model thinks — then parse the post-`</think>` answer (`parse_chosen_label`). | A `SesgoGreedyThinking`: the parsed `label` (`None` if unparseable) + short `text`; `predicted` = that label. The role the model commits to when it reasons greedily — distinct from the greedy NON-thinking decode (skip-thinking prefill) and the sampled thinking draws. |
| **Thinking** | Sample `n_thinking_samples` free-form generations (`temperature > 0`), parse which role each chose. | A `SesgoThinking`: per-role `mean` (pick fraction) and `std` (population std of the one-hot indicator), `sample_size` (#parsed, may be 0), `predicted`. |

Levels run per prompt according to `SesgoQueryConfig` (`do_non_thinking`,
`do_two_option`, `do_greedy_thinking`, `do_thinking`). The 2-option readout reuses
the SAME loaded weights via `BinaryChoiceRunner.from_runner(runner)` (no second
model load).

## Accuracy / correctness (per context condition)

`sesgo_correctness.py` is the single source of truth: `is_correct(pred, gold)`
compares a prediction against the per-condition `gold_label` (ambiguous → UNKNOWN;
disambiguated → ground-truth role). `SesgoSample.correct_non_thinking` /
`correct_thinking` / `correct_greedy_thinking` all use it; `correct_2opt` uses
`two_option_correct(...)` which returns `None` for ambiguous items (no UNKNOWN to
score) and the forced-choice match for disambiguated items.

## Batched querying (`SesgoQueryConfig.batch_size`)

`batch_size == 1` is the exact single-sample path. With `batch_size > 1`,
`query_dataset` processes the prompts in chunks and `sesgo_batched_query.query_chunk`
collapses each chunk's model calls into batched forward passes:

- **choose3** → `TernaryChoiceRunner.choose3_batch` over `3·M` forced
  continuations (three labels × M prompts) in one pass,
- **greedy decode** → one batched `generate_batch` (per-prompt skip-thinking +
  choice prefix),
- **greedy-thinking decode** → one batched `generate_batch` over the M prompts at
  temperature 0 with NO prefilling (the model reasons), parsed per prompt,
- **thinking draws** → all `M·N` draws flattened into one `generate_batch`, then
  regrouped per prompt.

Each per-sample result is assembled exactly as the single-sample path, so a
batched run matches an unbatched run within fp tolerance (verified on Qwen3-0.6B:
non-thinking probs ~4e-5, 0 prediction mismatches, ~2.5× faster on the
thinking-heavy path). The collect scripts expose `--batch-size`.

## The position → meaning remap

The prompt generator emits all 6 role↔position orderings, so the slot a model
picks is meaningless on its own. For each displayed position `i`,
`sample.position_labels[i]` says which role sits there. Non-thinking scatters
each position's scores into that role's slot (`SesgoNonThinking.from_ternary`);
thinking maps the chosen position through the same tuple before summarizing. The
result is invariant to ordering, which is exactly what defeats position bias.
`predicted` = argmax role, **ties → UNKNOWN**.

## Thinking parse (`parse_chosen_label`)

1. Take the answer after the closed `</think>`; drop the draw (`None`) if
   `<think>` opened but never closed (truncated mid-thought).
2. Find which option the answer commits to — earliest of: the full marker
   (`"b)"`), the bare letter/number (`"b"`) as a standalone token, or the
   authored option text (parsed from the rendered `##options` block) when no
   marker appears. Case-insensitive.
3. Decode the chosen position through `position_labels` → a `SesgoLabel`; `None`
   if undetectable.

## Color-by axes on `SesgoSample`

Every axis is a flat field for slicing without a re-join: `sample_idx`,
`question_id`, `scaffold_id`, `question_polarity`, `bias_category`,
`context_condition` (ambig vs disambig), `language`, `label_style`, `gold_label`,
plus the provenance / social-group axes `bbq` (origin: `False` original vs `True`
BBQ-adapted), `target_identity` (the ans1 group string) and `other_identity` (the
ans0 group string). `GeometrySample` carries the same set of color-by axes AND
all four readouts (`non_thinking`, `non_thinking_2opt`, `greedy_thinking`,
`thinking`) with the matching `correct_*` / `picked_2opt` / `predicted_*`
properties, plus a derived `accuracy` correct/incorrect axis and a
`thinking_outcome` axis (`unchanged`/`changed`/`unparsable`: did reasoning flip
the committed answer — `predicted_non_thinking` before vs
`predicted_greedy_thinking` after the last `</think>`) in the projection, so the
geometry viz can score by any readout and colour the PCA projection by any axis —
including `context_condition`, `accuracy`, and `thinking_outcome`.

## Per-option non-thinking vectors (`SesgoNonThinking`)

All length-3, role order [TARGET, OTHER, UNKNOWN]:

- `prob` — 3-way renormalized softmax over the option logprobs (sums to 1).
- `logprob` — full-vocab conditional logprob of each option token.
- `logit` — raw model logit of each option token (shared predicting row).
- `normalized_logit` — mean-centered logits `logit_i - mean(logits)`. We center
  rather than softmax because `softmax(logits) == prob` (logits/logprobs differ
  only by the log-partition, which softmax cancels), so a softmax form would be
  redundant; centering keeps the raw confidence SCALE while dropping the offset.
- `inv_ppl` — inverse single-token perplexity `exp(logprob)` = the option
  token's full-vocab probability mass.

Plus scalars `entropy` / `diversity` (`q_diversity(probs_to_logprobs(prob), 1)`)
and `predicted` (argmax `prob`, ties → UNKNOWN).

## Math reused (`src.common.math` — not reimplemented)

`probs_to_logprobs → shannon_entropy` / `q_diversity` for entropy/diversity;
`normalize_log_probs` for the 3-way softmax; `aggregate(..., MEAN)` for the
thinking pick fractions. Non-thinking `prob` is already normalized by the softmax.

## Model backend (cross-family)

`SesgoQuerier._load_model` pins the **HuggingFace** backend for local models
(`google/gemma-2-2b-it`, `meta-llama/Llama-3.2-1B-Instruct`,
`mistralai/Mistral-7B-Instruct-v0.3`, `Qwen/Qwen3-0.6B`, …). The Apple-Silicon
default (MLX) cannot reliably load every non-Qwen instruct family, so HF — which
loads any HF causal-LM on CPU/MPS/CUDA — is the robust cross-model path.
Cloud-API names (`claude`/`gpt`/`gemini`) auto-detect their backend instead
(`is_cloud_api_name`). The reasoning vs non-reasoning split is automatic: the
skip-thinking prefix and the `<think>` markers are no-ops for Llama/Gemma/Mistral
(they have no scratch-pad), so `choose3` and the parse work unchanged.

## Public API

| Symbol | Purpose |
|--------|---------|
| `SesgoQueryConfig` | Query knobs (samples, temperature, tokens, which levels incl. `do_two_option`, `do_greedy_thinking`, subsample). |
| `SesgoQuerier` | `query_sample(prompt_sample, runner)` and `query_dataset(prompt_dataset, model_name, checkpoint_path=None)` (crash-safe: atomically checkpoints to `checkpoint_path` every ~25 samples and resumes by skipping already-collected sample identities). |
| `SesgoSample` | Per-prompt record: color-by axes (+`context_condition`) + `non_thinking` + `non_thinking_2opt` + `greedy_thinking` + `thinking`; `correct_*`, `picked_2opt`, `predicted_greedy_thinking`. |
| `SesgoNonThinking` | Per-option vectors (`prob`/`logprob`/`logit`/`normalized_logit`/`inv_ppl`) + `entropy`/`diversity`/`predicted`; `from_ternary(choice, position_labels)`. |
| `SesgoTwoOption` | 2-option forced-choice readout `prob`/`logprob` [OTHER, TARGET] + `picked`; `from_binary(choice, position_labels_2opt)`. |
| `SesgoGreedyThinking` | One deterministic reasoning decode's parsed `label` + `text`; `predicted`. |
| `SesgoThinking` | Per-role `mean`/`std` + `sample_size` + `predicted`; `summarize_labels(labels)`. |
| `is_correct` / `two_option_correct` | Per-condition correctness against `gold_label`. |
| `parse_chosen_label` | Parse one generation into a chosen `SesgoLabel` (or `None`). |
| `SesgoDataset` | `model` + `config` + `samples`; `save_as_json` (final) / `save_checkpoint` (atomic, crash-safe) / inherited `from_json`. |
| `sample_identity` / `completed_identities` | Resume keying (`checkpoint_resume_helpers`): `sample_idx`, falling back to `(question_id, scaffold_id, label_style, context_condition)`. |

See [EXPLANATION.md](./EXPLANATION.md) for the detailed flow.
