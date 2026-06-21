# SESGO Ambiguous-Bias Querying

Runs `SesgoPromptSample`s (from `src/datasets/prompt/`) through a local LLM and
records a `SesgoSample` per prompt with up to two analysis levels. Each level is
a 3-way distribution over the answer *roles* — TARGET / OTHER / UNKNOWN — not the
displayed positions. The ambiguous-context gold is always UNKNOWN, so a "correct"
sample is one where the model abstains rather than picking a group.

This is the SESGO counterpart of `src/datasets/risk/`; the loader package
`src/datasets/sesgo/` stays torch-free, and the model querier lives here.

## The two analysis levels

| Level | How | Output |
|-------|-----|--------|
| **Non-thinking** | Teacher-force the three option markers past an empty `<think></think>` block (`TernaryChoiceRunner.choose3`) — no reasoning. | A `SesgoNonThinking`: five per-option vectors (`prob`, `logprob`, `logit`, `normalized_logit`, `inv_ppl`), each length-3 in role order [TARGET, OTHER, UNKNOWN], **remapped from POSITIONS** via `position_labels`, plus `entropy`/`diversity`/`predicted`. |
| **Thinking** | Sample `n_thinking_samples` free-form generations (`temperature > 0`), parse which role each chose. | A `SesgoThinking`: per-role `mean` (pick fraction) and `std` (population std of the one-hot indicator), `sample_size` (#parsed, may be 0), `predicted`. |

Levels run per prompt according to `SesgoQueryConfig` (`do_non_thinking`,
`do_thinking`).

## Batched querying (`SesgoQueryConfig.batch_size`)

`batch_size == 1` is the exact single-sample path. With `batch_size > 1`,
`query_dataset` processes the prompts in chunks and `sesgo_batched_query.query_chunk`
collapses each chunk's model calls into batched forward passes:

- **choose3** → `TernaryChoiceRunner.choose3_batch` over `3·M` forced
  continuations (three labels × M prompts) in one pass,
- **greedy decode** → one batched `generate_batch` (per-prompt skip-thinking +
  choice prefix),
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
`question_id`, `scaffold_id`, `question_polarity`, `bias_category`, `language`,
`label_style`, `gold_label`.

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
| `SesgoQueryConfig` | Query knobs (samples, temperature, tokens, which levels, subsample). |
| `SesgoQuerier` | `query_sample(prompt_sample, runner)` and `query_dataset(prompt_dataset, model_name)`. |
| `SesgoSample` | Per-prompt record: color-by axes + `non_thinking` + `thinking`. |
| `SesgoNonThinking` | Per-option vectors (`prob`/`logprob`/`logit`/`normalized_logit`/`inv_ppl`) + `entropy`/`diversity`/`predicted`; `from_ternary(choice, position_labels)`. |
| `SesgoThinking` | Per-role `mean`/`std` + `sample_size` + `predicted`; `summarize_labels(labels)`. |
| `parse_chosen_label` | Parse one generation into a chosen `SesgoLabel` (or `None`). |
| `SesgoDataset` | `model` + `config` + `samples`; `save_as_json` / inherited `from_json`. |

See [EXPLANATION.md](./EXPLANATION.md) for the detailed flow.
