# ternary_choice

A 3-way (N-way) teacher-forced choice runner — the direct N=3 generalization
of `src/binary_choice/`. It scores which of **three** option labels a model
assigns the highest probability to, **without letting it reason**.

## The 3-way recipe

Given a `prompt`, a shared `choice_prefix`, and three `labels`:

1. `apply_chat_template(prompt)`.
2. Build `effective_prefix = skip_thinking_prefix + choice_prefix`. The
   `skip_thinking_prefix` auto-suppresses `<think>` for reasoning models, so
   the model never reasons before committing to a label.
3. For each of the three labels, construct the forced continuation
   `effective_prefix + label` and jointly encode `prompt + continuation` with
   the same `encode_into_trajectory_ids` helper used by the binary runner
   (preserves BPE merges and resolves BOS handling).
4. Run **one** batched `compute_trajectories_batch([ids0, ids1, ids2])` — three
   teacher-forced forward passes.
5. The three continuations share `effective_prefix`, so their token-id
   sequences are identical up to the **option-label token**. The divergence
   position is the first index where the three sequences differ; each label's
   `logprobs[div_pos]` is its conditional logprob `P(label | prompt+prefix)`.
6. A **3-way softmax** over those three logprobs gives a probability
   distribution over the options. Argmax = the chosen option (`-1` on a tie).

This is exactly the binary fork comparison (two logprobs at one divergence
point) extended to three: one softmax instead of one pairwise comparison.

## API

```python
from src.ternary_choice import TernaryChoiceRunner

runner = TernaryChoiceRunner("Qwen/Qwen3-0.6B")
choice = runner.choose3(
    prompt,
    "Answer: ",
    labels=("0", "1", "2"),
)

choice.probs         # (p0, p1, p2), sums to 1.0
choice.choice_idx    # 0 | 1 | 2 | -1 (tie)
choice.chosen_label  # e.g. "1", or None on a tie
choice.entropy       # Shannon entropy (nats) of the distribution
```

`TernaryChoice` (in `src/common/choice/ternary_choice.py`) is a clean
`BaseSchema`: it holds only `labels` and the three `logprobs`, so it
serializes and round-trips (`to_dict` / `from_dict`) cheaply.

## Labels

Labels should be **single-token-distinct** — e.g. `"0"/"1"/"2"` or
`"a"/"b"/"c"` — so that the three continuations diverge at exactly the
option-label token and the compared logprobs are directly meaningful.
