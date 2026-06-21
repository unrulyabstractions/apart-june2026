"""Read teacher-forced option-token logprobs out of a vLLM batched call.

The SESGO non-thinking readout teacher-forces three option labels onto a shared
prompt+prefix and reads each label token's conditional logprob. vLLM returns
``prompt_logprobs`` — the per-position logprob of each prompt token under the
model — so for a forced text ending in the option token we read the logprob at
the LAST prompt position. One ``llm.generate`` call (max_tokens=1) scores the
whole batch; we never need to generate, only to score the given continuations.
"""

from __future__ import annotations

from typing import Any


def option_token_logprobs(
    llm: Any,
    sampling_params_cls: Any,
    forced_texts: list[str],
    option_token_ids: list[int],
) -> list[float]:
    """Conditional logprob of each forced text's final (option) token.

    Args:
        llm: a constructed ``vllm.LLM`` engine.
        sampling_params_cls: ``vllm.SamplingParams`` (injected so this stays
            import-light and unit-testable off a GPU box).
        forced_texts: prompt+prefix+option_i, fully rendered, one per option.
        option_token_ids: the token id whose logprob to read for each forced text
            (its last prompt token).

    Returns:
        One conditional logprob per forced text, in input order.
    """
    if not forced_texts:
        return []
    # max_tokens=1 so vLLM still runs prefill (needed for prompt_logprobs) without
    # spending decode budget; prompt_logprobs=1 returns each position's top logprob
    # plus the actual token's logprob, which is what we read.
    params = sampling_params_cls(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    outputs = llm.generate(forced_texts, params, use_tqdm=False)

    logprobs: list[float] = []
    for output, tok_id in zip(outputs, option_token_ids):
        # prompt_logprobs[i] maps token-id -> Logprob for the i-th prompt token;
        # the option token is the LAST prompt token, so read position -1.
        last = output.prompt_logprobs[-1]
        entry = last.get(tok_id) if last else None
        logprobs.append(float(entry.logprob) if entry is not None else float("-inf"))
    return logprobs
