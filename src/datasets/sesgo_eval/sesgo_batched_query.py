"""Batched assembly of SesgoSamples for a chunk of prompts.

The single-sample querier issues, per prompt, three separate model calls (a
teacher-forced choose3, a greedy non-thinking decode, and N thinking draws). For
a chunk of M prompts each of those collapses into ONE batched forward:

  * choose3      -> ``choose3_batch`` over 3M continuations,
  * greedy decode-> ``generate_batch`` over M prompts (per-sample prefill),
  * thinking     -> ``generate_batch`` over M*N draws (flattened, then regrouped).

Every per-sample result is then assembled exactly as the single-sample path does,
so a batched run matches an unbatched run within fp tolerance. This module owns
only the batched orchestration; the scoring/parsing it calls is the shared,
already-tested single-sample logic.
"""

from __future__ import annotations

from src.datasets.prompt import SesgoPromptSample
from src.ternary_choice import TernaryChoiceRunner
from .sesgo_non_thinking import SesgoNonThinking
from .sesgo_query_config import SesgoQueryConfig
from .sesgo_response_parsing import parse_chosen_label
from .sesgo_sample import SesgoSample
from .sesgo_thinking import SesgoThinking, summarize_labels

# choose3's default response prefix when a prompt doesn't carry its own.
_DEFAULT_PREFIX = "Answer: "
# Greedy non-thinking decode stays short — we only need the committed label.
_GREEDY_TOKENS = 24


def _prefix(sample: SesgoPromptSample) -> str:
    """The prompt's choice prefix, falling back to the shared default."""
    return sample.choice_prefix or _DEFAULT_PREFIX


def _batch_non_thinking(
    samples: list[SesgoPromptSample],
    runner: TernaryChoiceRunner,
    config: SesgoQueryConfig,
) -> list[SesgoNonThinking]:
    """choose3 (always) + greedy decode (optional) for the whole chunk, batched."""
    prefixes = [_prefix(s) for s in samples]
    choices = runner.choose3_batch(
        [s.text for s in samples], prefixes, [s.option_labels for s in samples]
    )
    nts = [
        SesgoNonThinking.from_ternary(choice, s.position_labels)
        for choice, s in zip(choices, samples)
    ]
    if not config.do_greedy:
        return nts

    # One batched greedy decode; each prompt keeps its own skip-thinking + prefix.
    prefillings = [runner.skip_thinking_prefix + p for p in prefixes]
    greedies = runner.generate_batch(
        [s.text for s in samples],
        max_new_tokens=_GREEDY_TOKENS,
        temperature=0.0,
        prefillings=prefillings,
    )
    for nt, greedy, sample in zip(nts, greedies, samples):
        nt.greedy_text = greedy.strip()[:200]
        nt.greedy_label = parse_chosen_label(greedy, sample)
        nt.decoding_mismatch = (
            nt.greedy_label is not None and nt.greedy_label != nt.predicted
        )
    return nts


def _batch_thinking(
    samples: list[SesgoPromptSample],
    runner: TernaryChoiceRunner,
    config: SesgoQueryConfig,
) -> tuple[list[SesgoThinking], list[list[str]]]:
    """N free-form draws per prompt, all flattened into ONE batched decode."""
    n = config.n_thinking_samples
    flat_prompts = [s.text for s in samples for _ in range(n)]
    flat = runner.generate_batch(
        flat_prompts,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
    )
    summaries: list[SesgoThinking] = []
    completions: list[list[str]] = []
    for i, sample in enumerate(samples):
        draws = flat[i * n : (i + 1) * n]
        labels = [
            label
            for label in (parse_chosen_label(c, sample) for c in draws)
            if label is not None
        ]
        summaries.append(summarize_labels(labels))
        completions.append(draws)
    return summaries, completions


def query_chunk(
    samples: list[SesgoPromptSample],
    runner: TernaryChoiceRunner,
    config: SesgoQueryConfig,
) -> list[SesgoSample]:
    """Assemble SesgoSamples for a chunk via batched non-thinking + thinking."""
    nts = (
        _batch_non_thinking(samples, runner, config)
        if config.do_non_thinking
        else [None] * len(samples)
    )
    if config.do_thinking:
        thinkings, completions = _batch_thinking(samples, runner, config)
    else:
        thinkings = [None] * len(samples)
        completions = [None] * len(samples)

    return [
        SesgoSample(
            sample_idx=s.sample_idx,
            question_id=s.question_id,
            scaffold_id=s.scaffold_id,
            question_polarity=s.question_polarity,
            bias_category=s.bias_category,
            language=s.language,
            label_style=s.label_style,
            gold_label=s.gold_label,
            prompt_text=s.text,
            non_thinking=nt,
            thinking=th,
            _thinking_completions=comp,
        )
        for s, nt, th, comp in zip(samples, nts, thinkings, completions)
    ]
