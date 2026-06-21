"""Batched residual-geometry capture for a chunk of SESGO prompts.

Mirrors the single-sample ``capture_activations`` but collapses the per-prompt
greedy decode and per-prompt ``run_with_cache`` into two batched forward passes:

  1. ONE batched greedy decode (skip-thinking prefill) for every prompt,
  2. ONE left-padded ``run_with_cache_batch`` over the realized sequences.

``run_with_cache_batch`` returns each sample's cache already sliced back to its
real (unpadded) length, so the SAME ``find_positions`` / ``save_positions`` used
by the single-sample path apply per sample with no offset bookkeeping. The
captured geometry is therefore identical to the unbatched path within fp tolerance.
"""

from __future__ import annotations

from pathlib import Path

from src.datasets.prompt import SesgoPromptSample
from src.datasets.sesgo_eval import GeometryActivation
from src.ternary_choice import TernaryChoiceRunner
from .geometry_capture_helpers import (
    _GREEDY_TOKENS,
    _resid_filter,
    answer_start_for,
    find_positions,
    save_positions,
)


def capture_activations_batch(
    runner: TernaryChoiceRunner,
    prompts: list[SesgoPromptSample],
    layers: list[int],
    sample_dir: Path,
    rel_root: Path,
) -> tuple[list[list[GeometryActivation]], list[list[str]]]:
    """Snapshot residuals for a chunk along each prompt's greedy non-thinking path."""
    prefixes = [p.choice_prefix or "Answer: " for p in prompts]
    heads = [
        runner.apply_chat_template(p.text) + runner.skip_thinking_prefix + pre
        for p, pre in zip(prompts, prefixes)
    ]
    # One batched greedy decode for the whole chunk (each prompt's own prefill).
    greedies = runner.generate_batch(
        [p.text for p in prompts],
        max_new_tokens=_GREEDY_TOKENS,
        temperature=0.0,
        prefillings=[runner.skip_thinking_prefix + pre for pre in prefixes],
    )
    forced = [h + g for h, g in zip(heads, greedies)]
    ids_batch = [runner.encode_ids(f, add_special_tokens=True) for f in forced]
    answer_starts = [answer_start_for(runner, h, f) for h, f in zip(heads, forced)]

    # One left-padded forward pass capturing the residual stream for the chunk.
    caches = runner.run_with_cache_batch(ids_batch, names_filter=_resid_filter(layers))

    all_captured: list[list[GeometryActivation]] = []
    all_missing: list[list[str]] = []
    for prompt, ids, cache, ans in zip(prompts, ids_batch, caches, answer_starts):
        positions = find_positions(runner, ids, ans)
        captured, missing = save_positions(
            runner, ids, cache, positions, layers, prompt.sample_idx, sample_dir, rel_root
        )
        all_captured.append(captured)
        all_missing.append(missing)
    return all_captured, all_missing
