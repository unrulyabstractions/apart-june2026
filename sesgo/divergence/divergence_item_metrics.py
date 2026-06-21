"""Per-item extraction for the divergence study (sample_size == 0 excluded).

Pure numeric helpers shared by the panel modules and the driver: an item's
default entropy, its JS-deviation from the safe default, and per-provenance-axis
grouping. All reuse src.common.math so the metric definitions live in one place.
"""

from __future__ import annotations

from collections import defaultdict

from src.common.math import js_divergence, probs_to_logprobs, shannon_entropy
from src.datasets.sesgo_eval import SesgoDataset, SesgoSample
from .divergence_plot_styles import GOLD_UNKNOWN


def scored_samples(dataset: SesgoDataset) -> list[SesgoSample]:
    """Samples whose thinking readout is backed by >=1 parsed draw."""
    return [s for s in dataset.samples
            if s.thinking is not None and s.thinking.sample_size > 0]


def ambig_scored(samples: list[SesgoSample]) -> list[SesgoSample]:
    """The AMBIGUOUS subset — the items the safe-default framing applies to."""
    return [s for s in samples if s.context_condition == "ambig"]


def item_entropy(s: SesgoSample) -> float:
    """Shannon entropy (nats) of an item's default [t,o,u] mix."""
    return float(shannon_entropy(probs_to_logprobs(s.thinking.mean)))


def item_deviation(s: SesgoSample) -> float:
    """JS-divergence of an item's default mix from the safe default [0,0,1]."""
    return float(js_divergence(s.thinking.mean, GOLD_UNKNOWN))


def group_values(samples, axis: str, value_fn) -> dict[str, list[float]]:
    """Per-group list of `value_fn` values keyed by provenance `axis` (sorted)."""
    groups: dict[str, list[float]] = defaultdict(list)
    for s in samples:
        groups[str(getattr(s, axis))].append(value_fn(s))
    return dict(sorted(groups.items()))


def split_by_condition(samples) -> dict[str, list[SesgoSample]]:
    """Bucket samples into {'ambig': [...], 'disambig': [...]}."""
    out: dict[str, list[SesgoSample]] = {"ambig": [], "disambig": []}
    for s in samples:
        out.setdefault(s.context_condition, []).append(s)
    return out
