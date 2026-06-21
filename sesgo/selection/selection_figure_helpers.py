"""Assemble the SELECTION study's three plain-language abstention figures.

This study probes ONE thing on ambiguous SESGO items (where the only correct answer
is "unknown"): how often does the model correctly abstain? Higher abstention = better,
because guessing a social group on an ambiguous question is exactly the biased move.
Three figures read this from complementary angles, all in plain English:

  abstention_by_scaffold.png          - headline: abstention by how the model answered
  abstention_by_scaffold_category.png - abstention split by bias category
  accuracy_by_scaffold_ambig.png      - per-readout panels, the full small-n picture

Every bar carries a Wilson 95% CI and its sample size; the tiny n is shown honestly so
no figure implies more statistical power than ~35 ambiguous items actually give.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from sesgo.common.plain_language_labels import (
    CATEGORY_LABEL,
    CATEGORY_ORDER,
    READOUT_LABEL,
)
from selection_metrics_helpers import (
    counts_by_category,
    counts_by_level,
    readout_title,
)
from selection_plot_helpers import (
    SERIES_COLOR,
    figure_titles,
    panel,
    save_figure,
)

# Random guessing on a three-way (target / other / unknown) ambiguous question.
_CHANCE_AMBIG = 1 / 3
_ABSTAIN_AXIS = "Abstention rate\n(answers 'unknown')"
_HOW_TO_READ_BETTER = "Each bar = the share of ambiguous items the model abstained on (higher is better). Whiskers = 95% CIs; numbers = sample size."


def _present_categories(dataset, level: str) -> list[str]:
    """Bias categories that actually have data, in the stable display order."""
    counts = counts_by_category(dataset, level, "ambig", list(CATEGORY_ORDER))
    return [c for c in CATEGORY_ORDER if c in counts and counts[c].total > 0]


def figure_abstention_by_readout(dataset, model: str, out_path: Path) -> Path:
    """Headline: abstention rate by how the model answered (one bar per readout)."""
    levels = [lvl for lvl in ("non_thinking", "thinking")]
    counts = counts_by_level(dataset, levels, "ambig")
    drawn = [lvl for lvl in levels if lvl in counts]
    labels = [READOUT_LABEL[lvl] for lvl in drawn]
    colors = [SERIES_COLOR[lvl] for lvl in drawn]
    indexed = {READOUT_LABEL[lvl]: counts[lvl] for lvl in drawn}

    fig, ax = plt.subplots(figsize=(max(6.5, 2.6 * len(drawn) + 1.5), 5.4),
                           layout="constrained")
    panel(ax, labels, indexed, colors, _ABSTAIN_AXIS, _CHANCE_AMBIG)
    figure_titles(
        fig,
        f"How often the model refuses to guess on ambiguous questions  ·  {model}",
        _HOW_TO_READ_BETTER,
    )
    return save_figure(fig, out_path)


def figure_abstention_by_category(dataset, model: str, out_path: Path) -> Path:
    """Abstention rate split by bias category, at the direct-answer readout."""
    cats = _present_categories(dataset, "non_thinking")
    counts = counts_by_category(dataset, "non_thinking", "ambig", cats)
    labels = [CATEGORY_LABEL[c] for c in cats]
    indexed = {CATEGORY_LABEL[c]: counts[c] for c in cats}
    colors = [SERIES_COLOR["category"]] * len(cats)

    fig, ax = plt.subplots(figsize=(max(6.5, 2.0 * len(cats) + 1.5), 5.4),
                           layout="constrained")
    panel(ax, labels, indexed, colors, _ABSTAIN_AXIS, _CHANCE_AMBIG,
          badge=READOUT_LABEL["non_thinking"])
    figure_titles(
        fig,
        f"Does the model guess more on some kinds of bias?  ·  {model}",
        _HOW_TO_READ_BETTER,
    )
    return save_figure(fig, out_path)


def figure_abstention_panels(dataset, model: str, out_path: Path) -> Path:
    """Per-readout panels (the full small-n picture), one row per answering style."""
    levels = [lvl for lvl in ("non_thinking", "thinking")]
    counts = counts_by_level(dataset, levels, "ambig")
    drawn = [lvl for lvl in levels if lvl in counts] or ["non_thinking"]

    fig, axes = plt.subplots(len(drawn), 1, figsize=(6.8, 3.3 * len(drawn) + 0.7),
                             layout="constrained", squeeze=False)
    for ax, lvl in zip(axes[:, 0], drawn):
        label = READOUT_LABEL[lvl]
        indexed = {label: counts[lvl]} if lvl in counts else {label: None}
        # Badge carries the short readout name + one-line gloss as the panel's
        # plain caption; the x-tick would just repeat it, so it is hidden.
        panel(ax, [label], indexed, [SERIES_COLOR[lvl]], _ABSTAIN_AXIS,
              _CHANCE_AMBIG, badge=readout_title(lvl), show_xticklabels=False,
              show_legend=(ax is axes[0, 0]))
    figure_titles(
        fig,
        f"Abstention on ambiguous questions, by answering style  ·  {model}",
        _HOW_TO_READ_BETTER,
    )
    return save_figure(fig, out_path)
