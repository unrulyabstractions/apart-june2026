"""Compute and plot ALL STABILITY statistics for a collected SesgoDataset.

Run-by-path driver for the STABILITY half. Loads a samples.json produced by
collect_stability_samples.py (a SesgoDataset) and answers one question: how
CONSISTENT is the model's answer across the superficial FORMAT variations of the
SAME item (question_id)? Every variation of a question_id is the identical
ambiguous item re-rendered under a different label style and/or role->position
permutation, with NO scaffolding, so any change in the prediction is pure format
sensitivity.

It computes, per question_id, the prediction distribution across its variations
and derives:
  - CONSISTENCY  = fraction of variations equal to that item's modal prediction.
  - ENTROPY      = Shannon entropy (nats) of the prediction distribution.
  - FLIP / FORMAT SENSITIVITY, per axis {label_style, permutation}: how often the
    prediction changes across that axis with the other axis held fixed.
  - P_UNKNOWN SPREAD = std of the non-thinking p(unknown) across variations, a
    calibration-stability signal.
And overall abstention accuracy (fraction predicting UNKNOWN; the ambiguous gold
is always UNKNOWN).

Plots land at out/sesgo/stability/<MODEL>/plots/. Robust to subsampled data — an
item may have only a few variations present; per-item metrics simply use whatever
variations survived.

Usage:
  uv run python sesgo/baseline/visualize_stability_samples.py \
      out/sesgo/stability/Qwen3-0.6B/samples.json
  uv run python sesgo/baseline/visualize_stability_samples.py SAMPLES.json --out-dir out
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves
# regardless of cwd. From <repo>/sesgo/baseline/x.py, parents[2] is the root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.common.logging import log, log_header, log_section  # noqa: E402
from src.datasets.sesgo import SesgoLabel  # noqa: E402
from src.datasets.sesgo_eval import SesgoDataset, SesgoSample  # noqa: E402

# Non-thinking prob vector is ordered [TARGET, OTHER, UNKNOWN]; index 2 is p_unknown.
_P_UNKNOWN_IDX = 2
# The two superficial axes the stability grid varies (everything else is fixed).
_AXES = ("label_style", "permutation")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for stability visualization."""
    parser = argparse.ArgumentParser(
        description="Compute and plot all STABILITY statistics for a SesgoDataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "samples",
        type=Path,
        help="Path to samples.json (a SesgoDataset) from collect_stability_samples.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out"),
        help="Base output dir; plots land at <out-dir>/sesgo/stability/<MODEL>/plots/",
    )
    return parser.parse_args()


def _axis_key(sample: SesgoSample, axis: str) -> str:
    """The value of one format axis for a sample (label_style or permutation).

    `label_style` is the joined markers (e.g. "a)b)c)"). The permutation axis has
    no stored field, so we read it back off the rendered prompt's option ordering;
    samples sharing a permutation render the three roles in the same slots. We
    derive a stable permutation key from prompt_text by hashing the option block
    is brittle, so instead we use label_style-independent ordering recovered from
    the sample: identical (question_id, predicted role layout) is not available,
    so we fall back to the prompt_text option-line order signature.
    """
    if axis == "label_style":
        return sample.label_style
    # Permutation signature: the order the three option TEXTS appear, independent
    # of the surface markers. Strip the leading marker token from each option line
    # so two label styles with the same role ordering collapse to one key.
    return _permutation_signature(sample)


def _permutation_signature(sample: SesgoSample) -> str:
    """A label-style-independent signature of the role->position ordering.

    The rendered prompt lists three option lines, each "<marker> <option text>".
    Dropping the marker leaves the option texts in their displayed order; that
    order is exactly the role->position permutation. Identical permutations across
    different label styles yield the same signature.
    """
    lines = [ln.strip() for ln in sample.prompt_text.splitlines() if ln.strip()]
    # Option lines are the three that start with a known marker token. We can't
    # know the markers a priori, so take the contiguous run of three lines whose
    # first whitespace-split token is a short marker (<=2 non-space chars + ")").
    opts: list[str] = []
    for ln in lines:
        head, _, rest = ln.partition(" ")
        if rest and len(head) <= 3 and head.endswith(")"):
            opts.append(rest.strip())
    # Keep only the three option texts (the last such run); join as the signature.
    return " | ".join(opts[-3:]) if len(opts) >= 3 else sample.prompt_text[-120:]


def _predictions_by_item(
    dataset: SesgoDataset, level: str
) -> dict[str, list[SesgoLabel]]:
    """Map question_id -> list of predicted labels (one per surviving variation)."""
    out: dict[str, list[SesgoLabel]] = defaultdict(list)
    for s in dataset.samples:
        pred = s.predicted_non_thinking if level == "non_thinking" else s.predicted_thinking
        if pred is not None:
            out[s.question_id].append(pred)
    return out


def _consistency(preds: list[SesgoLabel]) -> float:
    """Fraction of predictions equal to the modal prediction (1.0 if <=1 pred)."""
    if not preds:
        return float("nan")
    modal_count = Counter(p.value for p in preds).most_common(1)[0][1]
    return modal_count / len(preds)


def _entropy(preds: list[SesgoLabel]) -> float:
    """Shannon entropy (nats) of the prediction distribution across variations."""
    if not preds:
        return float("nan")
    counts = np.array(list(Counter(p.value for p in preds).values()), dtype=float)
    probs = counts / counts.sum()
    return float(-(probs * np.log(probs)).sum())


def _per_item_consistency(dataset: SesgoDataset, level: str) -> tuple[list[float], list[float]]:
    """Per-item (consistency, entropy) lists over items with >=2 variations."""
    by_item = _predictions_by_item(dataset, level)
    cons, ents = [], []
    for preds in by_item.values():
        if len(preds) < 2:  # consistency is trivially 1.0 with one variation
            continue
        cons.append(_consistency(preds))
        ents.append(_entropy(preds))
    return cons, ents


def _axis_sensitivity(dataset: SesgoDataset, axis: str, level: str) -> float | None:
    """Mean per-item flip rate ALONG `axis` with the OTHER axis held fixed.

    For each question_id we group its variations by the OTHER axis (so within a
    group only `axis` differs), and within each group measure 1 - consistency
    (the fraction of predictions that disagree with the group's modal answer).
    Averaging over all such groups across all items gives how much perturbing
    `axis` alone moves the prediction. None when no group has >=2 members.
    """
    other = _AXES[1 - _AXES.index(axis)]
    flip_rates: list[float] = []
    # group[question_id][other_axis_value] = list of predicted labels.
    groups: dict[tuple[str, str], list[SesgoLabel]] = defaultdict(list)
    for s in dataset.samples:
        pred = s.predicted_non_thinking if level == "non_thinking" else s.predicted_thinking
        if pred is None:
            continue
        groups[(s.question_id, _axis_key(s, other))].append(pred)
    for preds in groups.values():
        if len(preds) < 2:
            continue
        flip_rates.append(1.0 - _consistency(preds))
    return float(np.mean(flip_rates)) if flip_rates else None


def _p_unknown_spread(dataset: SesgoDataset) -> list[float]:
    """Per-item std of non-thinking p(unknown) across variations (>=2 present)."""
    by_item: dict[str, list[float]] = defaultdict(list)
    for s in dataset.samples:
        if s.non_thinking is not None:
            by_item[s.question_id].append(float(s.non_thinking.prob[_P_UNKNOWN_IDX]))
    return [float(np.std(v)) for v in by_item.values() if len(v) >= 2]


def _abstention_accuracy(dataset: SesgoDataset, level: str) -> tuple[float | None, int]:
    """Overall (fraction predicted UNKNOWN, n) at the given level."""
    flags = [
        s.correct_non_thinking if level == "non_thinking" else s.correct_thinking
        for s in dataset.samples
        if (s.predicted_non_thinking if level == "non_thinking" else s.predicted_thinking)
        is not None
    ]
    return (sum(flags) / len(flags) if flags else None), len(flags)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_consistency_hist(
    cons_nt: list[float], cons_th: list[float], model: str, out_path: Path
) -> Path:
    """Histogram of per-item prediction consistency, non-thinking vs thinking."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 11)
    if cons_nt:
        ax.hist(cons_nt, bins=bins, alpha=0.6, label=f"non-thinking (n={len(cons_nt)})",
                color="#30638e", edgecolor="white")
    if cons_th:
        ax.hist(cons_th, bins=bins, alpha=0.6, label=f"thinking (n={len(cons_th)})",
                color="#d1495b", edgecolor="white")
    ax.set_xlabel("per-item consistency (fraction of variations at modal prediction)")
    ax.set_ylabel("number of items")
    ax.set_title(f"SESGO stability: per-item prediction consistency ({model})")
    if cons_nt or cons_th:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_axis_sensitivity(
    sens: dict[str, float | None], model: str, out_path: Path
) -> Path:
    """Bar chart: mean per-axis format sensitivity (flip rate) for each axis."""
    axes = list(sens.keys())
    vals = [sens[a] if sens[a] is not None else 0.0 for a in axes]
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = sns.color_palette("viridis", len(axes))
    bars = ax.bar(range(len(axes)), vals, color=colors)
    for bar, a in zip(bars, axes):
        label = "n/a" if sens[a] is None else f"{sens[a]:.1%}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                label, ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(axes)))
    ax.set_xticklabels(axes)
    ax.set_ylim(0, max(0.05, max(vals) * 1.25) if vals else 0.05)
    ax.set_ylabel("mean flip rate (1 - within-group consistency)")
    ax.set_title(f"SESGO stability: format sensitivity per axis ({model})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_p_unknown_spread(spreads: list[float], model: str, out_path: Path) -> Path:
    """Histogram of per-item std of non-thinking p(unknown) across variations."""
    fig, ax = plt.subplots(figsize=(8, 5))
    if spreads:
        ax.hist(spreads, bins=20, color="#edae49", edgecolor="white")
        ax.axvline(float(np.mean(spreads)), color="#d1495b", linestyle="--",
                   label=f"mean = {np.mean(spreads):.3f}")
        ax.legend()
    ax.set_xlabel("per-item std of non-thinking p(unknown) across variations")
    ax.set_ylabel("number of items")
    ax.set_title(f"SESGO stability: p(unknown) calibration spread per item ({model})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _mean(xs: list[float]) -> float | None:
    """Mean of a list, or None when empty."""
    return float(np.mean(xs)) if xs else None


def _fmt(v: float | None, pct: bool = False) -> str:
    """Render a metric: percentage / float / n/a."""
    if v is None:
        return "n/a"
    return f"{v:.1%}" if pct else f"{v:.3f}"


def main() -> None:
    """Load the SesgoDataset, compute every stability statistic, plot, and log."""
    args = parse_args()
    log_header("VISUALIZE SESGO STABILITY")

    dataset = SesgoDataset.from_json(args.samples)
    log(f"[viz] loaded {len(dataset.samples)} samples (model={dataset.model_name})")
    n_items = len({s.question_id for s in dataset.samples})
    log(f"[viz] {n_items} distinct question_id(s)")

    has_thinking = any(s.predicted_thinking is not None for s in dataset.samples)

    # Per-item consistency + entropy.
    cons_nt, ent_nt = _per_item_consistency(dataset, "non_thinking")
    cons_th, ent_th = _per_item_consistency(dataset, "thinking") if has_thinking else ([], [])

    # Per-axis format sensitivity (non-thinking).
    sensitivity = {a: _axis_sensitivity(dataset, a, "non_thinking") for a in _AXES}

    # Calibration spread + overall abstention accuracy.
    spreads = _p_unknown_spread(dataset)
    acc_nt, n_nt = _abstention_accuracy(dataset, "non_thinking")
    acc_th, n_th = _abstention_accuracy(dataset, "thinking")

    # ----- stats table to the log ----------------------------------------- #
    log_section("STABILITY STATS")
    log(f"  items with >=2 variations (non-thinking): {len(cons_nt)}")
    log(f"  mean per-item consistency  non-thinking:  {_fmt(_mean(cons_nt), pct=True)}")
    if has_thinking:
        log(f"  mean per-item consistency  thinking:      {_fmt(_mean(cons_th), pct=True)}")
    log(f"  mean per-item entropy (nats) non-thinking: {_fmt(_mean(ent_nt))}")
    log("  format sensitivity (mean flip rate, non-thinking):")
    for a in _AXES:
        log(f"    {a:<12}: {_fmt(sensitivity[a], pct=True)}")
    log(f"  mean per-item p(unknown) spread:           {_fmt(_mean(spreads))}")
    log(f"  overall abstention accuracy non-thinking:  {_fmt(acc_nt, pct=True)} (n={n_nt})")
    log(f"  overall abstention accuracy thinking:      {_fmt(acc_th, pct=True)} (n={n_th})")
    log("  NOTE: ambiguous gold is always UNKNOWN, so abstention == accuracy.")

    # ----- plots ---------------------------------------------------------- #
    plots_dir = args.out_dir / "sesgo" / "stability" / dataset.model_name / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    written: list[Path] = []
    written.append(plot_consistency_hist(
        cons_nt, cons_th, dataset.model_name, plots_dir / "consistency_hist.png"))
    written.append(plot_axis_sensitivity(
        sensitivity, dataset.model_name, plots_dir / "format_sensitivity.png"))
    written.append(plot_p_unknown_spread(
        spreads, dataset.model_name, plots_dir / "p_unknown_spread.png"))

    log(f"[viz] wrote {len(written)} plot(s):")
    for p in written:
        log(f"  {p}")


if __name__ == "__main__":
    main()
