"""Compute, plot, and SELECT the best scaffold for a collected SesgoDataset.

Run-by-path driver for the SELECTION study. Loads a samples.json produced by
collect_selection_samples.py (a SesgoDataset) and answers one question: of the
five scaffold conditions — the no-scaffold baseline (scaffold_id == None) plus
the four debiasing scaffolds — which one best pushes the model toward abstaining
on ambiguous SESGO items?

On ambiguous SESGO items the gold answer is always UNKNOWN, so "accuracy" is the
fraction of predictions that ABSTAIN (predict UNKNOWN) rather than pick a group.
We compute that abstention accuracy PER SCAFFOLD at both the non-thinking
(teacher-forced 3-way softmax) and thinking (sampled, parsed) levels, rank the
scaffolds, and SELECT the best one by its thinking abstention (the behavior we
care about — the model's actual free-form answer), falling back to non-thinking
when no thinking draw survived. The combined (mean of the two levels) score is
reported alongside as a tie-aware sanity check.

It renders into out/sesgo/selection/<MODEL>/plots/:
  abstention_by_scaffold.png        - grouped bars, non-thinking vs thinking, one
                                       group per scaffold (baseline + 4); the
                                       SELECTED best scaffold is annotated in the
                                       title and starred on its bar group.
  abstention_by_scaffold_category.png- abstention accuracy by scaffold x
                                       bias_category (grouped bars per category).

Robust to subsampled / partial data: a scaffold with no parsed thinking draw
(every sample's thinking sample_size == 0) is simply skipped from the thinking
series rather than plotted as zero, so the SELECT never rewards a scaffold for
having no decodable answer.

Usage:
  uv run python sesgo/selection/visualize_selection_samples.py \
      out/sesgo/selection/Qwen3-0.6B/samples.json
  uv run python sesgo/selection/visualize_selection_samples.py SAMPLES.json --out-dir out
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves
# regardless of cwd. From <repo>/sesgo/selection/x.py, parents[2] is the root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.common.logging import log, log_box, log_header, log_section  # noqa: E402
from src.datasets.sesgo_eval import SesgoDataset  # noqa: E402

# The no-scaffold condition has scaffold_id == None; label it so it reads clearly.
_BASELINE = "(baseline)"
# House palette: non-thinking is blue, thinking is red (mirrors stability/geometry).
_NT_COLOR = "#30638e"
_TH_COLOR = "#d1495b"
# Warm gold band drawn behind the SELECTED scaffold's bar group so it pops.
_SELECT_BG = "#ffe9a8"
_SELECT_EDGE = "#e0a800"


def _wrap_label(text: str, width: int = 14) -> str:
    """Wrap a long scaffold id onto multiple lines for a readable x-tick.

    Scaffold ids are snake_case, so textwrap (which only breaks on whitespace)
    leaves them as one long line. We soft-break on underscores first, then wrap.
    """
    return textwrap.fill(text.replace("_", "_ "), width=width).replace("_ ", "_")


def _tick_labels(scaffolds: list[str], selected: str | None) -> list[str]:
    """Wrapped x-tick labels; the SELECTED scaffold gets a [SELECTED] caption."""
    out = []
    for sc in scaffolds:
        wrapped = _wrap_label(sc)
        out.append(f"{wrapped}\n[SELECTED]" if sc == selected else wrapped)
    return out


def _highlight_selected(ax, x: np.ndarray, scaffolds: list[str], selected: str | None) -> None:
    """Draw a soft gold band + a star marker for the SELECTED scaffold group.

    The star is a matplotlib path marker (not a glyph), so it always renders
    regardless of the available system fonts.
    """
    if selected is None or selected not in scaffolds:
        return
    i = scaffolds.index(selected)
    ax.axvspan(
        x[i] - 0.5, x[i] + 0.5, color=_SELECT_BG, alpha=0.55,
        zorder=0, ymin=0.0, ymax=1.0,
    )
    for edge in (x[i] - 0.5, x[i] + 0.5):
        ax.axvline(edge, color=_SELECT_EDGE, lw=1.0, ls=":", zorder=1)
    ax.scatter(
        [x[i]], [1.17], marker="*", s=320, color=_SELECT_EDGE,
        edgecolor="#7a5c00", linewidth=0.6, zorder=6, clip_on=False,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for selection visualization."""
    parser = argparse.ArgumentParser(
        description="Compute, plot, and SELECT the best scaffold for a SesgoDataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "samples",
        type=Path,
        help="Path to samples.json (a SesgoDataset) from collect_selection_samples.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out"),
        help="Base output dir; plots land at <out-dir>/sesgo/selection/<MODEL>/plots/",
    )
    return parser.parse_args()


def _scaffold_label(scaffold_id: str | None) -> str:
    """Human label for a scaffold condition; None == the no-scaffold baseline."""
    return scaffold_id or _BASELINE


def _scaffold_order(dataset: SesgoDataset) -> list[str]:
    """Scaffold labels with the baseline first, the rest sorted after it."""
    labels = {_scaffold_label(s.scaffold_id) for s in dataset.samples}
    rest = sorted(labels - {_BASELINE})
    return ([_BASELINE] if _BASELINE in labels else []) + rest


def _acc(flags: list[bool]) -> float | None:
    """Abstention accuracy over a list of correctness flags, None when empty."""
    return sum(flags) / len(flags) if flags else None


def _abstention_by_scaffold(
    dataset: SesgoDataset, level: str
) -> tuple[dict[str, float | None], dict[str, int]]:
    """Per-scaffold (accuracy, n) at the given level.

    A sample contributes only when it has a prediction at this level. At the
    thinking level that means at least one parsed draw survived (sample_size > 0);
    scaffolds whose every sample yielded no parsed draw end up with n == 0 and an
    accuracy of None, which the SELECT and the plot both treat as "skip".
    """
    flags: dict[str, list[bool]] = defaultdict(list)
    for s in dataset.samples:
        label = _scaffold_label(s.scaffold_id)
        if level == "non_thinking" and s.predicted_non_thinking is not None:
            flags[label].append(s.correct_non_thinking)
        elif level == "thinking" and s.predicted_thinking is not None:
            flags[label].append(s.correct_thinking)
    accs = {k: _acc(v) for k, v in flags.items()}
    ns = {k: len(v) for k, v in flags.items()}
    return accs, ns


def _rank_and_select(
    scaffolds: list[str],
    nt_acc: dict[str, float | None],
    th_acc: dict[str, float | None],
) -> tuple[list[tuple[str, float | None, float | None, float | None]], str | None]:
    """Rank scaffolds and SELECT the best by abstention (thinking-priority).

    For each scaffold we form a "combined" score: the mean of its non-thinking
    and thinking abstention when both are defined, else whichever single level is
    defined. The SELECT key prioritizes THINKING abstention (the model's actual
    free-form behavior), then the combined score, then non-thinking, then prefers
    a real (non-baseline) scaffold over the baseline on an exact tie — a debiasing
    preamble that merely matches the baseline is not worth selecting over it. The
    returned rows are sorted by that same key (best first).
    """
    rows: list[tuple[str, float | None, float | None, float | None]] = []
    for sc in scaffolds:
        nt = nt_acc.get(sc)
        th = th_acc.get(sc)
        defined = [v for v in (nt, th) if v is not None]
        combined = float(np.mean(defined)) if defined else None
        rows.append((sc, nt, th, combined))

    def sort_key(row: tuple[str, float | None, float | None, float | None]):
        sc, nt, th, combined = row
        # Higher is better; None sinks to the bottom via -inf. The final term
        # breaks exact ties in favor of a real scaffold over the baseline.
        return (
            th if th is not None else float("-inf"),
            combined if combined is not None else float("-inf"),
            nt if nt is not None else float("-inf"),
            0 if sc == _BASELINE else 1,
        )

    ranked = sorted(rows, key=sort_key, reverse=True)
    # Only select among scaffolds that have ANY defined abstention score.
    selectable = [r for r in ranked if r[3] is not None]
    selected = selectable[0][0] if selectable else None
    return ranked, selected


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _annotate_bars(ax, bars, scaffolds, accs, ns) -> None:
    """Label each bar with its abstention % and n (n/a when undefined)."""
    for bar, sc in zip(bars, scaffolds):
        acc = accs.get(sc)
        n = ns.get(sc, 0)
        txt = "n/a" if acc is None else f"{acc:.0%}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{txt}\nn={n}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            linespacing=1.05,
        )


def plot_abstention_by_scaffold(
    scaffolds: list[str],
    nt_acc: dict[str, float | None],
    nt_n: dict[str, int],
    th_acc: dict[str, float | None],
    th_n: dict[str, int],
    selected: str | None,
    total_n: int,
    model: str,
    out_path: Path,
) -> Path:
    """Grouped bars: per-scaffold non-thinking vs thinking abstention accuracy.

    Scaffolds arrive pre-sorted by abstention (best first). Missing (None) bars
    render as zero height with an "n/a" label so the group still reserves its
    slot. The SELECTED best scaffold gets a gold highlight band, a starred tick,
    and a callout in the title, so it pops at a glance.
    """
    x = np.arange(len(scaffolds))
    width = 0.4

    nt_vals = [nt_acc.get(sc) or 0.0 for sc in scaffolds]
    th_vals = [th_acc.get(sc) or 0.0 for sc in scaffolds]

    fig, ax = plt.subplots(figsize=(max(9, 2.0 * len(scaffolds)), 6.2), layout="constrained")
    _highlight_selected(ax, x, scaffolds, selected)
    bars_nt = ax.bar(x - width / 2, nt_vals, width, label="non-thinking", color=_NT_COLOR, zorder=3)
    bars_th = ax.bar(x + width / 2, th_vals, width, label="thinking", color=_TH_COLOR, zorder=3)
    _annotate_bars(ax, bars_nt, scaffolds, nt_acc, nt_n)
    _annotate_bars(ax, bars_th, scaffolds, th_acc, th_n)

    ax.set_xticks(x)
    ax.set_xticklabels(_tick_labels(scaffolds, selected), fontsize=9)
    for lbl, sc in zip(ax.get_xticklabels(), scaffolds):
        if sc == selected:
            lbl.set_fontweight("bold")
            lbl.set_color("#8a6d00")
    ax.set_xlim(-0.6, len(scaffolds) - 0.4)
    ax.set_ylim(0, 1.22)  # headroom so value-labels never collide with the title
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("abstention accuracy\n(fraction predicted UNKNOWN, gold = UNKNOWN)")
    sel_txt = selected if selected is not None else "n/a"
    ax.set_title(
        f"SESGO selection — abstention by scaffold  ·  {model}\n"
        f"SELECTED best scaffold (gold star, highlighted): {sel_txt}   ·   sorted best→worst   ·   total n={total_n}",
        fontsize=12, fontweight="bold", pad=16,
    )
    # Legend pinned outside the axes (upper-right, just under the title) so it
    # never sits on the gold SELECTED highlight band or on any value-label.
    ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), framealpha=0.95, fontsize=9)
    ax.margins(x=0.02)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_abstention_by_scaffold_category(
    scaffolds: list[str],
    categories: list[str],
    cell: dict[tuple[str, str], float | None],
    cell_n: dict[tuple[str, str], int],
    selected: str | None,
    level: str,
    model: str,
    out_path: Path,
) -> Path:
    """Grouped bars: abstention accuracy by scaffold x bias_category.

    One group per scaffold (sorted best→worst), one bar per bias_category. Each
    present bar carries its n; absent cells get a faint "n/a" stub so the small,
    uneven coverage of a minimal run stays visible rather than reading as zero.
    The SELECTED scaffold gets the same gold highlight band as the main plot.
    """
    x = np.arange(len(scaffolds))
    n_cat = max(1, len(categories))
    width = 0.82 / n_cat
    colors = sns.color_palette("colorblind", n_cat)

    fig, ax = plt.subplots(figsize=(max(9, 2.0 * len(scaffolds)), 6.2), layout="constrained")
    _highlight_selected(ax, x, scaffolds, selected)
    for ci, cat in enumerate(categories):
        offset = (ci - (n_cat - 1) / 2) * width
        vals = [cell.get((sc, cat)) or 0.0 for sc in scaffolds]
        bars = ax.bar(x + offset, vals, width, label=cat, color=colors[ci], zorder=3)
        for bar, sc in zip(bars, scaffolds):
            v = cell.get((sc, cat))
            cx = bar.get_x() + bar.get_width() / 2
            if v is None:
                ax.text(cx, 0.02, "n/a", ha="center", va="bottom",
                        fontsize=6.5, rotation=90, color="#9a9a9a")
            else:
                ax.text(cx, v + 0.012, f"{v:.0%}\nn={cell_n.get((sc, cat), 0)}",
                        ha="center", va="bottom", fontsize=6.5, linespacing=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(_tick_labels(scaffolds, selected), fontsize=9)
    for lbl, sc in zip(ax.get_xticklabels(), scaffolds):
        if sc == selected:
            lbl.set_fontweight("bold")
            lbl.set_color("#8a6d00")
    ax.set_xlim(-0.6, len(scaffolds) - 0.4)
    ax.set_ylim(0, 1.22)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("abstention accuracy\n(fraction predicted UNKNOWN)")
    ax.set_title(
        f"SESGO selection — abstention by scaffold × bias_category  ·  {level}  ·  {model}\n"
        f"SELECTED (gold star): {selected or 'n/a'}   ·   per-bar n shown   ·   small-n run",
        fontsize=12, fontweight="bold", pad=16,
    )
    ax.legend(title="bias_category", fontsize=8.5, title_fontsize=9,
              ncol=1, loc="center right", framealpha=0.95)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _abstention_by_scaffold_category(
    dataset: SesgoDataset, level: str
) -> tuple[dict[tuple[str, str], float | None], dict[tuple[str, str], int]]:
    """Per-(scaffold, bias_category) abstention accuracy and n at the given level."""
    flags: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for s in dataset.samples:
        key = (_scaffold_label(s.scaffold_id), s.bias_category)
        if level == "non_thinking" and s.predicted_non_thinking is not None:
            flags[key].append(s.correct_non_thinking)
        elif level == "thinking" and s.predicted_thinking is not None:
            flags[key].append(s.correct_thinking)
    return {k: _acc(v) for k, v in flags.items()}, {k: len(v) for k, v in flags.items()}


def _fmt(value: float | None) -> str:
    """Render an accuracy as a percentage, or n/a when undefined."""
    return f"{value:.1%}" if value is not None else "n/a"


def main() -> None:
    """Load the SesgoDataset, rank scaffolds, SELECT the best, plot, and log."""
    args = parse_args()
    log_header("VISUALIZE SESGO SELECTION")

    dataset = SesgoDataset.from_json(args.samples)
    log(f"[viz] loaded {len(dataset.samples)} samples (model={dataset.model_name})")
    scaffolds = _scaffold_order(dataset)
    log(f"[viz] {len(scaffolds)} scaffold condition(s): {', '.join(scaffolds)}")

    has_thinking = any(s.predicted_thinking is not None for s in dataset.samples)

    # Per-scaffold abstention accuracy at both levels.
    nt_acc, nt_n = _abstention_by_scaffold(dataset, "non_thinking")
    th_acc, th_n = (
        _abstention_by_scaffold(dataset, "thinking") if has_thinking else ({}, {})
    )

    # Rank + SELECT the best scaffold. `ranked` is sorted best→worst, so its
    # scaffold order also drives the plots (the prompt: sort by abstention).
    ranked, selected = _rank_and_select(scaffolds, nt_acc, th_acc)
    plot_order = [row[0] for row in ranked]

    # ----- per-scaffold table + ranking to the log ------------------------ #
    log_section("PER-SCAFFOLD ABSTENTION (accuracy = fraction predicted UNKNOWN)")
    log(f"  {'scaffold':<34} {'non-thinking':>16} {'thinking':>16} {'combined':>10}")
    for sc, nt, th, combined in ranked:
        nt_txt = f"{_fmt(nt)} (n={nt_n.get(sc, 0)})"
        th_txt = f"{_fmt(th)} (n={th_n.get(sc, 0)})"
        log(f"  {sc:<34} {nt_txt:>16} {th_txt:>16} {_fmt(combined):>10}")
    log("  NOTE: ambiguous gold is always UNKNOWN, so abstention == accuracy.")
    log("  SELECT key: thinking abstention, then combined, then non-thinking.")

    # ----- the SELECTED best scaffold, prominently ------------------------ #
    if selected is not None:
        sel_nt = _fmt(nt_acc.get(selected))
        sel_th = _fmt(th_acc.get(selected))
        log_box(
            f"SELECTED BEST SCAFFOLD: {selected}",
            subtitle=f"thinking abstention {sel_th}  |  non-thinking abstention {sel_nt}",
            gap=1,
        )
    else:
        log_box("SELECTED BEST SCAFFOLD: n/a (no scaffold had a defined score)", gap=1)

    # ----- plots ---------------------------------------------------------- #
    plots_dir = args.out_dir / "sesgo" / "selection" / dataset.model_name / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    written: list[Path] = []
    written.append(
        plot_abstention_by_scaffold(
            plot_order,
            nt_acc,
            nt_n,
            th_acc,
            th_n,
            selected,
            len(dataset.samples),
            dataset.model_name,
            plots_dir / "abstention_by_scaffold.png",
        )
    )

    # Abstention by scaffold x bias_category (prefer thinking, fall back to non-thinking).
    cat_level = "thinking" if has_thinking else "non_thinking"
    categories = sorted({s.bias_category for s in dataset.samples})
    cell, cell_n = _abstention_by_scaffold_category(dataset, cat_level)
    written.append(
        plot_abstention_by_scaffold_category(
            plot_order,
            categories,
            cell,
            cell_n,
            selected,
            cat_level,
            dataset.model_name,
            plots_dir / "abstention_by_scaffold_category.png",
        )
    )

    log(f"[viz] wrote {len(written)} plot(s):")
    for p in written:
        log(f"  {p}")


if __name__ == "__main__":
    main()
