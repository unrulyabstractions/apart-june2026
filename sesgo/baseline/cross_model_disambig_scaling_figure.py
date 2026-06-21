"""F2: on CLEAR questions, do bigger models do better, and does wording sway them?

Run-by-path figure over the SESGO baseline size sweep. On disambiguated items only
(one option is actually correct) we score the direct-answer readout. TOP: accuracy
vs model size (log x), one line per family, faceted by bias category. BOTTOM: the
signed wording gap per model per category (negative-worded minus neutral-worded
accuracy; above 0 = more accurate when negatively loaded). Wilson 95% CIs and n
everywhere; sizing/family from sesgo_model_sizing, cells from disambig_wording_cells.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write a PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.datasets.sesgo_eval import SesgoDataset  # noqa: E402
from sesgo.baseline.disambig_wording_cells import (  # noqa: E402
    DisambigCell,
    cells_for_model,
    family_accuracy_series,
    wording_gap,
)
from sesgo.common.plain_language_labels import (  # noqa: E402
    CATEGORY_LABEL,
    CATEGORY_ORDER,
    RANDOM_GUESS_LABEL,
)

# Okabe-Ito colourblind-safe hues: one per family (top facets) and a distinct one
# per bias category (bottom diverging bars), so colour never means two things.
_FAMILY_COLORS = {
    "Qwen": "#0072B2", "Llama": "#D55E00", "Gemma": "#009E73", "Mistral": "#CC79A7",
}
_CATEGORY_COLORS = {
    "clasismo": "#0072B2", "racismo": "#E69F00",
    "xenofobia": "#009E73", "genero": "#CC79A7",
}
_FAMILY_ORDER = ("Qwen", "Llama", "Gemma", "Mistral")


def _accuracy_facet(ax, cat: str, cells: list[DisambigCell]) -> None:
    """One facet: pooled-over-wording accuracy vs size, one line per family."""
    families = {c.family for c in cells if c.category == cat}
    for fam in families:
        sizes, accs, yerr = family_accuracy_series(cells, cat, fam)
        ax.errorbar(sizes, accs, yerr=np.array(yerr).T, color=_FAMILY_COLORS[fam],
                    marker="o", linewidth=1.6, markersize=5, capsize=2.5,
                    elinewidth=0.9, alpha=0.9)
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.12)
    ax.axhline(1.0 / 3, ls="--", lw=1.0, color="#888888", alpha=0.7, zorder=0)
    ax.set_title(CATEGORY_LABEL[cat], fontsize=12, fontweight="bold", loc="left")
    ax.grid(True, which="both", axis="x", ls=":", alpha=0.4)


def _gap_panel(ax, cells: list[DisambigCell]) -> None:
    """Bottom panel: signed (negative - neutral) wording gap per model x category."""
    models = sorted({(c.params_b, c.family, c.model) for c in cells})
    by_key = {(c.model, c.category, c.polarity): c for c in cells}
    width = 0.8 / len(CATEGORY_ORDER)
    x = np.arange(len(models))
    for j, cat in enumerate(CATEGORY_ORDER):
        offs, gaps, errs = [], [], []
        for i, (_, _fam, model) in enumerate(models):
            neg = by_key.get((model, cat, "neg"))
            non = by_key.get((model, cat, "nonneg"))
            if neg is None or non is None:
                continue
            gap, half = wording_gap(neg, non)
            offs.append(x[i] + (j - (len(CATEGORY_ORDER) - 1) / 2) * width)
            gaps.append(gap)
            errs.append(half)
        ax.bar(offs, gaps, width, yerr=errs, color=_CATEGORY_COLORS[cat], capsize=1.5,
               edgecolor="white", linewidth=0.3, label=CATEGORY_LABEL[cat],
               error_kw={"elinewidth": 0.6, "ecolor": "#555555"})
    ax.axhline(0, color="#333333", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{fam}\n{s:g}B" for s, fam, _ in models], fontsize=8,
                       rotation=0, ha="center")
    ax.set_ylabel("Better on negatively-\nworded questions  ->", fontsize=9.5)
    ax.set_title(
        "How much does negative wording help each model? (accuracy on negative "
        "minus neutral wording, by bias category)",
        fontsize=11.5, fontweight="bold", loc="left")
    ax.legend(title="Bias category", ncol=4, loc="upper left", frameon=False,
              fontsize=9, title_fontsize=9.5)
    ax.grid(True, axis="y", ls=":", alpha=0.4)


def _family_legend(fig, cells: list[DisambigCell]) -> None:
    """Colour key for the accuracy facets (one entry per present model family)."""
    handles = [plt.Line2D([], [], color=_FAMILY_COLORS[f], marker="o", linestyle="-",
                          label=f) for f in _FAMILY_ORDER if any(c.family == f for c in cells)]
    fig.legend(handles=handles, title="Model family (line colour)", loc="upper left",
               bbox_to_anchor=(1.005, 0.9), frameon=False, fontsize=10, title_fontsize=10.5)


def plot_disambig_scaling(cells: list[DisambigCell], out_path, n_models: int) -> None:
    """Render the two-panel scaling + wording-gap figure for the baseline sweep."""
    fig = plt.figure(figsize=(13, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.15])
    facets = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(len(CATEGORY_ORDER))]
    for ax, cat in zip(facets, CATEGORY_ORDER):
        _accuracy_facet(ax, cat, cells)
        ax.set_ylabel("Accuracy on clear questions\n(higher is better)", fontsize=9)
        ax.set_xlabel("Model size (billions of parameters, log scale)", fontsize=9.5)
    facets[0].text(0.02, 1.0 / 3 + 0.01, f"{RANDOM_GUESS_LABEL} (1 in 3)",
                   transform=facets[0].get_yaxis_transform(),
                   ha="left", va="bottom", fontsize=8, color="#777777")
    _family_legend(fig, cells)
    _gap_panel(fig.add_subplot(gs[2, :]), cells)
    fig.suptitle(
        f"On clear questions (one answer is correct), do bigger models do better, "
        f"and does wording sway them?   ({n_models} models)\n"
        "Top: accuracy vs model size, one line per family, by bias category. Bottom: "
        "above 0 = more accurate when the question is negatively loaded. Wilson 95% CIs.",
        fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _collect(base_dir: Path) -> list[DisambigCell]:
    """Load every model dir and reduce it to its disambiguated accuracy cells."""
    out: list[DisambigCell] = []
    for path in sorted(base_dir.glob("*/response_samples.json")):
        dataset = SesgoDataset.from_json(path)
        out += cells_for_model(dataset.model_name, dataset.samples)
    return out


def main() -> None:
    """Scan the baseline sweep, build cells, render the F2 figure."""
    base_dir = Path("out/sesgo/baseline")
    cells = _collect(base_dir)
    n_models = len({c.model for c in cells})
    plots_dir = base_dir / "cross_model" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "disambig_scaling_wording_gap.png"
    plot_disambig_scaling(cells, out_path, n_models)
    print(f"wrote {out_path}  ({n_models} models, {len(cells)} cells)")


if __name__ == "__main__":
    main()
