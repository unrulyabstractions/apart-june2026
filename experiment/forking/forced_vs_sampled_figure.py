"""Compare models on three PER-TOKEN-POSITION quantities along the chain of thought:
  1. KL(O_t || forced): divergence between the SAMPLED outcome distribution O_t (forking,
     renormalized to the 3 labels target/other/unknown) and the IMMEDIATE forced-answer
     distribution (the normalized next-token distribution over the 3 option labels if the
     model were forced to answer at position t).
  2. the immediate forced answer: P(unknown) under that forced distribution (the share the
     model would abstain if forced to commit now).
  3. vocab entropy: Shannon entropy of the full next-token distribution at the forced answer.

One line per model (size-coloured). x is reasoning progress (fraction of the chain of
thought) so different CoT lengths are comparable. Auto-discovers every model under the
forking dir that has both ``forking_trajectory.json`` and ``forced_answer_dynamics.json``.

  uv run python -m experiment.forking.forced_vs_sampled_figure --forking-dir out/forking \
      --out paper/figures/forced_vs_sampled.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiment.common.sweep_models import parse_model

_EPS = 1e-6
_SMOOTH = 9  # rolling-mean window for legibility on 250-380 noisy per-token points


def _kl3(p4: list[float], q3: list[float]) -> float:
    """KL(p || q) over the 3 labels target/other/unknown; p4 is the 4-way O_t (drop+renorm
    the unparseable mass), q3 the forced 3-way. Epsilon-smoothed so zeros don't blow up."""
    p = np.array(p4[:3], float) + _EPS; p /= p.sum()
    q = np.array(q3, float) + _EPS; q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def _smooth(y: np.ndarray) -> np.ndarray:
    if len(y) < _SMOOTH:
        return y
    k = np.ones(_SMOOTH) / _SMOOTH
    return np.convolve(y, k, mode="same")


def _series(model_dir: Path):
    traj = json.load((model_dir / "forking_trajectory.json").open())
    forced = json.load((model_dir / "forced_answer_dynamics.json").open())
    n = min(len(traj["positions"]), len(forced))
    o = [traj["positions"][t]["outcome_histogram"] for t in range(n)]
    kl = np.array([_kl3(o[t], forced[t]["forced_prob"]) for t in range(n)])
    p_unknown = np.array([forced[t]["forced_prob"][2] for t in range(n)])  # role order t,o,u
    vocab = np.array([forced[t]["vocab_entropy"] for t in range(n)])
    x = np.linspace(0, 1, n)
    return x, kl, p_unknown, vocab


def build(forking_dir: Path, out_path: Path) -> Path:
    rows = []
    for d in sorted(forking_dir.iterdir()):
        sm = parse_model(d.name) if d.is_dir() else None
        if sm and (d / "forking_trajectory.json").exists() and (d / "forced_answer_dynamics.json").exists():
            rows.append((sm.size_b, f"{sm.name} (think)", _series(d)))
    rows.sort(key=lambda r: r[0])
    if not rows:
        print(f"[forced-fig] no models with both trajectory + forced_answer under {forking_dir}")
        return out_path
    sizes = [r[0] for r in rows]
    cmap = plt.cm.viridis
    cnorm = lambda s: cmap(0.15 + 0.7 * (np.log(s) - np.log(min(sizes))) /
                           max(np.log(max(sizes)) - np.log(min(sizes)), 1e-9))

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    panels = [("KL(sampled $O_t$ $\\parallel$ forced answer)  [nats]", 1),
              ("immediate forced P(unknown / abstain)", 2),
              ("vocab entropy of forced answer  [nats]", 3)]
    for (ylabel, idx), ax in zip(panels, axes):
        for size, name, (x, kl, pu, vocab) in rows:
            y = {1: kl, 2: pu, 3: vocab}[idx]
            ax.plot(x, _smooth(y), color=cnorm(size), lw=1.6, label=name if idx == 1 else None)
        ax.set_ylabel(ylabel, fontsize=10); ax.grid(True, alpha=0.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[-1].set_xlabel("reasoning progress  (fraction of the chain of thought)", fontsize=11)
    axes[0].legend(fontsize=9, loc="upper right", frameon=False, ncol=len(rows))
    fig.suptitle("Sampled vs. immediate-forced answer along the chain of thought "
                 "(Qwen3.5 thinking, racismo item)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[forced-fig] wrote {out_path}  ({len(rows)} models: {', '.join(r[1] for r in rows)})")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--forking-dir", type=Path, default=Path("out/forking"))
    ap.add_argument("--out", type=Path, default=Path("paper/figures/forced_vs_sampled.png"))
    a = ap.parse_args()
    build(a.forking_dir, a.out)


if __name__ == "__main__":
    main()
