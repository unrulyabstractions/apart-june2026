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


def _changepoints(model_dir: Path) -> list[float]:
    """The model's forking-token positions (change points) as fractions of the chain of
    thought, read from the change-point posterior: the MAP number of change points, taken as
    the highest, well-separated peaks of the location posterior tau."""
    cp = json.load((model_dir / "forking_analysis.json").open())["change_points"]
    tau = np.array(cp.get("tau_posterior", []), float)
    if not len(tau):
        return []
    ncp = cp.get("num_changepoints_posterior")
    k = int(np.argmax(ncp)) if ncp else (1 if cp.get("significant") else 0)
    if k <= 0:
        return []
    peaks = [(tau[i], i) for i in range(len(tau)) if tau[i] > 0.4
             and (i == 0 or tau[i] >= tau[i - 1]) and (i == len(tau) - 1 or tau[i] >= tau[i + 1])]
    peaks.sort(reverse=True)
    chosen: list[int] = []
    for _h, i in peaks:
        if all(abs(i - j) > 5 for j in chosen):
            chosen.append(i)
        if len(chosen) >= k:
            break
    return sorted(c / (len(tau) - 1) for c in chosen)


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
            rows.append((sm.size_b, f"{sm.name} (think)", _series(d), _changepoints(d)))
    rows.sort(key=lambda r: r[0])
    if not rows:
        print(f"[forced-fig] no models with both trajectory + forced_answer under {forking_dir}")
        return out_path
    # Four clearly-distinct, size-ordered colours (a continuous colormap made 2B/4B hard to tell).
    _COLORS = ("#4C72B0", "#55A868", "#E8A33D", "#C44E52")  # 0.8B, 2B, 4B, 9B
    col_of = {name: _COLORS[i % len(_COLORS)] for i, (_s, name, _d, _c) in enumerate(rows)}

    fig, axes = plt.subplots(3, 1, figsize=(10, 9.2), sharex=True)
    # (panel title, short y-label, key)
    panels = [("(a) Sampled outcome vs. immediate forced answer disagree",
               "KL  (nats)", "kl"),
              ("(b) Would the model abstain if forced to answer now?",
               "P(abstain)", "pu"),
              ("(c) Uncertainty of the forced next token",
               "entropy (nats)", "vocab")]
    for (ptitle, ylabel, key), ax in zip(panels, axes):
        # Mark each model's forking tokens (change points) with dashed lines in its own colour;
        # the KL spikes in (a) line up with them.
        for _size, name, _series_d, cps in rows:
            for frac in cps:
                ax.axvline(frac, color=col_of[name], ls="--", lw=1.0, alpha=0.55, zorder=1)
        for _size, name, (x, kl, pu, vocab), _cps in rows:
            y = {"kl": kl, "pu": pu, "vocab": vocab}[key]
            ax.plot(x, _smooth(y), color=col_of[name], lw=1.9,
                    label=name if key == "kl" else None, zorder=3)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ptitle, fontsize=11, fontweight="bold", loc="left", pad=4)
        ax.grid(True, axis="y", ls=":", alpha=0.4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    # In panel (b) the gold answer is to abstain, so P(abstain)=1 is the correct target.
    axes[1].set_ylim(-0.03, 1.08)
    axes[1].axhline(1.0, color="#666666", ls="--", lw=1.0, zorder=0)
    axes[1].text(0.005, 1.03, "abstain $=$ correct (ambiguous item)", fontsize=8.5,
                 style="italic", color="#555", va="bottom")
    axes[-1].set_xlabel("reasoning progress  (fraction of the chain of thought; "
                        "$0$ $=$ start, $1$ $=$ end)", fontsize=11)
    axes[0].legend(fontsize=9.5, loc="lower center", bbox_to_anchor=(0.5, 1.18),
                   frameon=False, ncol=len(rows), columnspacing=1.4)
    fig.suptitle("Sampled vs. immediate-forced answer along the chain of thought\n"
                 "Qwen3.5 thinking, racismo item  (ambiguous, negative framing; "
                 "correct answer $=$ abstain / unknown)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
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
