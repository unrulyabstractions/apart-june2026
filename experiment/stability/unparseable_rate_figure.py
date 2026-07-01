"""Stacked bar chart of the NON-COMMITTAL greedy-readout rate per model, split into two
failure modes: an explicit safety REFUSAL vs an otherwise UNPARSEABLE answer.

A greedy readout is "invalid" when no committed option label can be recovered. We split those:
  * REFUSAL      -- an explicit safety decline ("No puedo proporcionar ayuda ... discriminacion").
  * UNPARSEABLE  -- everything else: a label outside the item's option set ("b)" under x)y)z)),
                    a prose non-answer, or garbled output.
Both matter for the bias study: neither is the gold "unknown" abstention, so both count against
accuracy, but a refusal is a deliberate policy choice while an unparseable answer is a
format/competence failure.

  uv run python -m experiment.stability.unparseable_rate_figure
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from experiment.common.sweep_models import parse_model

REPO_ROOT = Path(__file__).resolve().parents[2]

_REFUSAL_COLOR, _UNPARSE_COLOR = "#CC3311", "#9E9E9E"

def _rates(stab_root: Path) -> list[tuple]:
    """(total_frac, name, refusal_frac, unparse_frac, refusal_n, unparse_n, total_n) per slice.

    Reads the persisted ``choice`` directly: ``refusal`` is an explicit safety decline,
    ``invalid`` is an otherwise-unparseable answer (both written by the readout runner)."""
    rows = []
    for d in sorted(stab_root.iterdir()):
        f = d / "response_samples.json"
        if not f.exists():
            continue
        m = parse_model(d.name)
        if m is None:
            continue
        s = json.load(f.open())["samples"]
        n = len(s)
        ref = sum(1 for x in s if x.get("choice") == "refusal")
        unp = sum(1 for x in s if x.get("choice") == "invalid")
        rows.append(((ref + unp) / n, m.name, ref / n, unp / n, ref, unp, n))
    return sorted(rows)  # ascending -> barh draws bottom-up, highest total ends on top


def build(stab_root: Path, out: Path) -> None:
    rows = _rates(stab_root)
    names = [r[1] for r in rows]
    refusal = [100 * r[2] for r in rows]
    unparse = [100 * r[3] for r in rows]
    ys = range(len(rows))

    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    ax.barh(ys, refusal, color=_REFUSAL_COLOR, edgecolor="white", height=0.72, label="refusal")
    ax.barh(ys, unparse, left=refusal, color=_UNPARSE_COLOR, edgecolor="white", height=0.72,
            label="unparseable (other)")
    ax.set_yticks(list(ys)); ax.set_yticklabels(names, fontsize=9)
    totals = [refusal[i] + unparse[i] for i in ys]
    ax.set_xlim(0, max(totals) * 1.20 + 1)
    ax.set_xlabel("Non-committal answers (% of items, greedy readout)", fontsize=11)
    ax.set_title("Refusal vs. unparseable, by model", fontsize=13, fontweight="bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, axis="x", ls=":", alpha=0.4)
    for i in ys:
        r = rows[i]
        note = f"{totals[i]:.0f}%" + (f"  ({refusal[i]:.0f} ref $+$ {unparse[i]:.0f} unp)"
                                      if r[4] else "")
        ax.text(totals[i] + max(totals) * 0.012, i, note, va="center", ha="left",
                fontsize=7.5, color="#333")
    ax.legend(fontsize=9.5, loc="lower right", frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({len(rows)} models)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stability-dir", type=Path, default=REPO_ROOT / "out" / "stability")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "paper" / "figures" / "unparseable_rate.png")
    a = ap.parse_args()
    build(a.stability_dir, a.out)


if __name__ == "__main__":
    main()
