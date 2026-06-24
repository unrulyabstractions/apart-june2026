"""Write per-model ``best.json`` and ``worst.json`` into each ``out/stability/<model>/``
dir: the top-N (best) and bottom-N (worst) samples by PERFORMANCE in each of the four
context x polarity cells (ambiguous/disambiguated x neutral/negative).

Performance of one sample = its committed-answer confidence signed by correctness:
``label_prob`` when the answer is correct (the gold role: ``unknown`` on ambiguous items,
the named role on disambiguated ones), ``-label_prob`` when it is wrong. So a confident
correct answer ranks highest and a confident wrong answer lowest; every entry also carries
``correct`` and ``label_prob`` so the basis is explicit. Reuses the readout<->metadata join
(``enrich``) so correctness/polarity/context come from the SAME code the figures use.

  uv run python -m experiment.stability.sample_ranking_export --stability-dir out/stability \
      --dataset data/full_prompt_dataset.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

from src.common import BaseSchema

from experiment.bias.stability_readout_join import EnrichedResponse, enrich, load_metadata

TOP_N = 10
# (context, polarity) -> ModelRanking field. polarity: nonneg == neutral, neg == negative.
_CELL = {
    ("ambig", "nonneg"): "ambiguous_neutral",
    ("ambig", "neg"): "ambiguous_negative",
    ("disambig", "nonneg"): "disambiguated_neutral",
    ("disambig", "neg"): "disambiguated_negative",
}


@dataclass
class RankingEntry(BaseSchema):
    """One ranked sample: its performance plus the fields that performance is built from."""

    rank: int
    prompt_id: str
    sample_idx: int
    bias_category: str
    choice: str           # committed role: target / other / unknown / invalid
    gold: str
    correct: bool
    label_prob: float
    vocab_diversity: float
    performance: float
    prompt_excerpt: str


@dataclass
class ModelRanking(BaseSchema):
    """Best- or worst-N samples by performance for one model, split into the four cells."""

    model: str
    kind: str             # "best" (top-N) or "worst" (bottom-N)
    top_n: int
    ambiguous_neutral: list[RankingEntry]
    ambiguous_negative: list[RankingEntry]
    disambiguated_neutral: list[RankingEntry]
    disambiguated_negative: list[RankingEntry]


def _performance(r: EnrichedResponse) -> float:
    """Confidence signed by correctness: +label_prob if correct, -label_prob if wrong."""
    return r.label_prob if r.correct else -r.label_prob


def _excerpt(text: str, n: int = 200) -> str:
    return " ".join(text.split())[:n]


def _rank_cell(scored: list[tuple[float, RankingEntry]], worst: bool) -> list[RankingEntry]:
    """Top-N (best) or bottom-N (worst) of one cell, re-ranked 1..N from that end."""
    ordered = sorted(scored, key=lambda t: t[0], reverse=not worst)[:TOP_N]
    return [replace(e, rank=i) for i, (_, e) in enumerate(ordered, 1)]


def build_rankings(model_dir: Path, meta: dict) -> tuple[ModelRanking, ModelRanking]:
    """Best (top-N) and worst (bottom-N) per-cell rankings for one model."""
    samples = json.load((model_dir / "response_samples.json").open())["samples"]
    by_pid = {s["prompt_id"]: s for s in samples}
    cells: dict[str, list[tuple[float, RankingEntry]]] = {f: [] for f in _CELL.values()}
    for r in enrich(samples, model_dir.name, meta):
        field = _CELL.get((r.context, r.polarity))
        if field is None:
            continue
        s = by_pid[r.prompt_id]
        perf = _performance(r)
        cells[field].append((perf, RankingEntry(
            rank=0, prompt_id=r.prompt_id, sample_idx=s["sample_idx"],
            bias_category=r.bias_category, choice=r.role, gold=r.gold, correct=r.correct,
            label_prob=r.label_prob, vocab_diversity=r.vocab_diversity, performance=perf,
            prompt_excerpt=_excerpt(s["prompt_text"]),
        )))
    best = ModelRanking(model=model_dir.name, kind="best", top_n=TOP_N,
                        **{f: _rank_cell(cells[f], worst=False) for f in _CELL.values()})
    worst = ModelRanking(model=model_dir.name, kind="worst", top_n=TOP_N,
                         **{f: _rank_cell(cells[f], worst=True) for f in _CELL.values()})
    return best, worst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stability-dir", type=Path, default=Path("out/stability"))
    ap.add_argument("--dataset", type=Path, default=Path("data/full_prompt_dataset.json"))
    a = ap.parse_args()
    meta = load_metadata(a.dataset)
    dirs = sorted(d for d in a.stability_dir.iterdir()
                  if (d / "response_samples.json").exists())
    for d in dirs:
        best, worst = build_rankings(d, meta)
        (d / "best.json").write_text(json.dumps(best.to_dict(), indent=2, ensure_ascii=False))
        (d / "worst.json").write_text(json.dumps(worst.to_dict(), indent=2, ensure_ascii=False))
        (d / "ranking.json").unlink(missing_ok=True)  # superseded by best.json
        print(f"[ranking] {d.name:36s} -> best.json + worst.json")
    print(f"[ranking] wrote best.json + worst.json for {len(dirs)} models")


if __name__ == "__main__":
    main()
