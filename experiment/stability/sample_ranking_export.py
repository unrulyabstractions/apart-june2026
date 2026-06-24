"""Write a per-model ``ranking.json`` into each ``out/stability/<model>/`` dir: the
top-N samples by PERFORMANCE in each of the four context x polarity cells
(ambiguous/disambiguated x neutral/negative).

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
from dataclasses import dataclass
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
    """Top-N samples by performance for one model, split into the four cells."""

    model: str
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


def build_ranking(model_dir: Path, meta: dict) -> ModelRanking:
    """Rank each cell's samples by performance and keep the top-N per cell."""
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
    ranked = {}
    for field, scored in cells.items():
        top = [e for _, e in sorted(scored, key=lambda t: t[0], reverse=True)[:TOP_N]]
        for i, e in enumerate(top, 1):
            e.rank = i
        ranked[field] = top
    return ModelRanking(model=model_dir.name, top_n=TOP_N, **ranked)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stability-dir", type=Path, default=Path("out/stability"))
    ap.add_argument("--dataset", type=Path, default=Path("data/full_prompt_dataset.json"))
    a = ap.parse_args()
    meta = load_metadata(a.dataset)
    dirs = sorted(d for d in a.stability_dir.iterdir()
                  if (d / "response_samples.json").exists())
    for d in dirs:
        ranking = build_ranking(d, meta)
        out = d / "ranking.json"
        out.write_text(json.dumps(ranking.to_dict(), indent=2, ensure_ascii=False))
        counts = " ".join(f"{f.split('_')[0][:5]}/{f.split('_')[1][:3]}={len(getattr(ranking, f))}"
                          for f in _CELL.values())
        print(f"[ranking] {d.name:36s} -> {out.name}  ({counts})")
    print(f"[ranking] wrote ranking.json for {len(dirs)} models")


if __name__ == "__main__":
    main()
