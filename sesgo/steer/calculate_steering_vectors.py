"""Compute per-layer SESGO steering vectors from captured geometry (run-by-path).

Loads a GeometryDataset (out/sesgo/geometry/<MODEL>/response_samples.json), pairs
scaffold vs no-scaffold samples by question_id, makes a SEEDED 70/30 train/test
split over the question_ids, and on the TRAIN pairs ONLY computes a per-LAYER
diff-of-means steering vector
``v[L] = mean over (train pairs x change-of-turn positions) of
(resid_scaffold - resid_noscaffold)``
for every captured layer. The primary steering layer defaults to the mid-depth
layer where the scaffold silhouette peaks (read from analysis/projections.json).
The vectors AND the split are saved as one BaseSchema bundle so the test driver
reuses the exact held-out items.

Usage:
  uv run python sesgo/steer/calculate_steering_vectors.py \
      out/sesgo/geometry/Qwen3-0.6B/response_samples.json
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.common.file_io import save_json  # noqa: E402
from src.common.logging import log, log_header  # noqa: E402
from src.datasets.sesgo_eval import GeometryDataset  # noqa: E402
from src.steer import (  # noqa: E402
    CHANGE_OF_TURN_POSITIONS,
    SteeringVectorBundle,
    build_pairs,
    captured_layers,
    primary_layer,
    split_question_ids,
    steering_vectors_all_layers,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI args for the steering-vector calculation."""
    parser = argparse.ArgumentParser(description="Compute SESGO steering vectors")
    parser.add_argument(
        "samples", type=Path, help="response_samples.json (a GeometryDataset)"
    )
    parser.add_argument("--seed", type=int, default=42, help="train/test split seed")
    parser.add_argument(
        "--train-fraction", type=float, default=0.7, help="fraction of pairs for train"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("out"), help="base output directory"
    )
    return parser.parse_args()


def main() -> None:
    """Pair by qid, split, fit per-layer diff-of-means vectors, and save them."""
    args = parse_args()
    log_header("CALCULATE SESGO STEERING VECTORS")

    dataset = GeometryDataset.from_json(args.samples)
    root = args.samples.resolve().parent  # activation paths resolve against here
    pairs = build_pairs(dataset.samples)
    log(f"[steer] model={dataset.model_name} pairs={len(pairs)}")

    qids = [p.question_id for p in pairs]
    train_ids, test_ids = split_question_ids(qids, args.train_fraction, args.seed)
    train_set = set(train_ids)
    train_pairs = [p for p in pairs if p.question_id in train_set]
    log(f"[steer] split seed={args.seed} train={len(train_ids)} test={len(test_ids)}")

    layers = captured_layers(dataset.samples)
    prim_layer, prim_depth = primary_layer(root, layers)
    log(f"[steer] captured layers {layers[0]}..{layers[-1]} "
        f"primary={prim_layer} (depth {prim_depth:.2f})")

    vectors = steering_vectors_all_layers(root, train_pairs, layers)
    log(f"[steer] built {len(vectors)} layer vectors "
        f"(L{prim_layer} norm={next(v.norm for v in vectors if v.layer == prim_layer):.3f})")

    bundle = SteeringVectorBundle(
        model=dataset.model,
        d_model=len(vectors[0].vector) if vectors else 0,
        seed=args.seed,
        train_fraction=args.train_fraction,
        primary_layer=prim_layer,
        primary_layer_depth=prim_depth,
        change_of_turn_positions=list(CHANGE_OF_TURN_POSITIONS),
        train_question_ids=train_ids,
        test_question_ids=test_ids,
        vectors=vectors,
    )

    out_path = (
        args.out_dir / "sesgo" / "steer" / dataset.model_name / "steering_vectors.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(bundle.to_dict(), out_path)
    log(f"\n[steer] wrote {out_path}")


if __name__ == "__main__":
    main()
