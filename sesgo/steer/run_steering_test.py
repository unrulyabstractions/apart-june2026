"""Held-out causal steering test: abstention vs steering strength (run-by-path).

Reuses the seeded split saved in steering_vectors.json: on the TEST AMBIGUOUS
items (where UNKNOWN is the unbiased gold) with NO scaffold in the prompt, it
sweeps the steering strength alpha — including 0 (the unsteered baseline) and a
NEGATIVE control — adding ``alpha * v[L]`` to resid_post at the primary layer, and
measures abstention (UNKNOWN mass). It also scores the ACTUAL-scaffold prompt
unsteered (the behaviour +v aims to reproduce). The causal claim holds when
abstention rises monotonically with positive alpha on items the vector never saw.

All hooks are the existing inference-stack add-mode resid_post intervention; this
driver only wires the steered runner to the held-out readouts.

Usage:
  uv run python sesgo/steer/run_steering_test.py \
      out/sesgo/steer/Qwen3-0.6B/steering_vectors.json \
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
from src.inference.backends import ModelBackend  # noqa: E402
from src.steer import (  # noqa: E402
    SteeredTernaryChoiceRunner,
    SteeringTestResult,
    SteeringVectorBundle,
    build_pairs,
    build_readout,
    run_alpha_sweep,
    unsteered_reference,
)

_DEFAULT_ALPHAS = (-2.0, 0.0, 0.5, 1.0, 2.0, 4.0)
_AMBIG = "ambig"


def parse_args() -> argparse.Namespace:
    """Parse CLI args for the held-out steering test."""
    parser = argparse.ArgumentParser(description="Held-out SESGO steering test")
    parser.add_argument("vectors", type=Path, help="steering_vectors.json bundle")
    parser.add_argument("samples", type=Path, help="response_samples.json geometry")
    parser.add_argument(
        "--layer", type=int, default=None, help="steer layer (default: bundle primary)"
    )
    parser.add_argument(
        "--alphas", type=str, default=None, help="comma list (default: -2,0,0.5,1,2,4)"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="unit-normalize v (alpha is absolute magnitude); default scales raw v",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="cap held-out items (0 = all; for pilots)"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("out"), help="base output directory"
    )
    return parser.parse_args()


def _held_out_pairs(samples, test_ids: set[str]):
    """The TEST contrastive pairs whose context is ambiguous (UNKNOWN is gold)."""
    return [
        p
        for p in build_pairs(samples)
        if p.question_id in test_ids and p.context_condition == _AMBIG
    ]


def main() -> None:
    """Sweep alpha on held-out ambiguous no-scaffold items and save the result."""
    args = parse_args()
    log_header("HELD-OUT SESGO STEERING TEST")

    bundle = SteeringVectorBundle.from_json(args.vectors)
    dataset = GeometryDataset.from_json(args.samples)
    layer = args.layer if args.layer is not None else bundle.primary_layer
    direction = bundle.vector_for(layer).vector
    alphas = (
        tuple(float(a) for a in args.alphas.split(","))
        if args.alphas
        else _DEFAULT_ALPHAS
    )

    test_ids = set(bundle.test_question_ids)
    pairs = _held_out_pairs(dataset.samples, test_ids)
    if args.limit:
        pairs = pairs[: args.limit]
    # No-scaffold prompts drive the sweep; the scaffold prompts are the reference.
    no_scaf = [build_readout(p.no_scaffold) for p in pairs]
    scaf = [build_readout(p.scaffold) for p in pairs]
    log(f"[test] model={bundle.model} layer={layer} normalize={args.normalize} "
        f"held-out ambiguous items={len(pairs)}")

    # HuggingFace backend is forced for these local SESGO models (the only backend
    # the geometry/intervention path supports); SesgoQuerier pins it the same way.
    runner = SteeredTernaryChoiceRunner(
        model_name=bundle.model, backend=ModelBackend.HUGGINGFACE
    )

    reference = unsteered_reference(runner, scaf)
    log(f"[test] scaffold reference: unknown_prob={reference.mean_unknown_prob:.3f} "
        f"abstain_rate={reference.abstain_rate:.3f}")

    sweep = run_alpha_sweep(
        runner, no_scaf, layer, direction, list(alphas), args.normalize, log_fn=log
    )

    result = SteeringTestResult(
        model=bundle.model,
        layer=layer,
        normalize=args.normalize,
        seed=bundle.seed,
        n_test_pairs=len(test_ids),
        n_ambiguous_test_items=len(pairs),
        alphas=[float(a) for a in sorted(alphas)],
        sweep=sweep,
        scaffold_reference=reference,
    )

    out_path = (
        args.out_dir / "sesgo" / "steer" / dataset.model_name / "steering_test.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(result.to_dict(), out_path)
    log(f"\n[test] wrote {out_path}")


if __name__ == "__main__":
    main()
