"""Collect the CHEAPEST STABILITY answer for every prompt into a SesgoDataset.

Run-by-path driver for the STABILITY half. Loads the stability prompt dataset
(all superficial FORMAT variation across label style + role->position
permutation + polarity, NO scaffolding) and records ONE answer per prompt: the
non-thinking, no-thinking, max-logprob-over-labels pick. That answer is the
teacher-forced 3-way softmax over the three option labels remapped to roles
(TARGET/OTHER/UNKNOWN), whose argmax is just the max-logprob label. It is the
cheapest readout the model offers: a single forward pass, no sampled reasoning
and no extra greedy decode.

The two query methods:
  - maxlogprob (default): do_greedy=False. non_thinking is ONLY the choose3
    argmax (max-logprob over the 3 labels) — the cheapest answer, one pass.
  - greedy: do_greedy=True. Adds an EXTRA temperature-0 skip-thinking decode on
    top of the teacher-forced readout (a second generation, strictly costlier).
Thinking (sampled free-form reasoning) is never run here.

On ambiguous SESGO items the gold answer is always UNKNOWN, so "accuracy" is the
fraction of predictions that abstain (predict UNKNOWN) rather than pick a group.
Because the stability grid holds the item fixed and varies only surface form, the
downstream question is how CONSISTENT that abstention is across the variations —
computed by visualize_stability_samples.py. Here we just log a one-line overall
accuracy summary as a sanity check.

Output lands at out/sesgo/stability/<MODEL>/samples.json (MODEL == bare name).

Usage:
  uv run python sesgo/baseline/collect_stability_samples.py
  uv run python sesgo/baseline/collect_stability_samples.py \
      out/sesgo/stability/prompt_dataset.json --model Qwen/Qwen3-0.6B \
      --method greedy --subsample 0.25
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from pathlib import Path

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves
# regardless of cwd. From <repo>/sesgo/baseline/x.py, parents[2] is the root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.common.file_io import load_json  # noqa: E402
from src.common.logging import log, log_header, log_section  # noqa: E402
from src.common.profiler import P  # noqa: E402
from src.datasets.prompt import (  # noqa: E402
    SesgoPromptConfig,
    SesgoPromptDataset,
    SesgoPromptSample,
)
from src.datasets.sesgo_eval import (  # noqa: E402
    SesgoDataset,
    SesgoQuerier,
    SesgoQueryConfig,
)

# Query methods -> whether the EXTRA greedy decode runs. maxlogprob is the cheap
# default: do_greedy=False, so non_thinking is just the choose3 argmax.
_METHOD_DO_GREEDY = {"maxlogprob": False, "greedy": True}


def load_prompt_dataset(path: Path, subsample: float) -> SesgoPromptDataset:
    """Load the prompt dataset, striding the RAW json before deserializing.

    The full dataset can be large; deserializing every prompt just to keep a
    fraction is the run's bottleneck. When subsample < 1 we json-load once, take
    an evenly-spaced stride over the raw sample dicts (so the slice still spans
    every permutation/label-style/category block, not just the first item), and
    build only the kept SesgoPromptSamples. The querier then runs with
    subsample=1.0.
    """
    if subsample >= 1.0:
        return SesgoPromptDataset.from_json(path)
    # load_json (not raw json) rejoins readable_text line-lists back to strings.
    data = load_json(Path(path))
    raw = data["samples"]
    n = max(1, math.ceil(len(raw) * subsample))
    stride = max(1, len(raw) // n)
    kept = [SesgoPromptSample.from_dict(d) for d in raw[::stride][:n]]
    return SesgoPromptDataset(
        dataset_id=data["dataset_id"],
        config=SesgoPromptConfig.from_dict(data["config"]),
        scaffold_ids=data.get("scaffold_ids", []),
        samples=kept,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for stability-sample collection."""
    parser = argparse.ArgumentParser(
        description="Collect the cheapest STABILITY answer into a SesgoDataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "prompt_dataset",
        type=Path,
        nargs="?",
        default=Path("out/sesgo/stability/prompt_dataset.json"),
        help="Path to the stability prompt_dataset.json (default: out/sesgo/stability/...)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="HF model name (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--method",
        choices=sorted(_METHOD_DO_GREEDY),
        default="maxlogprob",
        help=(
            "Answer method: 'maxlogprob' (default, do_greedy=False) is the "
            "cheapest answer (choose3 argmax = max-logprob over the 3 labels); "
            "'greedy' (do_greedy=True) adds an EXTRA temperature-0 decode"
        ),
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=1.0,
        help="Fraction of prompts to query, 0-1 (default: 1.0 == all)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out"),
        help="Base output dir; samples land at <out-dir>/sesgo/stability/<MODEL>/",
    )
    return parser.parse_args()


def _fmt(value: float | None) -> str:
    """Render an accuracy as a percentage, or n/a when undefined."""
    return f"{value:.1%}" if value is not None else "n/a"


def log_summary(dataset: SesgoDataset) -> None:
    """Report overall non-thinking abstention accuracy (the only level run).

    Accuracy = fraction of predictions that are UNKNOWN (the ambiguous gold).
    The cheap max-logprob pick is the argmax of the teacher-forced 3-way softmax,
    so every sample contributes (there is no parse-failure denominator).
    """
    nt = [
        s.correct_non_thinking
        for s in dataset.samples
        if s.predicted_non_thinking is not None
    ]
    nt_acc = sum(nt) / len(nt) if nt else None

    log_section("summary (accuracy = fraction predicted UNKNOWN)")
    log(f"  samples: {len(dataset.samples)}")
    log(f"  overall accuracy (non-thinking): {_fmt(nt_acc):>6} (n={len(nt)})")


def main() -> None:
    """Load stability prompts, take the cheap answer, and persist the dataset."""
    args = parse_args()
    do_greedy = _METHOD_DO_GREEDY[args.method]
    log_header(f"COLLECT STABILITY SAMPLES ({args.model}, method={args.method})")

    # Stride the raw json before deserializing when subsampling (fast path).
    prompt_dataset = load_prompt_dataset(args.prompt_dataset, args.subsample)
    log(f"[collect] loaded {len(prompt_dataset.samples)} prompts")

    # Cheapest answer: non-thinking teacher-forced readout only. With method
    # 'maxlogprob' (do_greedy=False) the non_thinking record is JUST the choose3
    # argmax — the max-logprob label — with no extra greedy decode and no
    # sampled thinking. Already subsampled at load, so the querier runs over all.
    config = SesgoQueryConfig(
        do_non_thinking=True,
        do_thinking=False,
        do_greedy=do_greedy,
        subsample=1.0,
    )
    with P("query_dataset"):
        dataset = SesgoQuerier(config).query_dataset(prompt_dataset, args.model)

    log_summary(dataset)

    # out/sesgo/stability/<MODEL>/samples.json, keyed by bare model name.
    out_dir = args.out_dir / "sesgo" / "stability" / dataset.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "samples.json"
    dataset.save_as_json(out_path)
    log(f"[collect] wrote {out_path}")


if __name__ == "__main__":
    main()
