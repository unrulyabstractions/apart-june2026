"""Pick ONE ambiguous SESGO item whose committed outcome is likely to FLIP.

Run-by-path driver, STAGE 0 of the forking-paths study. A forking token only
exists when re-sampling can divert the FINAL outcome, so we pilot a few sampled
thinking decodes per candidate ambiguous item, parse each to its categorical
outcome, and rank items by the Shannon entropy of that pilot outcome distribution
(highest = most disagreement = best fork demo). The top item is written to
out/sesgo/forking/<MODEL>/selected_item.json for the capture driver to consume.

Usage:
  uv run python sesgo/forking/select_forking_item.py
  uv run python sesgo/forking/select_forking_item.py --model Qwen/Qwen3-0.6B \
      --n-pilot 12 --max-items 12 --categories gender,racism
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from sesgo.shard_output_paths import shard_out_dir  # noqa: E402
from src.common.file_io import save_json_atomic  # noqa: E402
from src.common.logging import log, log_header, log_section  # noqa: E402
from src.datasets.prompt import (  # noqa: E402
    SesgoPromptConfig,
    SesgoPromptDatasetGenerator,
)
from src.datasets.sesgo import load_items  # noqa: E402
from src.dynamics.forking_paths import (  # noqa: E402
    ForkOutcomeSet,
    pilot_item_entropy,
    rank_items_by_entropy,
)
from src.ternary_choice import TernaryChoiceRunner  # noqa: E402
from src.inference.backends import ModelBackend  # noqa: E402

from sesgo.forking.forking_item_io import selected_item_to_dict  # noqa: E402

_CANONICAL_STYLE = ("a)", "b)", "c)")


def parse_args() -> argparse.Namespace:
    """Parse CLI args for item selection."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HF model name")
    p.add_argument("--sesgo-dir", type=Path, default=Path("datasets/datasets/SESGO"))
    p.add_argument("--categories", default="gender", help="comma list (default: gender)")
    p.add_argument("--languages", default="es", help="comma list (default: es)")
    p.add_argument("--n-pilot", type=int, default=12, help="pilot decodes per item")
    p.add_argument("--max-items", type=int, default=10, help="candidate items to pilot")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--out-dir", type=Path, default=Path("out"))
    return p.parse_args()


def _ambiguous_prompts(args, generator) -> list:
    """Render the canonical single-permutation grid, keep AMBIGUOUS items only."""
    from src.datasets.sesgo import SesgoCategory  # local map for friendly names

    aliases = {"racism": SesgoCategory.RACISM, "xenophobia": SesgoCategory.XENOPHOBIA,
               "classism": SesgoCategory.CLASSISM, "gender": SesgoCategory.GENDER}
    cats = [aliases[c.strip()] for c in args.categories.split(",") if c.strip()]
    langs = tuple(c.strip() for c in args.languages.split(",") if c.strip())
    items = load_items(args.sesgo_dir, categories=cats, languages=langs)
    dataset = generator.generate(items, [])
    return [s for s in dataset.samples if s.context_condition == "ambig"][: args.max_items]


def main() -> None:
    """Pilot candidate items and persist the highest-outcome-entropy selection."""
    args = parse_args()
    log_header(f"SELECT FORKING ITEM ({args.model})")

    config = SesgoPromptConfig(
        name="forking", all_permutations=False, label_styles=[_CANONICAL_STYLE],
        include_no_scaffold=True,
    )
    prompts = _ambiguous_prompts(args, SesgoPromptDatasetGenerator(config))
    log(f"[select] piloting {len(prompts)} ambiguous prompts x {args.n_pilot} decodes")

    runner = TernaryChoiceRunner(model_name=args.model, backend=ModelBackend.HUGGINGFACE)
    outcome_set = ForkOutcomeSet()
    scored = [
        pilot_item_entropy(runner, s, outcome_set, args.n_pilot, args.max_new_tokens, args.temperature)
        for s in prompts
    ]
    ranked = rank_items_by_entropy(scored)
    by_idx = {s.sample_idx: s for s in prompts}

    log_section("pilot outcome entropy (nats), highest first")
    for it in ranked:
        log(f"  idx={it.sample_idx:>4} ent={it.entropy:.3f} hist={[round(x,2) for x in it.histogram]} (n={it.n_parsed})")

    best = ranked[0]
    chosen = by_idx[best.sample_idx]
    out_dir = shard_out_dir(args.out_dir, "forking", args.model.split("/")[-1], 0, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "selected_item.json"
    save_json_atomic(selected_item_to_dict(chosen, best, outcome_set), out_path)
    log(f"[select] chose idx={best.sample_idx} (entropy={best.entropy:.3f}); wrote {out_path}")


if __name__ == "__main__":
    main()
