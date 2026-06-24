"""Pick ONE ambiguous SESGO item whose committed outcome is likely to FLIP.

Run-by-path driver, STAGE 0 of the forking-paths study. A forking token only
exists when re-sampling can divert the FINAL outcome, so we pilot a few sampled
thinking decodes per candidate ambiguous item, parse each to its categorical
outcome, and rank items by the Shannon entropy of that pilot outcome distribution
(highest = most disagreement = best fork demo). The top item is written to
out/sesgo/forking/<MODEL><run-tag>/selected_item.json for the capture driver.

For a scaffold-vs-baseline comparison, `--force-question-id` selects the SAME item
without piloting (no GPU), `--scaffold` prepends a debiasing preamble, and
`--run-tag` suffixes the out subdir so the two conditions never collide.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from experiment.shard_output_paths import shard_out_dir  # noqa: E402
from src.common.file_io import save_json_atomic  # noqa: E402
from src.common.logging import log, log_header, log_section  # noqa: E402
from src.datasets.prompt import (  # noqa: E402
    Scaffold,
    SesgoPromptConfig,
    SesgoPromptDatasetGenerator,
    SesgoPromptSample,
)
from src.datasets.sesgo import SesgoCategory, load_items  # noqa: E402
from src.dynamics.forking_paths import (  # noqa: E402
    ForkOutcomeSet,
    ItemEntropy,
    pilot_item_entropy,
    rank_items_by_entropy,
)
from src.inference import ModelRunner  # noqa: E402
from src.inference.backends import ModelBackend  # noqa: E402

from experiment.forking.forking_item_io import selected_item_to_dict  # noqa: E402

_CANONICAL_STYLE = ("a)", "b)", "c)")
_CATEGORY_ALIASES = {
    "racism": SesgoCategory.RACISM, "xenophobia": SesgoCategory.XENOPHOBIA,
    "classism": SesgoCategory.CLASSISM, "gender": SesgoCategory.GENDER,
}


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
    p.add_argument("--thinking", action="store_true", help="pilot decodes in THINKING mode (enable_thinking for Qwen3.5 etc.)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--out-dir", type=Path, default=Path("out"))
    # Debiasing-scaffold condition: prepend this scaffold's preamble to the prompt
    # (matched by scaffold_id against sesgo.scaffolds). Default "" = no-scaffold.
    p.add_argument("--scaffold", default="", help="scaffold_id to prepend (default: none)")
    # Force the SAME item across conditions (skip the GPU pilot): pick the single
    # ambiguous prompt whose question_id matches, render it, and write it directly.
    p.add_argument("--force-question-id", default="", help="select this item without piloting")
    # Suffix the bare-model output subdir so conditions write disjoint paths.
    p.add_argument("--run-tag", default="", help="output subdir suffix (default: none)")
    return p.parse_args()


def get_scaffolds() -> list[Scaffold]:
    """Registered debiasing scaffolds. None are bundled yet (scaffolds are future work);
    `--scaffold ""` is the only supported value until a registry is added."""
    return []


def _resolve_scaffold(scaffold_id: str) -> Scaffold | None:
    """Look up the Scaffold whose scaffold_id matches, or None for no-scaffold."""
    if not scaffold_id:
        return None
    matched = next((s for s in get_scaffolds() if s.scaffold_id == scaffold_id), None)
    if matched is None:
        raise SystemExit(f"[select] unknown scaffold_id {scaffold_id!r}")
    return matched


def _ambiguous_prompts(args, generator, scaffold: Scaffold | None) -> list:
    """Render the canonical single-permutation grid, keep AMBIGUOUS items only.

    With a scaffold the prompt carries that scaffold's preamble (no no-scaffold
    baseline cell); without one it renders the plain no-scaffold prompt.
    """
    cats = [_CATEGORY_ALIASES[c.strip()] for c in args.categories.split(",") if c.strip()]
    langs = tuple(c.strip() for c in args.languages.split(",") if c.strip())
    items = load_items(args.sesgo_dir, categories=cats, languages=langs)
    dataset = generator.generate(items, [scaffold] if scaffold else [])
    return [s for s in dataset.samples if s.context_condition == "ambig"]


def _placeholder_entropy(sample: SesgoPromptSample, outcome_set: ForkOutcomeSet) -> ItemEntropy:
    """Empty pilot record for a forced (un-piloted) selection — no GPU needed."""
    return ItemEntropy(
        sample_idx=sample.sample_idx, question_id=sample.question_id,
        histogram=[0.0] * outcome_set.dim, entropy=0.0, n_parsed=0,
    )


def _select_chosen(args, prompts, scaffold, outcome_set):
    """Return (chosen sample, its ItemEntropy): forced lookup or piloted ranking."""
    if args.force_question_id:
        chosen = next((s for s in prompts if s.question_id == args.force_question_id), None)
        if chosen is None:
            raise SystemExit(f"[select] no ambiguous prompt with question_id {args.force_question_id!r}")
        log(f"[select] forced item q={chosen.question_id[:12]} scaffold={chosen.scaffold_id} (no pilot)")
        return chosen, _placeholder_entropy(chosen, outcome_set)

    candidates = prompts[: args.max_items]
    log(f"[select] piloting {len(candidates)} ambiguous prompts x {args.n_pilot} decodes")
    runner = ModelRunner(model_name=args.model, backend=ModelBackend.HUGGINGFACE)
    runner.force_thinking = getattr(args, "thinking", False)  # pilot decodes in the chosen mode
    scored = [
        pilot_item_entropy(runner, s, outcome_set, args.n_pilot, args.max_new_tokens, args.temperature)
        for s in candidates
    ]
    ranked = rank_items_by_entropy(scored)
    log_section("pilot outcome entropy (nats), highest first")
    for it in ranked:
        log(f"  idx={it.sample_idx:>4} ent={it.entropy:.3f} hist={[round(x,2) for x in it.histogram]} (n={it.n_parsed})")
    best = ranked[0]
    return next(s for s in candidates if s.sample_idx == best.sample_idx), best


def main() -> None:
    """Select the forking item (forced or piloted) and persist it for the capture driver."""
    args = parse_args()
    log_header(f"SELECT FORKING ITEM ({args.model})")

    scaffold = _resolve_scaffold(args.scaffold)
    config = SesgoPromptConfig(
        name="forking", all_permutations=False, label_styles=[_CANONICAL_STYLE],
        include_no_scaffold=scaffold is None,
    )
    prompts = _ambiguous_prompts(args, SesgoPromptDatasetGenerator(config), scaffold)
    outcome_set = ForkOutcomeSet()
    chosen, pilot = _select_chosen(args, prompts, scaffold, outcome_set)

    out_dir = shard_out_dir(args.out_dir, "forking", args.model.split("/")[-1] + args.run_tag, 0, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "selected_item.json"
    save_json_atomic(selected_item_to_dict(chosen, pilot, outcome_set), out_path)
    log(f"[select] chose idx={chosen.sample_idx} (entropy={pilot.entropy:.3f}); wrote {out_path}")


if __name__ == "__main__":
    main()
