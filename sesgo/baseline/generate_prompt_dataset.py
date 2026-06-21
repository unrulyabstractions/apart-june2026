"""Build a SesgoPromptDataset from the SESGO ambiguous-bias corpus.

Run-by-path driver for baseline task 1.a (plus the part-2 scaffolding axis).
Loads the SESGO ambiguous-context items, renders the full prompt grid with
SesgoPromptDatasetGenerator, and writes the dataset to out/sesgo/. By default it
generates EVERYTHING (all categories, both languages, all four scaffolds plus
the no-scaffold baseline); --limit is an optional cap for quick runs. The
generator emits, per item, both the WITH-each-scaffold conditions and the
WITHOUT (scaffold_id=None) baseline — that pairing is the with/without
comparison set this study reports on.

Usage:
  uv run python sesgo/baseline/generate_prompt_dataset.py
  uv run python sesgo/baseline/generate_prompt_dataset.py \
      --categories racism,gender --languages es --limit 5
  uv run python sesgo/baseline/generate_prompt_dataset.py --no-scaffolds
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter
from pathlib import Path

# Bootstrap the repo root onto sys.path so `from src... import ...` and
# `from sesgo.scaffolds import ...` resolve regardless of cwd. From
# <repo>/sesgo/baseline/x.py, parents[2] is the repo root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from sesgo.scaffolds import get_scaffolds  # noqa: E402
from src.common.file_io import ensure_dir  # noqa: E402
from src.common.logging import log, log_header, log_section  # noqa: E402
from src.common.profiler import P  # noqa: E402
from src.datasets.prompt import (  # noqa: E402
    SesgoPromptConfig,
    SesgoPromptDataset,
    SesgoPromptDatasetGenerator,
)
from src.datasets.sesgo import SesgoCategory, load_items  # noqa: E402

# Friendly CLI names → SesgoCategory enum. The enum's own values ("racismo",
# "genero", ...) are Spanish file stems used on disk, not user-facing, so we
# expose readable English aliases for the --categories flag instead.
_CATEGORY_ALIASES: dict[str, SesgoCategory] = {
    "racism": SesgoCategory.RACISM,
    "xenophobia": SesgoCategory.XENOPHOBIA,
    "classism": SesgoCategory.CLASSISM,
    "gender": SesgoCategory.GENDER,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for prompt-dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate a SesgoPromptDataset from the SESGO corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Source location and selection knobs.
    parser.add_argument(
        "--sesgo-dir",
        type=Path,
        default=Path("datasets/SESGO"),
        help="Root holding the prompts/ directory of .xlsx files",
    )
    parser.add_argument(
        "--categories",
        default=None,
        help="Comma list of racism/xenophobia/classism/gender (default: all)",
    )
    parser.add_argument(
        "--languages",
        default="es,en",
        help="Comma list of language codes (default: es,en == all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="OPTIONAL cap on items PER (category, language) (default: all)",
    )
    # Scaffolding axis. The generator always emits the no-scaffold baseline;
    # --no-scaffolds drops the WITH-scaffold conditions so only that baseline
    # remains (useful to render the raw, unguided grid on its own).
    parser.add_argument(
        "--no-scaffolds",
        action="store_true",
        help="Generate ONLY the no-scaffold baseline (skip every scaffold)",
    )
    # Output.
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out"),
        help="Base output directory; dataset lands at <out-dir>/sesgo/",
    )
    return parser.parse_args()


def resolve_categories(spec: str | None) -> list[SesgoCategory] | None:
    """Map a comma list of friendly names to SesgoCategory enums (None == all)."""
    if spec is None:
        return None
    names = [n.strip().lower() for n in spec.split(",") if n.strip()]
    unknown = [n for n in names if n not in _CATEGORY_ALIASES]
    if unknown:
        raise SystemExit(
            f"Unknown category(ies) {unknown}; choose from {sorted(_CATEGORY_ALIASES)}"
        )
    return [_CATEGORY_ALIASES[n] for n in names]


def resolve_languages(spec: str) -> tuple[str, ...]:
    """Parse the comma-separated --languages flag into a tuple of codes."""
    return tuple(c.strip().lower() for c in spec.split(",") if c.strip())


def log_grid_counts(dataset: SesgoPromptDataset) -> None:
    """Report per-axis prompt counts so the grid coverage is visible.

    The scaffold axis includes the None baseline (rendered as "(none)") so the
    with/without comparison size is legible at a glance.
    """
    scaffolds = Counter(s.scaffold_id or "(none)" for s in dataset.samples)
    categories = Counter(s.bias_category for s in dataset.samples)
    langs = Counter(s.language for s in dataset.samples)
    polarities = Counter(s.question_polarity for s in dataset.samples)
    log(f"  by scaffold:  {dict(scaffolds)}")
    log(f"  by category:  {dict(categories)}")
    log(f"  by language:  {dict(langs)}")
    log(f"  by polarity:  {dict(polarities)}")


def main() -> None:
    """Load items, render the scaffold-crossed prompt grid, and persist it."""
    args = parse_args()
    log_header("GENERATE PROMPT DATASET (sesgo)")

    categories = resolve_categories(args.categories)
    languages = resolve_languages(args.languages)
    with P("load_items"):
        items = load_items(
            args.sesgo_dir,
            categories=categories,
            languages=languages,
            limit=args.limit,
        )
    log(f"[generate] loaded {len(items)} items (languages={list(languages)})")

    # By default cross every concrete debiasing scaffold against the always-on
    # no-scaffold baseline; --no-scaffolds renders only that baseline.
    scaffolds = [] if args.no_scaffolds else get_scaffolds()
    config = SesgoPromptConfig(
        name="sesgo",
        categories=[c.value for c in categories] if categories else [],
        languages=list(languages),
        limit=args.limit,
    )
    with P("generate_prompts"):
        dataset = SesgoPromptDatasetGenerator(config).generate(items, scaffolds)

    log_section("dataset")
    log(f"  items:   {len(items)}")
    log(f"  prompts: {len(dataset.samples)}")
    log_grid_counts(dataset)

    out_dir = ensure_dir(args.out_dir / "sesgo")
    dataset_path = out_dir / "prompt_dataset.json"
    dataset.save_as_json(dataset_path)
    log(f"[generate] wrote {dataset_path}")


if __name__ == "__main__":
    main()
