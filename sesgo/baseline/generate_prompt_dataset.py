"""Build BOTH SESGO prompt datasets (STABILITY + GEOMETRY) in one run.

Run-by-path driver for baseline task 1. Loads the SESGO ambiguous-context items
once and renders two complementary prompt grids, each isolating a single axis so
the two downstream studies stay orthogonal:

  STABILITY  - all superficial FORMAT variation, NO scaffolding. Per item: every
               role->position permutation (6) x every label style (3) x the lone
               no-scaffold condition = 18 prompts, scaffold_id always None. Probes
               how consistent the model's answer is across format-only rewrites of
               the SAME item.
  GEOMETRY   - NO format variation, VARYING scaffold. Per item: the canonical
               permutation (1) x one label style (1) x {no-scaffold + each of the
               4 scaffolds} = 5 prompts. Probes how each debiasing scaffold moves
               the answer with format held fixed.

By default it generates EVERYTHING (all categories, both languages); --limit is
an optional cap for quick runs. Outputs:
  out/sesgo/stability/prompt_dataset.json
  out/sesgo/geometry/prompt_dataset.json

Usage:
  uv run python sesgo/baseline/generate_prompt_dataset.py
  uv run python sesgo/baseline/generate_prompt_dataset.py \
      --categories racism,gender --languages es --limit 5
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

# The single canonical label style the GEOMETRY grid holds fixed (format off).
_CANONICAL_LABEL_STYLE: tuple[str, str, str] = ("a)", "b)", "c)")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for prompt-dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate the STABILITY + GEOMETRY SesgoPromptDatasets",
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
    # Output.
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out"),
        help="Base output dir; datasets land at <out-dir>/sesgo/{stability,geometry}/",
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


def log_stability_counts(dataset: SesgoPromptDataset) -> None:
    """Report STABILITY coverage: scaffold (all None) and the format axes."""
    scaffolds = Counter(s.scaffold_id or "(none)" for s in dataset.samples)
    styles = Counter(s.label_style for s in dataset.samples)
    perms = Counter(tuple(s.position_labels) for s in dataset.samples)
    log(f"  by scaffold:    {dict(scaffolds)}")
    log(f"  by label_style: {dict(styles)}")
    log(f"  distinct permutations: {len(perms)}")


def log_geometry_counts(dataset: SesgoPromptDataset) -> None:
    """Report GEOMETRY coverage: the scaffold axis (None baseline + each)."""
    scaffolds = Counter(s.scaffold_id or "(none)" for s in dataset.samples)
    styles = Counter(s.label_style for s in dataset.samples)
    log(f"  by scaffold:    {dict(scaffolds)}")
    log(f"  by label_style: {dict(styles)}")


def main() -> None:
    """Load items once, render both prompt grids, and persist each."""
    args = parse_args()
    log_header("GENERATE PROMPT DATASETS (sesgo: stability + geometry)")

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

    cat_values = [c.value for c in categories] if categories else []

    # STABILITY: all format variation, no scaffolding (generate(items, []) leaves
    # only the no-scaffold condition -> scaffold_id always None).
    stability_config = SesgoPromptConfig(
        name="stability",
        all_permutations=True,
        label_styles=[("a)", "b)", "c)"), ("1)", "2)", "3)"), ("x)", "y)", "z)")],
        include_no_scaffold=True,
        categories=cat_values,
        languages=list(languages),
        limit=args.limit,
    )
    with P("generate_stability"):
        stability = SesgoPromptDatasetGenerator(stability_config).generate(items, [])

    log_section("stability dataset (format variation, no scaffold)")
    log(f"  items:   {len(items)}")
    log(f"  prompts: {len(stability.samples)}")
    log_stability_counts(stability)
    stability_dir = ensure_dir(args.out_dir / "sesgo" / "stability")
    stability_path = stability_dir / "prompt_dataset.json"
    stability.save_as_json(stability_path)
    log(f"[generate] wrote {stability_path}")

    # GEOMETRY: no format variation, varying scaffold (canonical perm + one style,
    # crossed with the no-scaffold baseline plus each scaffold).
    geometry_config = SesgoPromptConfig(
        name="geometry",
        all_permutations=False,
        label_styles=[_CANONICAL_LABEL_STYLE],
        include_no_scaffold=True,
        categories=cat_values,
        languages=list(languages),
        limit=args.limit,
    )
    with P("generate_geometry"):
        geometry = SesgoPromptDatasetGenerator(geometry_config).generate(
            items, get_scaffolds()
        )

    log_section("geometry dataset (scaffold variation, no format)")
    log(f"  items:   {len(items)}")
    log(f"  prompts: {len(geometry.samples)}")
    log_geometry_counts(geometry)
    geometry_dir = ensure_dir(args.out_dir / "sesgo" / "geometry")
    geometry_path = geometry_dir / "prompt_dataset.json"
    geometry.save_as_json(geometry_path)
    log(f"[generate] wrote {geometry_path}")


if __name__ == "__main__":
    main()
