"""Build ALL FIVE SESGO prompt datasets (one per baseline study) in one run.

Run-by-path driver. Loads the SESGO ambiguous-context items once and renders five
complementary prompt grids, each isolating a single axis so the downstream studies
stay orthogonal:

  NON_THINKING_BASELINE - the bare reference grid. Per item: the canonical
               permutation (1) x one label style (1) x the lone no-scaffold
               condition = 1 prompt, scaffold_id None. The label-only run the
               cheap (do_greedy off) non-thinking pass is built for.
  STABILITY  - all superficial FORMAT variation, NO scaffolding. Per item: every
               role->position permutation (6) x every label style (3) x the lone
               no-scaffold condition = 18 prompts, scaffold_id always None. Probes
               how consistent the model's answer is across format-only rewrites of
               the SAME item.
  SELECTION  - NO format variation, ALL scaffolds. Per item: the canonical
               permutation (1) x one label style (1) x {no-scaffold + each of the
               4 scaffolds} = 5 prompts. Probes how every debiasing scaffold moves
               the answer with format held fixed (scaffold selection).
  DIVERGENCE - NO format variation, NO scaffold. Per item: the canonical
               permutation (1) x one label style (1) x the lone no-scaffold
               condition = 1 prompt, scaffold_id None. Structurally identical to
               NON_THINKING_BASELINE; the bare grid the divergence probe contrasts
               against the thinking draws.
  GEOMETRY   - NO format variation, the SINGLE interpretive_direction scaffold.
               Per item: canonical perm (1) x one style (1) x {no-scaffold + that
               one scaffold} = 2 prompts. The cheapest scaffold contrast, used for
               the geometry probe.

By default it generates the es-original studies (the default --languages is "es"
and the loader keeps only original rows); --limit / --categories / --languages are
optional caps for quick runs. Outputs:
  out/sesgo/baseline/prompt_dataset.json
  out/sesgo/stability/prompt_dataset.json
  out/sesgo/selection/prompt_dataset.json
  out/sesgo/divergence/prompt_dataset.json
  out/sesgo/geometry/prompt_dataset.json

OPT-IN full-data study: set GENERATE_ALL_DATA=1 to ALSO render a full_data grid
crossing ALL languages (es+en) x ALL origins (original+bbq-adapted) x {no-scaffold +
3 representative scaffolds} into a DISTINCT location, so it never clobbers the
es-original runs above:
  out/sesgo/full_data/prompt_dataset.json

Usage:
  uv run python sesgo/generate/generate_prompt_dataset.py
  GENERATE_ALL_DATA=1 uv run python sesgo/generate/generate_prompt_dataset.py
  uv run python sesgo/generate/generate_prompt_dataset.py \
      --categories racism,gender --languages es --limit 5
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from collections import Counter
from pathlib import Path

# Bootstrap the repo root onto sys.path so `from src... import ...` and
# `from experiment.scaffolds import ...` resolve regardless of cwd. From
# <repo>/sesgo/generate/x.py, parents[2] is the repo root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from experiment.scaffolds import get_full_data_scaffolds, get_scaffolds  # noqa: E402
from src.common.file_io import ensure_dir  # noqa: E402
from src.common.logging import log, log_header, log_section  # noqa: E402
from src.common.profiler import P  # noqa: E402
from src.datasets.prompt import (  # noqa: E402
    Scaffold,
    SesgoPromptConfig,
    SesgoPromptDataset,
    SesgoPromptDatasetGenerator,
    get_sesgo_label_styles,
)
from src.datasets.sesgo import SesgoCategory, SesgoItem, load_items  # noqa: E402

# Friendly CLI names → SesgoCategory enum. The enum's own values ("racismo",
# "genero", ...) are Spanish file stems used on disk, not user-facing, so we
# expose readable English aliases for the --categories flag instead.
_CATEGORY_ALIASES: dict[str, SesgoCategory] = {
    "racism": SesgoCategory.RACISM,
    "xenophobia": SesgoCategory.XENOPHOBIA,
    "classism": SesgoCategory.CLASSISM,
    "gender": SesgoCategory.GENDER,
}

# The single canonical label style the fixed-format studies hold constant. Only
# STABILITY varies the style; the other four pin it here so format is off.
_CANONICAL_LABEL_STYLE: tuple[str, str, str] = ("a)", "b)", "c)")

# The one scaffold GEOMETRY isolates from the full set.
_GEOMETRY_SCAFFOLD_ID = "interpretive_direction"

# OPT-IN full-data axes for the FULL_DATA study (env GENERATE_ALL_DATA=1). The
# default five studies stay es-original so the running es-original runs are never
# disturbed; this extra study crosses BOTH languages x BOTH origins x {no-scaffold
# + the three representative scaffolds}, written to a DISTINCT out/sesgo/full_data/.
_FULL_LANGUAGES: tuple[str, ...] = ("es", "en")
_FULL_ORIGINS: tuple[str, ...] = ("original", "bbq-adapted")
_FULL_DATA_NAME = "full_data"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for prompt-dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate all five SESGO baseline SesgoPromptDatasets",
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
        default="es",
        help="Comma list of language codes (default: es; English disabled for now)",
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
        help="Base output dir; each dataset lands at <out-dir>/sesgo/<name>/",
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


def geometry_scaffold() -> Scaffold:
    """The lone interpretive_direction scaffold GEOMETRY crosses over."""
    # Pull from get_scaffolds() rather than re-importing the constant so the two
    # stay in lockstep if the scaffold set is ever renamed/reordered.
    for scaffold in get_scaffolds():
        if scaffold.scaffold_id == _GEOMETRY_SCAFFOLD_ID:
            return scaffold
    raise SystemExit(f"Scaffold {_GEOMETRY_SCAFFOLD_ID!r} not found in get_scaffolds()")


def log_counts(dataset: SesgoPromptDataset, n_items: int) -> None:
    """Report a dataset's coverage: item/prompt totals and the scaffold split.

    Per-scaffold counts are the load-bearing check (None baseline vs each
    scaffold), so they print for every study even when there is only one bucket.
    """
    by_scaffold = Counter(s.scaffold_id or "(none)" for s in dataset.samples)
    log(f"  items:       {n_items}")
    log(f"  prompts:     {len(dataset.samples)}")
    log(f"  per item:    {len(dataset.samples) / n_items:.0f}" if n_items else "  per item: 0")
    log(f"  by scaffold: {dict(by_scaffold)}")


def build_and_save(
    config: SesgoPromptConfig,
    items: list[SesgoItem],
    scaffolds: list[Scaffold],
    out_dir: Path,
    description: str,
) -> None:
    """Render one study's grid, log its counts, and persist prompt_dataset.json."""
    with P(f"generate_{config.name}"):
        dataset = SesgoPromptDatasetGenerator(config).generate(items, scaffolds)
    log_section(f"{config.name} dataset ({description})")
    log_counts(dataset, len(items))
    # One fixed filename per study dir; no config-snapshot json is written.
    path = ensure_dir(out_dir / "sesgo" / config.name) / "prompt_dataset.json"
    dataset.save_as_json(path)
    log(f"[generate] wrote {path}")


def build_full_data(args: argparse.Namespace) -> None:
    """OPT-IN: render the FULL-DATA grid to a DISTINCT location.

    All languages (es+en) x all origins (original+bbq-adapted) x {no-scaffold + the
    three representative scaffolds}, one rendering per item (canonical perm, one
    label style) — the same cheap non-thinking baseline shape, just over the whole
    SESGO grid. Lands at out/sesgo/full_data/prompt_dataset.json so it never
    clobbers the es-original out/sesgo/baseline/ the running studies read.
    """
    log_section("full_data: loading FULL grid (all languages x all origins)")
    with P("load_items_full"):
        items = load_items(
            args.sesgo_dir,
            categories=resolve_categories(args.categories),
            languages=_FULL_LANGUAGES,
            limit=args.limit,
            origins=_FULL_ORIGINS,
        )
    log(
        f"[generate] full_data loaded {len(items)} items "
        f"(languages={list(_FULL_LANGUAGES)}, origins={list(_FULL_ORIGINS)})"
    )
    config = SesgoPromptConfig(
        name=_FULL_DATA_NAME,
        all_permutations=False,
        label_styles=[_CANONICAL_LABEL_STYLE],
        include_no_scaffold=True,
        categories=[c.value for c in (resolve_categories(args.categories) or list(SesgoCategory))],
        languages=list(_FULL_LANGUAGES),
        limit=args.limit,
    )
    build_and_save(
        config,
        items,
        get_full_data_scaffolds(),
        args.out_dir,
        "FULL grid (all langs x all origins), no-scaffold + 3 representative scaffolds",
    )


def main() -> None:
    """Load items once, render all five prompt grids, and persist each.

    When GENERATE_ALL_DATA=1 is set, ALSO render the opt-in full_data grid
    (all languages x all origins x {none + 3 scaffolds}) into a distinct location.
    """
    args = parse_args()
    log_header("GENERATE PROMPT DATASETS (sesgo: 5-study split)")

    # OPT-IN full-data study runs first (own load) so a failure there never leaves
    # the es-original studies half-written. Default behavior (env unset) is a no-op.
    if os.environ.get("GENERATE_ALL_DATA", "") not in ("", "0", "false", "False"):
        build_full_data(args)

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

    # Source provenance recorded on every config; the caller loads items above.
    cat_values = [c.value for c in categories] if categories else []
    lang_values = list(languages)

    def make_config(name: str, *, all_permutations: bool, label_styles) -> SesgoPromptConfig:
        """Config sharing the common source provenance across all five studies."""
        return SesgoPromptConfig(
            name=name,
            all_permutations=all_permutations,
            label_styles=label_styles,
            include_no_scaffold=True,
            categories=cat_values,
            languages=lang_values,
            limit=args.limit,
        )

    # NON_THINKING_BASELINE: 1 prompt/item — canonical perm, one style, no
    # scaffold (generate(items, []) leaves only the no-scaffold condition).
    build_and_save(
        make_config(
            "baseline",
            all_permutations=False,
            label_styles=[_CANONICAL_LABEL_STYLE],
        ),
        items,
        [],
        args.out_dir,
        "bare reference, no format/scaffold variation",
    )

    # STABILITY: 18 prompts/item — all 6 permutations x all 3 styles x the lone
    # no-scaffold condition (scaffold_id always None).
    build_and_save(
        make_config(
            "stability",
            all_permutations=True,
            label_styles=get_sesgo_label_styles(),
        ),
        items,
        [],
        args.out_dir,
        "format variation, no scaffold",
    )

    # SELECTION: 5 prompts/item — canonical perm + one style, crossed with the
    # no-scaffold baseline plus each of the 4 scaffolds.
    build_and_save(
        make_config(
            "selection",
            all_permutations=False,
            label_styles=[_CANONICAL_LABEL_STYLE],
        ),
        items,
        get_scaffolds(),
        args.out_dir,
        "all scaffolds, no format variation",
    )

    # DIVERGENCE: 1 prompt/item — canonical perm + one style, no scaffold
    # (generate(items, []) leaves only the no-scaffold condition). Structurally
    # identical to NON_THINKING_BASELINE.
    build_and_save(
        make_config(
            "divergence",
            all_permutations=False,
            label_styles=[_CANONICAL_LABEL_STYLE],
        ),
        items,
        [],
        args.out_dir,
        "no scaffold, no format variation",
    )

    # GEOMETRY: 2 prompts/item — canonical perm + one style, crossed with the
    # no-scaffold baseline plus the single interpretive_direction scaffold.
    build_and_save(
        make_config(
            "geometry",
            all_permutations=False,
            label_styles=[_CANONICAL_LABEL_STYLE],
        ),
        items,
        [geometry_scaffold()],
        args.out_dir,
        "interpretive_direction scaffold only, no format variation",
    )


if __name__ == "__main__":
    main()
