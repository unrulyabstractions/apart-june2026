"""Build ALL FIVE mental_risk prompt datasets (one per study) in one run.

Run-by-path driver. Resolves the MentalRiskES subjects once (from an extracted
tree, or by decrypting the encrypted corpus first) and renders five complementary
prompt grids, each isolating a single axis so the downstream studies stay
orthogonal. It is the risk analogue of sesgo/generate/generate_prompt_dataset.py;
the SESGO "scaffold" axis maps to the risk FRAMING axis (see mental_risk/
scaffolds_risk.py for why).

  BASELINE   - the bare reference grid. Per subject: one framing (the canonical
               first) x one CATEGORIZE label style x no order flip = 1 prompt. The
               cheap non-thinking reference the other studies contrast against.
  STABILITY  - all superficial FORMAT variation, ONE framing. Per subject: every
               label style x every order flip x {SCORE over both scale directions,
               CATEGORIZE} for the canonical framing. Probes how consistent the
               risk answer is across format-only rewrites of the SAME subject.
  SELECTION  - ALL framings, canonical format. Per subject: one CATEGORIZE prompt
               per framing (at_risk_of/suffering/safe/intervene). Probes which
               framing best tracks the gold risk (the framing-selection analogue
               of SESGO scaffold selection — note there is NO no-op baseline
               framing, so framings are ranked against gold, not a baseline).
  DIVERGENCE - ONE framing, canonical format, both task types. Per subject: the
               canonical framing x {SCORE (both scale dirs), CATEGORIZE}. The bare
               grid the divergence probe contrasts against its thinking draws.
  GEOMETRY   - ALL framings, canonical CATEGORIZE format. Per subject: one
               CATEGORIZE prompt per framing. The framing contrast used for the
               residual-geometry probe.

By default it generates EVERYTHING (all disorders); --limit / --disorders are
optional caps for quick runs. Outputs:
  out/mental_risk/baseline/prompt_dataset.json
  out/mental_risk/stability/prompt_dataset.json
  out/mental_risk/selection/prompt_dataset.json
  out/mental_risk/divergence/prompt_dataset.json
  out/mental_risk/geometry/prompt_dataset.json

Usage:
  uv run python mental_risk/generate/generate_risk_prompt_datasets.py
  uv run python mental_risk/generate/generate_risk_prompt_datasets.py \
      --corpus-dir datasets/corpusMentalRiskES --password-file secret.txt
  uv run python mental_risk/generate/generate_risk_prompt_datasets.py \
      --disorders anxiety,depression --limit 5
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter
from pathlib import Path

# Bootstrap the repo root onto sys.path so `from src... import ...` and
# `from mental_risk.scaffolds_risk import ...` resolve regardless of cwd. From
# <repo>/mental_risk/generate/x.py, parents[2] is the repo root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from mental_risk.scaffolds_risk import framing_keys  # noqa: E402
from mental_risk.subject_resolution import (  # noqa: E402
    add_subject_source_args,
    resolve_subjects,
)
from src.common.file_io import ensure_dir  # noqa: E402
from src.common.logging import log, log_header, log_section  # noqa: E402
from src.common.profiler import P  # noqa: E402
from src.datasets.mental_risk import MentalRiskSubject  # noqa: E402
from src.datasets.prompt import (  # noqa: E402
    RiskPromptConfig,
    RiskPromptDataset,
    RiskPromptGenerator,
    RiskTaskType,
    get_risk_label_styles,
)

# The single canonical format the fixed-format studies hold constant: first label
# style, positive option first, English. Only STABILITY varies these.
_CANONICAL_LABEL_STYLE: tuple[str, str] = get_risk_label_styles()[0]
_CANONICAL_LANG = "en"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for prompt-dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate all five mental_risk study RiskPromptDatasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_subject_source_args(parser)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out"),
        help="Base output dir; each dataset lands at <out-dir>/mental_risk/<name>/",
    )
    return parser.parse_args()


def make_config(name: str, *, framings, label_styles, languages, task_types, order_flips, scale_highs) -> RiskPromptConfig:
    """A RiskPromptConfig pinned to the per-study axis slices."""
    return RiskPromptConfig(
        name=name,
        framings=framings,
        label_styles=label_styles,
        languages=languages,
        task_types=task_types,
        order_flips=order_flips,
        scale_highs=scale_highs,
    )


def log_counts(dataset: RiskPromptDataset, n_subjects: int) -> None:
    """Report coverage: subject/prompt totals and the framing/task split."""
    by_framing = Counter(s.framing for s in dataset.samples)
    by_task = Counter(s.task_type.value for s in dataset.samples)
    log(f"  subjects:   {n_subjects}")
    log(f"  prompts:    {len(dataset.samples)}")
    log(f"  per subj:   {len(dataset.samples) / n_subjects:.0f}" if n_subjects else "  per subj: 0")
    log(f"  by framing: {dict(by_framing)}")
    log(f"  by task:    {dict(by_task)}")


def build_and_save(
    config: RiskPromptConfig,
    subjects: list[MentalRiskSubject],
    out_dir: Path,
    description: str,
) -> None:
    """Render one study's grid, log its counts, and persist prompt_dataset.json."""
    with P(f"generate_{config.name}"):
        dataset = RiskPromptGenerator(config).generate(subjects)
    log_section(f"{config.name} dataset ({description})")
    log_counts(dataset, len(subjects))
    path = ensure_dir(out_dir / "mental_risk" / config.name) / "prompt_dataset.json"
    dataset.save_as_json(path)
    log(f"[generate] wrote {path}")


def main() -> None:
    """Resolve subjects once, render all five prompt grids, and persist each."""
    args = parse_args()
    log_header("GENERATE PROMPT DATASETS (mental_risk: 5-study split)")

    with P("resolve_subjects"):
        subjects = resolve_subjects(args)
    log(f"[generate] loaded {len(subjects)} subjects from source={args.source}")

    canonical = [framing_keys()[0]]
    all_framings = framing_keys()
    one_style = [_CANONICAL_LABEL_STYLE]
    one_lang = [_CANONICAL_LANG]
    no_flip = [False]

    # BASELINE: 1 prompt/subject — canonical framing, one CATEGORIZE format.
    build_and_save(
        make_config("baseline", framings=canonical, label_styles=one_style,
                    languages=one_lang, task_types=[RiskTaskType.CATEGORIZE],
                    order_flips=no_flip, scale_highs=[True]),
        subjects, args.out_dir, "bare reference, no format/framing variation",
    )

    # STABILITY: all format variation, one framing, both task types.
    build_and_save(
        make_config("stability", framings=canonical, label_styles=get_risk_label_styles(),
                    languages=one_lang, task_types=list(RiskTaskType),
                    order_flips=[False, True], scale_highs=[True, False]),
        subjects, args.out_dir, "format variation, one framing",
    )

    # SELECTION: all framings, canonical CATEGORIZE format.
    build_and_save(
        make_config("selection", framings=all_framings, label_styles=one_style,
                    languages=one_lang, task_types=[RiskTaskType.CATEGORIZE],
                    order_flips=no_flip, scale_highs=[True]),
        subjects, args.out_dir, "all framings, no format variation",
    )

    # DIVERGENCE: one framing, canonical format, both task types (for thinking).
    build_and_save(
        make_config("divergence", framings=canonical, label_styles=one_style,
                    languages=one_lang, task_types=list(RiskTaskType),
                    order_flips=no_flip, scale_highs=[True]),
        subjects, args.out_dir, "one framing, no format variation",
    )

    # GEOMETRY: all framings, canonical CATEGORIZE format (the framing contrast).
    build_and_save(
        make_config("geometry", framings=all_framings, label_styles=one_style,
                    languages=one_lang, task_types=[RiskTaskType.CATEGORIZE],
                    order_flips=no_flip, scale_highs=[True]),
        subjects, args.out_dir, "all framings, no format variation",
    )


if __name__ == "__main__":
    main()
