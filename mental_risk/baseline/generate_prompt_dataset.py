"""Build a RiskPromptDataset from the MentalRiskES corpus.

Run-by-path driver for baseline task 1.a. Resolves subjects (either from an
already-extracted tree, or by decrypting the encrypted official corpus first),
renders the full prompt grid with RiskPromptGenerator, and writes the dataset to
out/mental_risk/. By default it generates EVERYTHING (all disorders, all
framings/languages/task types); --limit is an optional cap for quick runs.

Usage:
  uv run python mental_risk/baseline/generate_prompt_dataset.py
  uv run python mental_risk/baseline/generate_prompt_dataset.py \
      --corpus-dir datasets/corpusMentalRiskES --password-file secret.txt
  uv run python mental_risk/baseline/generate_prompt_dataset.py \
      --disorders anxiety,depression --limit 5
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile
from collections import Counter
from pathlib import Path

# Bootstrap the repo root onto sys.path so `from src... import ...` resolves
# regardless of cwd. From <repo>/mental_risk/baseline/x.py, parents[2] is root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.common.file_io import ensure_dir  # noqa: E402
from src.common.logging import log, log_header, log_section  # noqa: E402
from src.common.profiler import P  # noqa: E402
from src.datasets.mental_risk import (  # noqa: E402
    Disorder,
    MentalRiskSubject,
    extract_corpus,
    load_subjects,
    resolve_password,
)
from src.datasets.prompt import (  # noqa: E402
    RiskPromptConfig,
    RiskPromptDataset,
    RiskPromptGenerator,
)

# Friendly CLI names → corpus Disorder enum. The enum's own values ("Anxiety",
# "Depress", "ED") are on-disk directory names, not user-facing, so we expose
# readable aliases instead.
_DISORDER_ALIASES: dict[str, Disorder] = {
    "anxiety": Disorder.ANXIETY,
    "depression": Disorder.DEPRESSION,
    "eating_disorder": Disorder.EATING_DISORDER,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for prompt-dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate a RiskPromptDataset from MentalRiskES",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Subject source: either a pre-extracted tree, or the encrypted corpus.
    parser.add_argument(
        "--extracted-dir",
        type=Path,
        default=Path("tests/fixtures/mental_risk/extracted"),
        help="Path to an already-extracted corpus (default: synthetic fixture)",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="Path to the ENCRYPTED corpus; if set, decrypt before loading",
    )
    parser.add_argument(
        "--password-file",
        type=Path,
        default=None,
        help="File holding the archive password (env MENTALRISK_ZIP_PASSWORD wins)",
    )
    # Selection knobs.
    parser.add_argument(
        "--source",
        choices=["processed", "raw"],
        default="processed",
        help="Which corpus rendering to load (default: processed)",
    )
    parser.add_argument(
        "--disorders",
        default=None,
        help="Comma list of anxiety/depression/eating_disorder (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="OPTIONAL cap on subjects PER disorder (default: all)",
    )
    # Output.
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out"),
        help="Base output directory; dataset lands at <out-dir>/mental_risk/",
    )
    return parser.parse_args()


def resolve_disorders(spec: str | None) -> list[Disorder] | None:
    """Map a comma list of friendly names to Disorder enums (None == all)."""
    if spec is None:
        return None
    names = [n.strip().lower() for n in spec.split(",") if n.strip()]
    unknown = [n for n in names if n not in _DISORDER_ALIASES]
    if unknown:
        raise SystemExit(
            f"Unknown disorder(s) {unknown}; choose from {sorted(_DISORDER_ALIASES)}"
        )
    return [_DISORDER_ALIASES[n] for n in names]


def resolve_subjects(args: argparse.Namespace) -> list[MentalRiskSubject]:
    """Load subjects, decrypting the encrypted corpus first when requested.

    Decryption goes to a temp dir because the extracted plaintext is sensitive
    and only needed for this run; the pre-extracted path is used as-is.
    """
    disorders = resolve_disorders(args.disorders)
    if args.corpus_dir is not None:
        password = resolve_password(args.password_file)
        out_dir = Path(tempfile.mkdtemp(prefix="mentalriskes_"))
        log(f"[generate] decrypting {args.corpus_dir} -> {out_dir}")
        extract_corpus(args.corpus_dir, out_dir, password)
        extracted_dir = out_dir
    else:
        extracted_dir = args.extracted_dir
    return load_subjects(
        extracted_dir,
        disorders=disorders,
        source=args.source,
        limit=args.limit,
    )


def log_axis_counts(dataset: RiskPromptDataset) -> None:
    """Report per-axis prompt counts so the grid coverage is visible."""
    tasks = Counter(s.task_type.value for s in dataset.samples)
    langs = Counter(s.language for s in dataset.samples)
    framings = Counter(s.framing for s in dataset.samples)
    log(f"  by task:    {dict(tasks)}")
    log(f"  by lang:    {dict(langs)}")
    log(f"  by framing: {dict(framings)}")


def main() -> None:
    """Resolve subjects, render the prompt grid, and persist the dataset."""
    args = parse_args()
    log_header("GENERATE PROMPT DATASET (mental_risk)")

    with P("resolve_subjects"):
        subjects = resolve_subjects(args)
    log(f"[generate] loaded {len(subjects)} subjects from source={args.source}")

    config = RiskPromptConfig(name="mental_risk")
    with P("generate_prompts"):
        dataset = RiskPromptGenerator(config).generate(subjects)

    log_section("dataset")
    log(f"  prompts: {len(dataset.samples)}")
    log_axis_counts(dataset)

    out_dir = ensure_dir(args.out_dir / "mental_risk")
    dataset_path = out_dir / "prompt_dataset.json"
    dataset.save_as_json(dataset_path)
    log(f"[generate] wrote {dataset_path}")


if __name__ == "__main__":
    main()
