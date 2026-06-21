"""Shared MentalRiskES subject resolution for the run-by-path drivers.

Every generator (the bare baseline one and the five-study one) needs the same
subject-source CLI surface and the same decrypt-or-load logic. This module owns
both so the drivers don't duplicate it: ``add_subject_source_args`` wires the
flags onto an argparse parser, and ``resolve_subjects`` turns the parsed args
into a list of MentalRiskSubject — decrypting the encrypted corpus into a temp
dir first when ``--corpus-dir`` is given.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from src.common.logging import log
from src.datasets.mental_risk import (
    Disorder,
    MentalRiskSubject,
    extract_corpus,
    load_subjects,
    resolve_password,
)

# Friendly CLI names -> corpus Disorder enum. The enum's own values ("Anxiety",
# "Depress", "ED") are on-disk directory names, not user-facing, so we expose
# readable aliases instead.
_DISORDER_ALIASES: dict[str, Disorder] = {
    "anxiety": Disorder.ANXIETY,
    "depression": Disorder.DEPRESSION,
    "eating_disorder": Disorder.EATING_DISORDER,
}


def add_subject_source_args(parser: argparse.ArgumentParser) -> None:
    """Register the subject-source + selection flags shared by the generators."""
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
        log(f"[subjects] decrypting {args.corpus_dir} -> {out_dir}")
        extract_corpus(args.corpus_dir, out_dir, password)
        extracted_dir = out_dir
    else:
        extracted_dir = args.extracted_dir
    return load_subjects(
        extracted_dir, disorders=disorders, source=args.source, limit=args.limit
    )
