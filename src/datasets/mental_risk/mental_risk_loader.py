"""Load the extracted MentalRiskES corpus into typed MentalRiskSubject objects."""

from __future__ import annotations

import re
from pathlib import Path

from src.common.file_io import load_json
from src.common.logging import log
from .mental_risk_disorder import Disorder
from .mental_risk_gold import normalize_subject_id, read_gold_csv
from .mental_risk_message import MentalRiskMessage
from .mental_risk_subject import MentalRiskSubject

# Trailing integer in a subject filename, used for numeric (not lexical) sorting
# so that subject2 sorts before subject10.
_NUM = re.compile(r"(\d+)")


def _subject_sort_key(path: Path) -> tuple[int, str]:
    match = _NUM.search(path.stem)
    return (int(match.group(1)) if match else 0, path.stem)


def _read_messages(path: Path) -> list[MentalRiskMessage]:
    """Parse a subjectN.json array into typed messages."""
    return [MentalRiskMessage.from_dict(record) for record in load_json(path)]


def _load_disorder(
    extracted_dir: Path, source: str, disorder: Disorder, limit: int | None
) -> list[MentalRiskSubject]:
    """Build every subject under one disorder directory, skipping if absent."""
    data_dir = extracted_dir / source / disorder.value / "data"
    if not data_dir.is_dir():
        log(f"[mental_risk] skipping missing disorder dir: {data_dir}")
        return []

    gold_path = extracted_dir / source / disorder.value / "gold" / "gold_label.csv"
    gold = read_gold_csv(gold_path) if gold_path.exists() else {}

    files = sorted(data_dir.glob("subject*.json"), key=_subject_sort_key)
    if limit is not None:
        files = files[:limit]

    subjects = []
    for path in files:
        subject_id = normalize_subject_id(path.stem)
        subjects.append(
            MentalRiskSubject(
                subject_id=subject_id,
                disorder=disorder,
                messages=_read_messages(path),
                labels=gold.get(subject_id, {}),
            )
        )
    return subjects


def load_subjects(
    extracted_dir: Path | str,
    disorders: list[Disorder] | None = None,
    source: str = "processed",
    limit: int | None = None,
) -> list[MentalRiskSubject]:
    """Load subjects from an extracted MentalRiskES tree.

    Args:
        extracted_dir: Root containing `<source>/<Disorder>/...`.
        disorders: Subset of disorders to load (default: all).
        source: Which variant to read ("processed" or "raw").
        limit: Cap on subjects per disorder (useful for smoke tests).

    Returns:
        Flat list of MentalRiskSubject across the requested disorders.
    """
    extracted_dir = Path(extracted_dir)
    disorders = disorders or list(Disorder)
    subjects: list[MentalRiskSubject] = []
    for disorder in disorders:
        subjects.extend(_load_disorder(extracted_dir, source, disorder, limit))
    return subjects
