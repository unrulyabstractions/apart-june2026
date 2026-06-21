"""Readers for MentalRiskES gold-label files (CSV and whitespace TXT).

Both readers key subjects by a normalized id so that bare numbers ("103"),
stems ("subject103"), and filenames ("subject103.json") all collapse to the
canonical "subject103" used throughout the loader.
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.common.logging import log


def normalize_subject_id(raw: str) -> str:
    """Canonicalize a subject identifier to the 'subject<N>' form.

    Accepts bare numbers, stems, or filenames; strips any extension and adds the
    'subject' prefix when the value is purely numeric.
    """
    sid = Path(raw.strip()).stem  # drop ".json" / ".csv" if present
    return f"subject{sid}" if sid.isdigit() else sid


def _parse_floats(header: list[str], row: list[str]) -> dict[str, float]:
    """Pair non-empty header columns with parseable float cells."""
    labels: dict[str, float] = {}
    for col, cell in zip(header, row):
        col, cell = col.strip(), cell.strip()
        if not col or not cell:
            continue
        try:
            labels[col] = float(cell)
        except ValueError:
            # Non-numeric cells (e.g. free-text notes) are not risk signals.
            continue
    return labels


def read_gold_csv(path: Path) -> dict[str, dict[str, float]]:
    """Read a gold_label.csv into {subject_id: {column: value}}.

    The header row supplies column names (casing is taken verbatim from the
    file); the first column is the subject id and is consumed as the key.
    """
    rows = list(csv.reader(Path(path).open(encoding="utf-8")))
    if not rows:
        return {}
    header = rows[0]
    out: dict[str, dict[str, float]] = {}
    for row in rows[1:]:
        if not row or not row[0].strip():
            continue
        out[normalize_subject_id(row[0])] = _parse_floats(header[1:], row[1:])
    return out


def read_gold_txt(path: Path) -> dict[str, dict[str, float]]:
    """Read a headerless whitespace/comma gold file into {subject_id: {...}}.

    Assumption: column 0 is the subject id and column 1 is its single risk value.
    Integer-looking values (0/1) are treated as the binary `bs` flag; anything
    with a fractional part is treated as the annotator fraction `rbs`.
    """
    out: dict[str, dict[str, float]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = [p for p in line.replace(",", " ").split() if p]
        if len(parts) < 2:
            continue
        try:
            value = float(parts[1])
        except ValueError:
            log(f"[mental_risk] skipping unparseable gold line: {line!r}")
            continue
        key = "bs" if value.is_integer() else "rbs"
        out[normalize_subject_id(parts[0])] = {key: value}
    return out
