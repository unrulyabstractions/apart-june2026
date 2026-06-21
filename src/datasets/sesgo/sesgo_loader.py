"""Load the SESGO benchmark's ambiguous-context prompts into SesgoItem objects.

Reads the per-(category, language) Excel prompt files, keeps only the AMBIGUOUS
rows, and unpacks each `answer_info` dict into role-labelled option texts using
the corpus' positional convention (ans0=OTHER, ans1=TARGET, ans2=UNKNOWN).
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pandas as pd

from src.common.logging import log
from .sesgo_category import SesgoCategory
from .sesgo_item import SesgoItem
from .sesgo_label import SesgoLabel


def _question_id(category: SesgoCategory, language: str, context: str) -> str:
    """Stable id grouping the neg/nonneg pair, invariant under the polarity flip.

    Built from (category, language, context) — the fields the two polarity
    phrasings share — so both rows of a pair collapse to the same id.
    """
    payload = f"{category.value}\x00{language}\x00{context}".encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _find_prompt_file(sesgo_dir: Path, category: SesgoCategory, language: str) -> Path | None:
    """Locate one prompt file, tolerating inconsistent filename casing.

    The corpus mixes cases (e.g. `prompts_genero_EN.xlsx` vs
    `prompts_racismo_en.xlsx`), so we match the stem/lang case-insensitively
    rather than assuming a fixed spelling.
    """
    wanted = f"prompts_{category.value}_{language}".lower()
    for path in (sesgo_dir / "prompts").glob("*.xlsx"):
        if path.stem.lower() == wanted:
            return path
    return None


def _row_to_item(row: pd.Series, category: SesgoCategory, language: str) -> SesgoItem:
    """Build one SesgoItem from an ambiguous prompt row."""
    # answer_info is a stringified Python dict; ans0/ans1/ans2 are fixed roles.
    info = ast.literal_eval(row["answer_info"])
    context = str(row["context"])
    return SesgoItem(
        question_id=_question_id(category, language, context),
        category=category,
        language=language,
        polarity=str(row["question_polarity"]),
        context=context,
        question=str(row["question"]),
        other_text=str(info["ans0"]),
        target_text=str(info["ans1"]),
        unknown_text=str(info["ans2"]),
        bbq=bool(row["bbq"]),
        gold_label=SesgoLabel.UNKNOWN,  # ambiguous gold is always "unknown"
    )


def _load_file(path: Path, category: SesgoCategory, language: str, limit: int | None) -> list[SesgoItem]:
    """Read one prompt file, keep AMBIGUOUS rows only, cap at `limit`."""
    df = pd.read_excel(path)
    ambiguous = df[df["context_condition"] == "ambig"]
    if limit is not None:
        ambiguous = ambiguous.head(limit)
    return [_row_to_item(row, category, language) for _, row in ambiguous.iterrows()]


def load_items(
    sesgo_dir: Path | str = "datasets/SESGO",
    categories: list[SesgoCategory] | None = None,
    languages: tuple[str, ...] = ("es", "en"),
    limit: int | None = None,
) -> list[SesgoItem]:
    """Load SESGO ambiguous-context items as typed SesgoItem objects.

    Args:
        sesgo_dir: Root containing the `prompts/` directory of `.xlsx` files.
        categories: Subset of bias categories to load (default: all).
        languages: Language codes to load (default: Spanish then English).
        limit: Cap on items per (category, language) — useful for smoke tests.

    Returns:
        Flat list of SesgoItem across the requested categories/languages.
    """
    sesgo_dir = Path(sesgo_dir)
    categories = categories or list(SesgoCategory)
    items: list[SesgoItem] = []
    for category in categories:
        for language in languages:
            path = _find_prompt_file(sesgo_dir, category, language)
            if path is None:
                log(f"[sesgo] skipping missing prompt file: {category.value}/{language}")
                continue
            items.extend(_load_file(path, category, language, limit))
    return items
