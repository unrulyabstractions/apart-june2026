"""Load SESGO benchmark prompts into SesgoItem objects.

Reads the per-(category, language) Excel prompt files and unpacks each
`answer_info` dict into role-labelled option texts using the corpus' positional
convention (ans0=OTHER, ans1=TARGET, ans2=UNKNOWN). Both context conditions are
kept: AMBIGUOUS items (gold = UNKNOWN, no evidence) and DISAMBIGUATED items
(gold = the role named by the `label` index). For now only Spanish ORIGINAL rows
are loaded — English and BBQ-adapted (bbq==True) rows are dropped at read time.
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


def _gold_label(row: pd.Series, condition: str) -> SesgoLabel:
    """Ground-truth role per condition: ambiguous -> UNKNOWN; else the `label`.

    AMBIGUOUS contexts give no evidence, so the correct answer is always UNKNOWN
    (abstention). DISAMBIGUATED contexts name the ground-truth role through the
    `label` index (0=other, 1=target, 2=unknown).
    """
    if condition == "ambig":
        return SesgoLabel.UNKNOWN
    return SesgoLabel.from_answer_index(row["label"])


def _row_to_item(row: pd.Series, category: SesgoCategory, language: str) -> SesgoItem:
    """Build one SesgoItem from a prompt row (either context condition)."""
    # answer_info is a stringified Python dict; ans0/ans1/ans2 are fixed roles.
    info = ast.literal_eval(row["answer_info"])
    context = str(row["context"])
    condition = str(row["context_condition"])
    return SesgoItem(
        question_id=_question_id(category, language, context),
        category=category,
        language=language,
        polarity=str(row["question_polarity"]),
        context_condition=condition,
        context=context,
        question=str(row["question"]),
        other_text=str(info["ans0"]),
        target_text=str(info["ans1"]),
        unknown_text=str(info["ans2"]),
        bbq=bool(row["bbq"]),
        gold_label=_gold_label(row, condition),
    )


def _load_file(path: Path, category: SesgoCategory, language: str, limit: int | None) -> list[SesgoItem]:
    """Read one prompt file, keep ORIGINAL (bbq==False) rows, cap at `limit`.

    Both context conditions (ambig + disambig) are retained; only the BBQ-adapted
    provenance is filtered out so the loaded set is Spanish-original SESGO.
    """
    df = pd.read_excel(path)
    original = df[~df["bbq"].astype(bool)]
    if limit is not None:
        original = original.head(limit)
    return [_row_to_item(row, category, language) for _, row in original.iterrows()]


def load_items(
    sesgo_dir: Path | str = "datasets/SESGO",
    categories: list[SesgoCategory] | None = None,
    languages: tuple[str, ...] = ("es",),
    limit: int | None = None,
) -> list[SesgoItem]:
    """Load SESGO items (both context conditions) as typed SesgoItem objects.

    Only ORIGINAL (bbq==False) rows are loaded; BBQ-adapted rows are dropped at
    read time. The default language is Spanish only — English is disabled for now.

    Args:
        sesgo_dir: Root containing the `prompts/` directory of `.xlsx` files.
        categories: Subset of bias categories to load (default: all).
        languages: Language codes to load (default: Spanish only).
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
