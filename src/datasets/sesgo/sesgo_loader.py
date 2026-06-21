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


# Provenance axis codes accepted by `origins`. "original" keeps bbq==False rows,
# "bbq-adapted" keeps bbq==True rows; both kept => the full provenance grid.
_ORIGIN_BBQ: dict[str, bool] = {"original": False, "bbq-adapted": True}


def _origin_mask(df: pd.DataFrame, origins: tuple[str, ...]) -> pd.Series:
    """Boolean row mask keeping only the requested provenance(s)."""
    keep_bbq = {_ORIGIN_BBQ[o] for o in origins}
    return df["bbq"].astype(bool).isin(keep_bbq)


def _load_file(
    path: Path,
    category: SesgoCategory,
    language: str,
    limit: int | None,
    origins: tuple[str, ...],
) -> list[SesgoItem]:
    """Read one prompt file, keep the requested provenance(s), cap at `limit`.

    Both context conditions (ambig + disambig) are always retained. `origins`
    selects the provenance axis: the default ("original",) keeps only bbq==False
    rows (Spanish-original SESGO); adding "bbq-adapted" opts the BBQ-adapted rows
    back in for the full-data grid.
    """
    df = pd.read_excel(path)
    kept = df[_origin_mask(df, origins)]
    if limit is not None:
        kept = kept.head(limit)
    return [_row_to_item(row, category, language) for _, row in kept.iterrows()]


def load_items(
    sesgo_dir: Path | str = "datasets/SESGO",
    categories: list[SesgoCategory] | None = None,
    languages: tuple[str, ...] = ("es",),
    limit: int | None = None,
    origins: tuple[str, ...] = ("original",),
) -> list[SesgoItem]:
    """Load SESGO items (both context conditions) as typed SesgoItem objects.

    By DEFAULT only ORIGINAL (bbq==False) Spanish rows are loaded — the es-original
    grid the running studies depend on. `languages` and `origins` are opt-in axes:
    pass `languages=("es", "en")` and/or `origins=("original", "bbq-adapted")` to
    widen to the full grid without changing the default behavior.

    Args:
        sesgo_dir: Root containing the `prompts/` directory of `.xlsx` files.
        categories: Subset of bias categories to load (default: all).
        languages: Language codes to load (default: Spanish only).
        limit: Cap on items per (category, language) — useful for smoke tests.
        origins: Provenance axis to keep (default: original only); valid codes are
            "original" and "bbq-adapted".

    Returns:
        Flat list of SesgoItem across the requested categories/languages/origins.
    """
    sesgo_dir = Path(sesgo_dir)
    categories = categories or list(SesgoCategory)
    unknown = [o for o in origins if o not in _ORIGIN_BBQ]
    if unknown:
        raise ValueError(f"Unknown origin(s) {unknown}; choose from {sorted(_ORIGIN_BBQ)}")
    items: list[SesgoItem] = []
    for category in categories:
        for language in languages:
            path = _find_prompt_file(sesgo_dir, category, language)
            if path is None:
                log(f"[sesgo] skipping missing prompt file: {category.value}/{language}")
                continue
            items.extend(_load_file(path, category, language, limit, origins))
    return items
