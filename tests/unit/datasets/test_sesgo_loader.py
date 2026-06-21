"""Tests for the SESGO loader against a synthetic xlsx fixture.

The fixture contains invented prompts (no real SESGO content): an ambiguous
neg/nonneg pair, a second ambiguous item, and one disambig row that must be
dropped. Its filename is mixed-case to also exercise the case-insensitive glob.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.datasets.sesgo import (
    SesgoCategory,
    SesgoItem,
    SesgoLabel,
    load_items,
)

FIXTURE = Path(__file__).parents[2] / "fixtures" / "sesgo"


@pytest.fixture
def items() -> list[SesgoItem]:
    # Only Spanish is requested implicitly via the default; restrict to the one
    # English fixture file we ship so the test is deterministic.
    return load_items(FIXTURE, categories=[SesgoCategory.RACISM], languages=("en",))


def test_loads_only_ambiguous_items(items):
    # 3 ambiguous rows survive; the single disambig row is filtered out.
    assert len(items) == 3
    assert all(i.category is SesgoCategory.RACISM for i in items)
    assert all(i.language == "en" for i in items)


def test_answer_info_unpacked_into_roles(items):
    pair = next(i for i in items if i.context == "Two coworkers shared an office.")
    assert pair.other_text == "The blue person"  # ans0 = OTHER
    assert pair.target_text == "The red person"  # ans1 = TARGET
    assert pair.unknown_text == "Not enough information"  # ans2 = UNKNOWN


def test_gold_is_always_unknown_for_ambiguous(items):
    assert all(i.gold_label is SesgoLabel.UNKNOWN for i in items)


def test_options_in_canonical_order(items):
    item = items[0]
    labels = [label for label, _ in item.options_in_canonical_order]
    assert labels == [SesgoLabel.OTHER, SesgoLabel.TARGET, SesgoLabel.UNKNOWN]
    texts = [text for _, text in item.options_in_canonical_order]
    assert texts == [item.other_text, item.target_text, item.unknown_text]


def test_polarity_pair_shares_question_id(items):
    # The neg/nonneg rows with the same context must collapse to one id.
    office = [i for i in items if i.context == "Two coworkers shared an office."]
    assert {i.polarity for i in office} == {"neg", "nonneg"}
    assert len({i.question_id for i in office}) == 1


def test_limit_caps_per_file():
    capped = load_items(
        FIXTURE, categories=[SesgoCategory.RACISM], languages=("en",), limit=1
    )
    assert len(capped) == 1


def test_item_round_trips(items):
    item = items[0]
    restored = SesgoItem.from_dict(item.to_dict())
    assert restored == item
    assert isinstance(restored.category, SesgoCategory)
    assert isinstance(restored.gold_label, SesgoLabel)
