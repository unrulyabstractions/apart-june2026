"""Tests for the MentalRiskES loader against a synthetic fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.datasets.mental_risk import (
    Disorder,
    MentalRiskMessage,
    MentalRiskSubject,
    collapse_risk,
    load_subjects,
)

FIXTURE = Path(__file__).parents[2] / "fixtures" / "mental_risk" / "extracted"

# Expected collapsed risk per subject id: rbs when present, bs fallback when rbs
# is absent (subject10), and None when the subject has no gold row (subject21).
EXPECTED_RISK = {
    "subject2": 0.7,
    "subject10": 1.0,
    "subject5": 0.9,
    "subject12": 0.1,
    "subject3": 0.6,
    "subject7": 0.2,
    "subject21": None,
}


@pytest.fixture
def subjects() -> list[MentalRiskSubject]:
    return load_subjects(FIXTURE)


def test_loads_all_subjects(subjects):
    assert len(subjects) == 7
    assert {s.subject_id for s in subjects} == set(EXPECTED_RISK)


def test_disorders_present(subjects):
    by_disorder = {s.disorder for s in subjects}
    assert by_disorder == {Disorder.ANXIETY, Disorder.DEPRESSION, Disorder.EATING_DISORDER}


def test_transcript_orders_messages():
    subject = next(s for s in load_subjects(FIXTURE) if s.subject_id == "subject2")
    assert subject.n_messages == 3
    # Joined in id_message order, regardless of file order.
    assert subject.transcript == (
        "Hola, hoy me siento un poco inquieto.\n"
        "No logro relajarme antes de dormir.\n"
        "Gracias por leerme, me ayuda a desahogarme."
    )


def test_risk_collapse_matches_expected(subjects):
    by_id = {s.subject_id: s for s in subjects}
    for subject_id, expected in EXPECTED_RISK.items():
        assert by_id[subject_id].risk == expected


def test_collapse_risk_unit():
    assert collapse_risk({"rbs": 0.7, "bs": 1.0}) == 0.7  # rbs preferred
    assert collapse_risk({"bs": 1.0}) == 1.0  # bs fallback
    assert collapse_risk({"bc": 1.0}) is None  # neither present
    assert collapse_risk({"rbs": 1.5}) == 1.0  # clamped to [0, 1]


def test_subject_round_trips(subjects):
    subject = subjects[0]
    restored = MentalRiskSubject.from_dict(subject.to_dict())
    assert restored == subject
    assert isinstance(restored.disorder, Disorder)
    assert all(isinstance(m, MentalRiskMessage) for m in restored.messages)
