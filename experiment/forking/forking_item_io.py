"""Serialize / reload the SELECTED forking item between the study's drivers.

The selection driver writes the chosen ambiguous SesgoPromptSample plus its pilot
outcome distribution; the capture driver reloads the SAME SesgoPromptSample (so
answer parsing uses the identical position_labels) without re-running selection.
Kept driver-local (not in src/) because it is glue between two run-by-path
scripts, but it touches no nested dict/list — it leans on BaseSchema round-trips.
"""

from __future__ import annotations

from pathlib import Path

from src.common.file_io import load_json
from src.datasets.prompt import SesgoPromptSample
from src.dynamics.forking_paths import ForkOutcomeSet
from src.dynamics.forking_paths.forking_item_selection import ItemEntropy


def selected_item_to_dict(
    sample: SesgoPromptSample, pilot: ItemEntropy, outcome_set: ForkOutcomeSet
) -> dict:
    """Flat record: the prompt sample, its pilot entropy, and the outcome set."""
    return {
        "sample": sample.to_dict(),
        "pilot_histogram": pilot.histogram,
        "pilot_entropy": pilot.entropy,
        "pilot_n_parsed": pilot.n_parsed,
        "outcome_labels": outcome_set.labels,
    }


def load_selected_sample(path: Path) -> tuple[SesgoPromptSample, ForkOutcomeSet]:
    """Reload the chosen SesgoPromptSample + outcome set from selected_item.json."""
    data = load_json(Path(path))
    sample = SesgoPromptSample.from_dict(data["sample"])
    outcome_set = ForkOutcomeSet(labels=list(data["outcome_labels"]))
    return sample, outcome_set
