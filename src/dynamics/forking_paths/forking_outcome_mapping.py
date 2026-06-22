"""Map a rollout's text to its categorical OUTCOME label (the R function).

Forking-paths defines R(rollout) as a one-hot over the answer set. For SESGO we
reuse the EXACT thinking-level answer parser (``parse_chosen_label``) the other
studies use, then name the parsed role; unparseable draws fall into the catch-all
UNPARSEABLE bucket. Keeping this in one tiny module means the outcome function is
swappable (e.g. an LLM answer-extractor) without touching the histogram math.
"""

from __future__ import annotations

from src.datasets.prompt import SesgoPromptSample
from src.datasets.sesgo_eval import parse_chosen_label

from .forking_outcome_set import UNPARSEABLE_OUTCOME


def rollout_to_outcome_label(rollout_text: str, sample: SesgoPromptSample) -> str:
    """Outcome label for one rollout: the parsed SESGO role, else UNPARSEABLE.

    ``rollout_text`` is the FULL continuation text (the answer lives after
    ``</think>``; ``parse_chosen_label`` strips the reasoning block itself).
    """
    label = parse_chosen_label(rollout_text, sample)
    return label.value if label is not None else UNPARSEABLE_OUTCOME
