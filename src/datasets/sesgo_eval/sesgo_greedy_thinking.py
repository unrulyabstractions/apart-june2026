"""The greedy-thinking SESGO readout: one deterministic decode WITH reasoning.

Distinct from the other three readouts: unlike the greedy NON-thinking decode
(which prefills the skip-thinking block so the model answers without reasoning),
this lets the model reason greedily — a single temperature-0 generation with NO
skip-thinking prefix — then parses the role it commits to AFTER ``</think>``. It
is the answer the model settles on when it reasons but cannot vary its draws,
sitting between the teacher-forced non-thinking choose3 and the sampled thinking
distribution.

Kept a clean BaseSchema: the parsed role + the (short) decoded answer text.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common import BaseSchema
from src.datasets.sesgo import SesgoLabel


@dataclass
class SesgoGreedyThinking(BaseSchema):
    """One greedy reasoning decode, parsed to the role the model committed to.

    ``label`` is ``None`` when the draw could not be parsed (e.g. the model ran
    out of budget mid-``<think>`` and emitted no answer), matching how the
    thinking level drops unparseable draws.
    """

    label: SesgoLabel | None = None  # role the greedy reasoning decode chose
    text: str = ""  # the decoded answer (post-</think>, short)

    @property
    def predicted(self) -> SesgoLabel | None:
        """Role the greedy reasoning decode committed to, or None if unparseable."""
        return self.label
