"""Per-position RAW rollout dump records (the auditable companion to {O_t}).

The captured ``ForkingTrajectory`` keeps only each rollout's PARSED outcome label
(``rollout_labels``), discarding the generated text. For debugging "why is this
draw unparseable?" we also dump, per base-path position, EVERY sampled alternate's
raw continuation TEXT alongside its parsed label + token info. One ``Forking
PositionDump`` is written per position (``pos_000.json`` ...) so a reader can map
any rollout back to its (position, alternate, sample) and re-inspect the text.

Every field is a flat scalar / list of flat records, so the dump roundtrips
cleanly via BaseSchema without nested dict/list arguments.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common.base_schema import BaseSchema


@dataclass
class RolloutDumpEntry(BaseSchema):
    """ONE sampled continuation after forcing alternate token w at a position.

    ``sample_index`` is this rollout's index within the alternate's S draws;
    ``raw_text`` is the full generated continuation (answer after ``</think>``);
    ``outcome_label`` is the parsed role (target/other/unknown) or the catch-all
    ``unparseable`` bucket. The token fields tie the entry to its forking-trajectory
    alternate so a reader can line dump entries up with ``AltTokenRollouts``.
    """

    alt_index: int  # index of the alternate token within the position's alternates
    token_id: int
    token_text: str
    token_prob: float
    is_base_token: bool  # True for the greedy base-path token w*
    sample_index: int  # which of the alternate's S continuations this is
    outcome_label: str  # parsed role, else "unparseable"
    raw_text: str  # the FULL generated continuation text


@dataclass
class ForkingPositionDump(BaseSchema):
    """All raw rollouts behind ONE base-path token position's O_t histogram.

    ``position`` indexes the ``ForkingTrajectory.positions`` list it mirrors, so a
    reader maps every ``entries`` row back to the trajectory position. ``entries``
    is the flat list of per-(alternate, sample) raw continuations.
    """

    position: int
    base_token_id: int
    base_token_text: str
    entries: list[RolloutDumpEntry] = field(default_factory=list)
