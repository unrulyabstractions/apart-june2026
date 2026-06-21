"""One prompt's multi-level SESGO readout, with provenance to interpret it.

Each sample carries every color-by axis as a flat field (so analyses can slice
by question, scaffold, polarity, category, language, label style, context
condition without re-joining) plus the model readouts: the 3-option non-thinking
distribution (``non_thinking``, teacher-forced positions remapped to roles), the
2-option forced-choice readout (``non_thinking_2opt``, no UNKNOWN), and a thinking
distribution (counted over parsed free-form draws). Gold depends on the context
condition (ambiguous -> UNKNOWN; disambiguated -> the ground-truth role), and
correctness is computed against that gold. Raw thinking completions are heavy and
not part of identity, so the `_` prefix drops them from the id hash and to_dict.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common import BaseSchema
from src.datasets.sesgo import SesgoLabel
from .sesgo_correctness import is_correct, two_option_correct
from .sesgo_non_thinking import SesgoNonThinking
from .sesgo_thinking import SesgoThinking
from .sesgo_two_option import SesgoTwoOption


@dataclass
class SesgoSample(BaseSchema):
    """SESGO judgement(s) for a single rendered prompt."""

    sample_idx: int
    question_id: str
    scaffold_id: str | None
    question_polarity: str
    bias_category: str
    context_condition: str  # "ambig" or "disambig"
    language: str
    label_style: str
    gold_label: SesgoLabel
    prompt_text: str
    # Provenance / social-group axes threaded from the prompt sample so analyses
    # can slice by origin (original vs BBQ-adapted) and the literal group strings.
    bbq: bool = False
    target_identity: str = ""
    other_identity: str = ""
    non_thinking: SesgoNonThinking | None = None
    non_thinking_2opt: SesgoTwoOption | None = None
    thinking: SesgoThinking | None = None
    # Heavy/private: raw generations are excluded from identity and to_dict.
    _thinking_completions: list[str] | None = None

    @property
    def predicted_non_thinking(self) -> SesgoLabel | None:
        """Argmax role from the 3-option teacher-forced readout, if present."""
        return self.non_thinking.predicted if self.non_thinking else None

    @property
    def predicted_thinking(self) -> SesgoLabel | None:
        """Argmax role over the parsed draws; None when no draw parsed (n == 0)."""
        if self.thinking is None or self.thinking.sample_size == 0:
            return None
        return self.thinking.predicted

    @property
    def picked_2opt(self) -> SesgoLabel | None:
        """Group the 2-option forced choice picked, if that readout ran."""
        return self.non_thinking_2opt.picked if self.non_thinking_2opt else None

    @property
    def correct_non_thinking(self) -> bool:
        """True iff the 3-option prediction matches the per-condition gold role."""
        return is_correct(self.predicted_non_thinking, self.gold_label)

    @property
    def correct_thinking(self) -> bool:
        """True iff the thinking prediction matches the per-condition gold role."""
        return is_correct(self.predicted_thinking, self.gold_label)

    @property
    def correct_2opt(self) -> bool | None:
        """Forced-choice correctness; None for ambiguous items (no UNKNOWN)."""
        return two_option_correct(
            self.picked_2opt, self.gold_label, self.context_condition
        )
