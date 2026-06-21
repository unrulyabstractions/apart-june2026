"""One prompt's SESGO readout PLUS the residual-stream snapshots taken on it.

This is the geometry-half analogue of SesgoSample: it carries the same flat
color-by axes and the same two model readouts (non_thinking / thinking), and
adds ``activations`` — the list of structural residual snapshots captured while
teacher-forcing the answer. The tensors themselves live on disk (see
GeometryActivation.path); this schema only references them, so it stays light
and serializes through BaseSchema like every other sample type.

On top of the categorical axes it also carries the DERIVED colour-by signals the
geometry viz scatters on: the continuous answer-distribution scalars (top-choice
prob / logit, answer-distribution entropy / diversity, inverse perplexity — see
GeometryAnswerSignals) plus flat categorical flags (which role the model picked,
the gold role, the readout used, and whether a scaffold was present). These are
filled once at collection time so the viz never re-derives them from the nested
readouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common import BaseSchema
from src.datasets.sesgo import SesgoLabel
from .geometry_activation import GeometryActivation
from .geometry_answer_signals import GeometryAnswerSignals
from .sesgo_correctness import is_correct, two_option_correct
from .sesgo_greedy_thinking import SesgoGreedyThinking
from .sesgo_non_thinking import SesgoNonThinking
from .sesgo_thinking import SesgoThinking
from .sesgo_two_option import SesgoTwoOption

# The readout whose answer distribution the continuous signals are derived from.
_THREE_OPTION = "3opt"
_NEGATIVE_POLARITY = "neg"  # SESGO marks negative-framed questions with this polarity


@dataclass
class GeometrySample(BaseSchema):
    """A SESGO judgement plus its captured residual geometry for one prompt."""

    sample_idx: int
    question_id: str
    scaffold_id: str | None
    bias_category: str
    question_polarity: str
    context_condition: str  # "ambig" or "disambig"
    language: str
    gold_label: SesgoLabel
    prompt_text: str
    # Every per-sample colour-by axis the geometry viz funnels the representation
    # through: label_style + provenance (bbq) + the literal social-group strings.
    label_style: str = ""
    bbq: bool = False
    target_identity: str = ""
    other_identity: str = ""
    non_thinking: SesgoNonThinking | None = None
    non_thinking_2opt: SesgoTwoOption | None = None
    greedy_thinking: SesgoGreedyThinking | None = None
    thinking: SesgoThinking | None = None
    activations: list[GeometryActivation] = field(default_factory=list)
    # Derived flat colour-by signals, filled at collection time (collect_geometry_
    # samples._make_sample). The continuous answer-distribution scalars live on
    # ``answer_signals``; the categorical flags are flattened straight here so a
    # frontend can scatter / colour by any of them with no nested access.
    answer_signals: GeometryAnswerSignals = field(default_factory=GeometryAnswerSignals)
    readout: str = _THREE_OPTION  # "3opt" (teacher-forced) | "2opt" (forced choice)
    has_scaffold: bool = False  # whether a debiasing scaffold preceded this prompt

    @property
    def is_negative_polarity(self) -> bool:
        """True iff the question is negative-framed (a colour-by binary)."""
        return self.question_polarity == _NEGATIVE_POLARITY

    @property
    def gold_role(self) -> str:
        """The per-condition gold role as a flat string (target/other/unknown)."""
        return getattr(self.gold_label, "value", str(self.gold_label))

    @property
    def selected_role(self) -> str:
        """Role the model picked, by the active readout (argmax); "(none)" if absent.

        The 2-option forced choice surfaces ``picked`` (no UNKNOWN); the 3-option
        teacher-forced readout surfaces its argmax ``predicted``. Mirrors the
        ``readout`` flag so the selected-role axis is consistent with it.
        """
        role = self.picked_2opt if self.readout != _THREE_OPTION else self.predicted_non_thinking
        return getattr(role, "value", "(none)") if role is not None else "(none)"

    @property
    def is_correct(self) -> bool:
        """Whether the active readout's prediction matches the per-condition gold."""
        if self.readout != _THREE_OPTION:
            return bool(self.correct_2opt)
        return self.correct_non_thinking

    @property
    def predicted_non_thinking(self) -> SesgoLabel | None:
        """Argmax role from the 3-option teacher-forced readout, if present."""
        return self.non_thinking.predicted if self.non_thinking else None

    @property
    def predicted_thinking(self) -> SesgoLabel | None:
        """Argmax role over parsed draws; None when no draw parsed (n == 0)."""
        if self.thinking is None or self.thinking.sample_size == 0:
            return None
        return self.thinking.predicted

    @property
    def predicted_greedy_thinking(self) -> SesgoLabel | None:
        """Role the single greedy reasoning decode committed to, if that ran."""
        return self.greedy_thinking.predicted if self.greedy_thinking else None

    @property
    def picked_2opt(self) -> SesgoLabel | None:
        """Group the 2-option forced choice picked, if that readout ran."""
        return self.non_thinking_2opt.picked if self.non_thinking_2opt else None

    @property
    def correct_non_thinking(self) -> bool:
        """True iff the 3-option prediction matches the per-condition gold."""
        return is_correct(self.predicted_non_thinking, self.gold_label)

    @property
    def correct_thinking(self) -> bool:
        """True iff the thinking prediction matches the per-condition gold role."""
        return is_correct(self.predicted_thinking, self.gold_label)

    @property
    def correct_greedy_thinking(self) -> bool:
        """True iff the greedy-thinking prediction matches the per-condition gold."""
        return is_correct(self.predicted_greedy_thinking, self.gold_label)

    @property
    def correct_2opt(self) -> bool | None:
        """Forced-choice correctness; None for ambiguous items (no UNKNOWN)."""
        return two_option_correct(
            self.picked_2opt, self.gold_label, self.context_condition
        )
