"""Build the full risk-assessment prompt grid from MentalRiskSubjects.

The grid crosses framing x language x task-format. SCORE expands over scale
direction (which end means "at risk"); CATEGORIZE expands over label style and
option order. Each cell is rendered to a clean, model-agnostic prompt string and
captured in a RiskPromptSample carrying the metadata needed to parse the reply.
"""

from __future__ import annotations

from itertools import product

from src.datasets.mental_risk import Disorder, MentalRiskSubject
from .risk_framing import RiskFraming, get_framing
from .risk_instruction import render_instruction
from .risk_language import RiskLanguage
from .risk_prompt_config import RiskPromptConfig
from .risk_prompt_dataset import RiskPromptDataset
from .risk_prompt_sample import RiskPromptSample
from .risk_task_type import RiskTaskType

# Transcript markers keep the subject text clearly delimited from instructions.
_OPEN, _CLOSE = "<transcript>", "</transcript>"

# Score direction phrasing per language; describes which end means "at risk".
_SCALE_HINT = {
    (RiskLanguage.EN.value, True): "where 1 means at risk and 0 means not at risk",
    (RiskLanguage.EN.value, False): "where 0 means at risk and 1 means not at risk",
    (RiskLanguage.ES.value, True): "donde 1 significa en riesgo y 0 significa sin riesgo",
    (RiskLanguage.ES.value, False): "donde 0 significa en riesgo y 1 significa sin riesgo",
}
# Localized "answer with a single number" cue for SCORE prompts.
_NUMBER_CUE = {
    RiskLanguage.EN.value: "Answer with a single number between 0 and 1",
    RiskLanguage.ES.value: "Responde con un único número entre 0 y 1",
}


def _transcript_block(transcript: str) -> str:
    """The subject transcript fenced between markers."""
    return f"{_OPEN}\n{transcript}\n{_CLOSE}"


def _disorder_label(disorder: Disorder, lang: str) -> str:
    """Disorder phrasing fragment in the prompt language."""
    return disorder.label_es if lang == RiskLanguage.ES.value else disorder.label


class RiskPromptGenerator:
    """Expands MentalRiskSubjects into the configured risk-prompt grid."""

    def __init__(self, config: RiskPromptConfig):
        self.config = config

    def generate(self, subjects: list[MentalRiskSubject]) -> RiskPromptDataset:
        """Render every selected grid cell for every subject."""
        samples: list[RiskPromptSample] = []
        idx = 0
        for subject in subjects:
            for framing_key, lang in product(self.config.framings, self.config.languages):
                framing = get_framing(framing_key)
                for sample in self._cells_for(subject, framing, lang, idx):
                    samples.append(sample)
                    idx += 1
        dataset = RiskPromptDataset(dataset_id="", config=self.config, samples=samples)
        dataset.dataset_id = dataset.get_id()
        return dataset

    def _cells_for(
        self, subject: MentalRiskSubject, framing: RiskFraming, lang: str, start: int
    ):
        """Yield every task/format cell for one (subject, framing, language)."""
        idx = start
        block = _transcript_block(subject.transcript)
        disorder = _disorder_label(subject.disorder, lang)
        question = framing.question(lang, disorder)
        for task_type in self.config.task_types:
            if task_type is RiskTaskType.SCORE:
                builders = self._score_cells(framing, lang, block, question)
            else:
                builders = self._categorize_cells(framing, lang, block, question)
            for build in builders:
                yield build(subject, idx)
                idx += 1

    def _score_cells(self, framing, lang, block, question):
        """One builder per scale direction (scale_high / scale_low)."""
        cells = []
        for scale_high in self.config.scale_highs:
            hint = _SCALE_HINT[(lang, scale_high)]
            cue = _NUMBER_CUE[lang]
            text = (
                f"{render_instruction(lang, RiskTaskType.SCORE)}\n\n{block}\n\n"
                f"{question} {cue}, {hint}."
            )

            def build(subject, idx, _text=text, _sh=scale_high, _f=framing, _l=lang):
                return RiskPromptSample(
                    sample_idx=idx,
                    text=_text,
                    subject_id=subject.subject_id,
                    disorder=subject.disorder.value,
                    gold_risk=subject.risk,
                    framing=_f.key,
                    task_type=RiskTaskType.SCORE,
                    language=_l,
                    scale_high=_sh,
                )

            cells.append(build)
        return cells

    def _categorize_cells(self, framing, lang, block, question):
        """One builder per (label style x order flip)."""
        cells = []
        pos = framing.positive_label(lang)
        neg = framing.negative_label(lang)
        prefix = self.config.choice_prefix(lang)
        instruction = render_instruction(lang, RiskTaskType.CATEGORIZE)
        for labels, flip in product(self.config.label_styles, self.config.order_flips):
            # flip decides whether the negative phrase is listed first; positive_idx
            # then records which label index points at the at-risk answer.
            phrase_a, phrase_b = (neg, pos) if flip else (pos, neg)
            positive_idx = 1 if flip else 0
            text = (
                f"{instruction}\n\n{block}\n\n{question}\n"
                f"{labels[0]} {phrase_a}\n{labels[1]} {phrase_b}\n{prefix}"
            )

            def build(subject, idx, _t=text, _lb=labels, _pi=positive_idx,
                      _fl=flip, _f=framing, _l=lang, _p=prefix):
                return RiskPromptSample(
                    sample_idx=idx,
                    text=_t,
                    subject_id=subject.subject_id,
                    disorder=subject.disorder.value,
                    gold_risk=subject.risk,
                    framing=_f.key,
                    task_type=RiskTaskType.CATEGORIZE,
                    language=_l,
                    labels=_lb,
                    positive_idx=_pi,
                    choice_prefix=_p,
                    label_flipped=_fl,
                )

            cells.append(build)
        return cells
