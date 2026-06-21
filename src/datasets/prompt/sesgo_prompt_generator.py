"""Build the full SESGO ambiguous-bias prompt grid from SesgoItems.

The grid crosses, per item: scaffold-condition (no-scaffold plus each supplied
scaffold) x role->position permutation x label style. Permuting which role
(target/other/unknown) sits at which displayed position across all 6 orderings
is what defeats position bias — the model cannot win by always picking slot 1.
Each cell renders to a clean, model-agnostic prompt string captured in a
SesgoPromptSample carrying the metadata needed to decode the reply.
"""

from __future__ import annotations

from itertools import permutations, product

from src.datasets.sesgo import SesgoItem, SesgoLabel
from .sesgo_prompt_config import SesgoPromptConfig
from .sesgo_prompt_dataset import SesgoPromptDataset
from .sesgo_prompt_localization import sesgo_choice_prefix, sesgo_markers
from .sesgo_prompt_sample import SesgoPromptSample
from .sesgo_scaffold import Scaffold

# All three roles in their canonical (corpus) order; permutations are taken over
# this tuple so the identity permutation reproduces the corpus ordering.
_CANONICAL_ROLES = (SesgoLabel.OTHER, SesgoLabel.TARGET, SesgoLabel.UNKNOWN)


def _render(scaffold: Scaffold | None, item: SesgoItem, markers, roles, prefix) -> str:
    """Render one prompt: optional scaffold, then context/question/N options.

    Section markers are localized to the item's language so a Spanish item reads
    as one coherent Spanish prompt; the option text itself is already authored
    in that language. ``roles``/``markers`` may hold 2 (forced-choice) or 3
    options — the option block is built over however many positions are passed.
    """
    preamble = f"{scaffold.text(item.language)}\n\n" if scaffold else ""
    ctx_marker, q_marker, opt_marker = sesgo_markers(item.language)
    texts = _role_texts(item)
    options = "\n".join(f"{markers[i]} {texts[roles[i]]}" for i in range(len(roles)))
    return (
        f"{preamble}{ctx_marker}\n{item.context}\n"
        f"{q_marker}\n{item.question}\n"
        f"{opt_marker}\n{options}\n"
        f"{prefix}"
    )


# The 2-option forced choice drops UNKNOWN, presenting only the two groups in the
# corpus' canonical order (OTHER, then TARGET).
_TWO_OPTION_ROLES = (SesgoLabel.OTHER, SesgoLabel.TARGET)


def _role_texts(item: SesgoItem) -> dict[SesgoLabel, str]:
    """Map each role to its authored option text for this (language-fixed) item."""
    return {label: text for label, text in item.options_in_canonical_order}


class SesgoPromptDatasetGenerator:
    """Expands SesgoItems into the configured ambiguous-bias prompt grid."""

    def __init__(self, config: SesgoPromptConfig):
        self.config = config

    def generate(
        self, items: list[SesgoItem], scaffolds: list[Scaffold] | None = None
    ) -> SesgoPromptDataset:
        """Render every selected grid cell for every item."""
        scaffolds = scaffolds or []
        conditions = self._scaffold_conditions(scaffolds)
        samples: list[SesgoPromptSample] = []
        idx = 0
        for item in items:
            for scaffold, roles, markers in self._cells(conditions):
                samples.append(self._build(item, scaffold, roles, markers, idx))
                idx += 1
        dataset = SesgoPromptDataset(
            dataset_id="",
            config=self.config,
            scaffold_ids=[s.scaffold_id for s in scaffolds],
            samples=samples,
        )
        dataset.dataset_id = dataset.get_id()
        return dataset

    def _scaffold_conditions(self, scaffolds: list[Scaffold]) -> list[Scaffold | None]:
        """The scaffold axis: optional no-scaffold baseline, then each scaffold."""
        baseline: list[Scaffold | None] = [None] if self.config.include_no_scaffold else []
        return baseline + list(scaffolds)

    def _cells(self, conditions: list[Scaffold | None]):
        """Yield (scaffold, role-order, markers) for every configured cell."""
        # all_permutations crosses all 6 role orderings; otherwise keep only the
        # canonical ordering so position is held fixed.
        orderings = (
            list(permutations(_CANONICAL_ROLES)) if self.config.all_permutations
            else [_CANONICAL_ROLES]
        )
        yield from product(conditions, orderings, self.config.label_styles)

    def _build(self, item, scaffold, roles, markers, idx) -> SesgoPromptSample:
        """Render one cell and capture its role-decoding metadata.

        Renders BOTH forms from the same cell: the 3-option prompt (the permuted
        roles/markers) and a 2-option forced-choice prompt (target+other only, no
        UNKNOWN) using the cell's first two markers in canonical group order.
        """
        # Default to the language-derived cue so es prompts teacher-force
        # "Respuesta: "; an explicit config override wins if one is set.
        prefix = self.config.choice_prefix or sesgo_choice_prefix(item.language)
        text = _render(scaffold, item, markers, roles, prefix)
        markers_2opt = (markers[0], markers[1])
        text_2opt = _render(scaffold, item, markers_2opt, _TWO_OPTION_ROLES, prefix)
        return SesgoPromptSample(
            sample_idx=idx,
            question_id=item.question_id,
            bias_category=item.category.value,
            question_polarity=item.polarity,
            context_condition=item.context_condition,
            language=item.language,
            scaffold_id=scaffold.scaffold_id if scaffold else None,
            label_style="".join(markers),
            text=text,
            option_labels=tuple(markers),
            position_labels=tuple(roles),
            choice_prefix=prefix,
            text_2opt=text_2opt,
            option_labels_2opt=markers_2opt,
            position_labels_2opt=_TWO_OPTION_ROLES,
            gold_label=item.gold_label,
            bbq=item.bbq,
            # answer_info convention: ans1=TARGET group, ans0=OTHER group.
            target_identity=item.target_text,
            other_identity=item.other_text,
        )
