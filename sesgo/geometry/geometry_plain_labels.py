"""Plain-language vocabulary for the geometry PCA figures (no pipeline jargon).

The geometry family scatters a model's internal activations (the residual
stream) in 2D and colours each point by some property of the prompt or answer.
A reviewer should read every title, axis, legend, and tick without knowing the
capture pipeline, so this module maps the raw row fields and structural-token
names to short, accessible English. It reuses the cross-study vocabulary in
``sesgo.common.plain_language_labels`` (one source of truth) and only adds the
geometry-specific pieces: human token-position names and per-axis value labels.

Key idea conveyed by every caption: points that land in SEPARATE clusters mean
the model encodes that property in its internal state; points that overlap mean
it does not (at that depth).
"""

from __future__ import annotations

from sesgo.common.plain_language_labels import (
    CATEGORY_LABEL,
    CONTEXT_LABEL,
    POLARITY_LABEL,
    ROLE_LABEL,
    scaffold_label,
)

# Structural token positions -> plain "where in the prompt we read the state".
POSITION_LABEL: dict[str, str] = {
    "im_end": "end-of-turn token",
    "newline": "newline token",
    "im_start": "start-of-turn token",
    "assistant": "assistant-role token",
    "think_open": "<think> open",
    "think_close": "</think> close",
    "answer_prefix": "just before the answer",
    "label": "the answer token",
    "mean": "averaged over the prompt",
}

# Plain headline label per colour-by axis (replaces the registry's terse names).
AXIS_TITLE: dict[str, str] = {
    "scaffold_id": "which debiasing scaffold",
    "scaffold": "debiasing scaffold vs none",
    "origin": "question source (adapted vs original)",
    "language": "question language",
    "bias_category": "bias category",
    "question_polarity": "question wording",
    "context_condition": "whether the answer is stated",
    "accuracy": "whether the model was right",
    "thinking_outcome": "did reasoning change the answer",
    "selected_role": "which group the model picked",
    "gold_role": "which group is correct",
    "readout": "how the model answered",
    "target_identity": "stereotyped group named",
    "other_identity": "other group named",
    "gold_label": "correct answer",
    "label_style": "answer-option style",
    "top_choice_prob": "confidence in its top answer",
    "top_choice_logit": "raw score of its top answer",
    "vocab_entropy": "how spread out the answer was",
    "answer_diversity": "how many answers were plausible",
    "inv_perplexity": "how predictable the top answer was",
}

# Per-axis value relabelling for the discrete legends / silhouette annotations.
_READOUT_VALUE = {"3opt": "Three-way (with 'unknown')", "2opt": "Forced two-way choice"}
_ACCURACY_VALUE = {"correct": "Correct", "incorrect": "Wrong"}
_THINKING_VALUE = {
    "unchanged": "Reasoning kept the answer",
    "changed": "Reasoning changed the answer",
    "unparsable": "No parsable reasoning",
    "None": "No reasoning step",
    "none": "No reasoning step",
}
_ORIGIN_VALUE = {"original": "Original item", "bbq_adapted": "Adapted from BBQ"}
_LANGUAGE_VALUE = {"es": "Spanish", "en": "English"}


def position_label(key: str) -> str:
    """Plain name for a structural token position; unknown keys pass through."""
    return POSITION_LABEL.get(key, key.replace("_", " "))


def axis_title(key: str, fallback: str) -> str:
    """Plain headline for a colour-by axis; falls back to the registry pretty name."""
    return AXIS_TITLE.get(key, fallback)


def axis_value_label(key: str, value: str) -> str:
    """Plain legend text for one raw value of a categorical colour-by axis.

    Delegates to the shared cross-study vocabulary where it exists (bias category,
    wording, context, role, scaffold) and keeps a small geometry-local table for
    the rest. Unknown values (e.g. free-text identities) pass through verbatim.
    """
    if value == "(other)":
        return "Other (grouped)"
    if value == "(missing)":
        return "Not recorded"
    if key == "bias_category":
        return CATEGORY_LABEL.get(value, value)
    if key == "question_polarity":
        return POLARITY_LABEL.get(value, value)
    if key == "context_condition":
        return CONTEXT_LABEL.get(value, value)
    if key in ("selected_role", "gold_role", "gold_label"):
        return ROLE_LABEL.get(value, value)
    if key in ("scaffold_id", "scaffold"):
        return _scaffold_value(value)
    if key == "readout":
        return _READOUT_VALUE.get(value, value)
    if key == "accuracy":
        return _ACCURACY_VALUE.get(value, value)
    if key == "thinking_outcome":
        return _THINKING_VALUE.get(value, value)
    if key == "origin":
        return _ORIGIN_VALUE.get(value, value)
    if key == "language":
        return _LANGUAGE_VALUE.get(value, value)
    return value


def _scaffold_value(value: str) -> str:
    """Plain scaffold legend text (the no-scaffold baseline reads as 'No scaffold')."""
    if value in ("None", "no-scaffold", "scaffold", "(baseline)"):
        mapped = {"scaffold": "With a scaffold"}.get(value)
        if mapped:
            return mapped
        return "No scaffold"
    return scaffold_label(value)
