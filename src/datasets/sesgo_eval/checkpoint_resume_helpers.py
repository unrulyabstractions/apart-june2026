"""Identity keying shared by the SESGO checkpoint/resume paths.

A resumed run must skip prompts whose result is already in the checkpoint. Both
prompt samples (SesgoPromptSample) and persisted results (SesgoSample /
GeometrySample) carry the same provenance axes, so one keying function serves
every collect driver. ``sample_idx`` is the natural primary key (unique per
rendered prompt); when it is absent/degenerate we fall back to the tuple of
(question_id, scaffold_id, label_style, context_condition), which uniquely
identifies a rendering within a study grid.
"""

from __future__ import annotations

from typing import Any

# A sample identity is either its int sample_idx or the 4-axis fallback tuple.
SampleIdentity = int | tuple[str, str | None, str, str]


def sample_identity(obj: Any) -> SampleIdentity:
    """Stable identity for a prompt sample or a collected result.

    Prefers ``sample_idx`` (the per-rendering primary key). Falls back to the
    (question_id, scaffold_id, label_style, context_condition) tuple when the
    index is missing or the sentinel -1, so two objects describing the SAME
    rendering always key equal regardless of which carries an index.
    """
    idx = getattr(obj, "sample_idx", None)
    if isinstance(idx, int) and idx >= 0:
        return idx
    return (
        getattr(obj, "question_id", ""),
        getattr(obj, "scaffold_id", None),
        getattr(obj, "label_style", ""),
        getattr(obj, "context_condition", ""),
    )


def completed_identities(results: list[Any]) -> set[SampleIdentity]:
    """Set of identities already present in a loaded checkpoint's results."""
    return {sample_identity(r) for r in results}
