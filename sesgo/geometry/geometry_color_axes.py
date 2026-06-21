"""The single registry of geometry colour-by axes (categorical + continuous).

ONE source of truth shared by analyze_geometry.py (which emits per-axis
silhouette separation) and visualize_geometry_samples.py (which scatters the PCA
cloud coloured by each axis). Adding a new colour-by axis means adding ONE
``ColorAxis`` row here — both the analysis separation block and the viz panel
grid pick it up automatically (DRY).

Each axis is either CATEGORICAL (discrete legend; silhouette / between-within
separability is meaningful) or CONTINUOUS (sequential colormap + colorbar; the
scalar is scattered directly, no silhouette). The ``key`` is the flat field name
on every projected-sample row (see analyze_geometry._sample_row), so the viz and
analysis read the value with the same key.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common import BaseSchema


@dataclass
class ColorAxis(BaseSchema):
    """One colour-by axis: its flat row key, label, and kind (cat/continuous)."""

    key: str  # flat field name on a projected-sample row
    pretty: str  # human-readable axis label for titles / legends
    continuous: bool = False  # True -> sequential colormap; False -> discrete legend


# Every colour-by axis the geometry PCA can be coloured by, in display order.
# CATEGORICAL axes get a discrete legend (high-cardinality identities fold to
# top-K + "(other)") AND a silhouette separability score; CONTINUOUS answer-
# distribution scalars get a sequential colormap with a colorbar (no silhouette).
COLOR_AXES: tuple[ColorAxis, ...] = (
    # ── Categorical (discrete legend + silhouette separability) ──────────────
    # ``pretty`` strings are plain-language headlines a non-expert reads at a
    # glance (see sesgo.geometry.geometry_plain_labels for the value relabelling).
    ColorAxis("scaffold_id", "which debiasing scaffold"),
    ColorAxis("scaffold", "debiasing scaffold vs none"),
    ColorAxis("origin", "question source (adapted vs original)"),
    ColorAxis("language", "question language"),
    ColorAxis("bias_category", "bias category"),
    ColorAxis("question_polarity", "question wording"),
    ColorAxis("context_condition", "whether the answer is stated"),
    ColorAxis("accuracy", "whether the model was right"),
    ColorAxis("thinking_outcome", "did reasoning change the answer"),
    ColorAxis("selected_role", "which group the model picked"),
    ColorAxis("gold_role", "which group is correct"),
    ColorAxis("readout", "how the model answered"),
    ColorAxis("target_identity", "stereotyped group named"),
    ColorAxis("other_identity", "other group named"),
    ColorAxis("gold_label", "correct answer"),
    ColorAxis("label_style", "answer-option style"),
    # ── Continuous answer-distribution signals (colormap + colorbar) ─────────
    ColorAxis("top_choice_prob", "confidence in its top answer", continuous=True),
    ColorAxis("top_choice_logit", "raw score of its top answer", continuous=True),
    ColorAxis("vocab_entropy", "how spread out the answer was", continuous=True),
    ColorAxis("answer_diversity", "how many answers were plausible", continuous=True),
    ColorAxis("inv_perplexity", "how predictable the top answer was", continuous=True),
)

# The scaffold axis carries None as the no-scaffold baseline; handled specially.
SCAFFOLD_AXIS_KEY = "scaffold_id"

# Convenience views (do NOT re-list — derive from the single registry above).
CATEGORICAL_AXES: tuple[ColorAxis, ...] = tuple(a for a in COLOR_AXES if not a.continuous)
CONTINUOUS_AXES: tuple[ColorAxis, ...] = tuple(a for a in COLOR_AXES if a.continuous)
# Categorical axes (minus scaffold_id, reported separately) we score separation for.
SEPARATION_AXES: tuple[ColorAxis, ...] = tuple(
    a for a in CATEGORICAL_AXES if a.key != SCAFFOLD_AXIS_KEY
)
# The key axes whose depth-of-separation we sweep across layers in the viz.
KEY_SWEEP_AXIS_KEYS: tuple[str, ...] = (
    "accuracy",
    "context_condition",
    "selected_role",
    SCAFFOLD_AXIS_KEY,
)
