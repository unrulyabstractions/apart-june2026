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
    ColorAxis("scaffold_id", "scaffold"),
    ColorAxis("scaffold", "scaffold vs no-scaffold"),
    ColorAxis("origin", "origin (BBQ-adapted vs original)"),
    ColorAxis("language", "language"),
    ColorAxis("bias_category", "bias category"),
    ColorAxis("question_polarity", "question polarity (negative-framed vs not)"),
    ColorAxis("context_condition", "context condition (ambig vs disambig)"),
    ColorAxis("accuracy", "accuracy (correct vs incorrect)"),
    ColorAxis("thinking_outcome", "thinking changed the answer (unchanged/changed/unparsable)"),
    ColorAxis("selected_role", "selected role (target/other/unknown)"),
    ColorAxis("gold_role", "gold role (target/other/unknown)"),
    ColorAxis("readout", "readout (2-opt vs 3-opt)"),
    ColorAxis("target_identity", "target identity"),
    ColorAxis("other_identity", "other identity"),
    ColorAxis("gold_label", "gold label"),
    ColorAxis("label_style", "label style"),
    # ── Continuous answer-distribution signals (colormap + colorbar) ─────────
    ColorAxis("top_choice_prob", "top-choice probability", continuous=True),
    ColorAxis("top_choice_logit", "top-choice logit", continuous=True),
    ColorAxis("vocab_entropy", "answer-distribution entropy (nats)", continuous=True),
    ColorAxis("answer_diversity", "answer-distribution diversity (Hill D1)", continuous=True),
    ColorAxis("inv_perplexity", "inverse perplexity of top option", continuous=True),
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
