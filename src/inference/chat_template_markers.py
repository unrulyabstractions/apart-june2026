"""Model-aware structural chat-template markers for residual-geometry probes.

Each instruct family delimits the assistant turn with its OWN single special
token, and only reasoning models wrap a scratch-pad in <think>/</think>:

    family   assistant-turn marker   think markers
    qwen     <|im_start|>            <think> / </think>
    llama    <|start_header_id|>     (none)
    gemma    <start_of_turn>         (none)
    mistral  [/INST]                 (none)

`structural_markers_for(model_name)` resolves the right marker strings from the
bare model name so geometry's position-finder stays model-agnostic: it asks the
runner for the markers instead of hardcoding Qwen's. Every marker here is a
SINGLE token in its family's vocab (verified), so the existing single-token
lookup in the geometry collector finds it directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common import BaseSchema


@dataclass
class ChatTemplateMarkers(BaseSchema):
    """Structural marker token strings for one model family.

    `turn_marker` is the special token that opens the assistant turn (its LAST
    occurrence in a forced sequence is the assistant turn boundary). The two
    think markers are empty strings for non-reasoning families (no scratch-pad),
    which the consumer treats as "skip this position".

    The geometry probe additionally splits the coarse turn boundary into the
    individual chat-template tokens that delimit it: `turn_end` closes the
    *previous* (user) turn, and `assistant_role` is the surface role word the
    template emits right after `turn_marker` to open the assistant turn (Qwen/
    ChatML "assistant", Gemma "model", Llama "assistant"). Both are empty for
    families whose template has no such distinct token, so the consumer skips
    that position rather than guessing.
    """

    turn_marker: str
    think_open: str = ""
    think_close: str = ""
    turn_end: str = ""  # token closing the PREVIOUS turn (e.g. <|im_end|>)
    assistant_role: str = ""  # role word opening the assistant turn ("assistant")


# Bare-name substrings -> the family's structural markers. Ordered most-specific
# first; the first family whose key appears in the lowercased name wins.
_FAMILY_MARKERS: list[tuple[str, ChatTemplateMarkers]] = [
    (
        "gemma",
        ChatTemplateMarkers(
            turn_marker="<start_of_turn>",
            turn_end="<end_of_turn>",
            assistant_role="model",
        ),
    ),
    (
        "llama",
        ChatTemplateMarkers(
            turn_marker="<|start_header_id|>",
            turn_end="<|eot_id|>",
            assistant_role="assistant",
        ),
    ),
    ("mistral", ChatTemplateMarkers(turn_marker="[/INST]")),
    ("mixtral", ChatTemplateMarkers(turn_marker="[/INST]")),
    # Qwen is the reasoning family: it alone carries the think scratch-pad markers.
    (
        "qwen",
        ChatTemplateMarkers(
            turn_marker="<|im_start|>",
            think_open="<think>",
            think_close="</think>",
            turn_end="<|im_end|>",
            assistant_role="assistant",
        ),
    ),
]

# Qwen's ChatML turn marker is the safest cross-family default: it is the most
# common instruct convention, so unknown ChatML-style models still find a turn.
_DEFAULT_MARKERS = ChatTemplateMarkers(
    turn_marker="<|im_start|>", turn_end="<|im_end|>", assistant_role="assistant"
)


def structural_markers_for(model_name: str) -> ChatTemplateMarkers:
    """Pick the structural chat-template markers for a bare/full model name."""
    name = model_name.lower()
    for key, markers in _FAMILY_MARKERS:
        if key in name:
            return markers
    return _DEFAULT_MARKERS
