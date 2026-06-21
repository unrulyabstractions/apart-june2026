"""Build a left-to-right BranchingTree from a captured ForkingTrajectory.

Pure logic (no I/O, no matplotlib): given the captured {O_t} and the trunk
(decision-token) position, assemble the root -> trunk -> branch nodes the
branching-tree plot renders. The trunk is the forking token; its top alternate
continuations become the branches, each carrying its own conditional outcome
distribution o_{t,w} (Eq. 1) and incoming-edge probability mass p(x_t = w). The
root is an earlier base-path position (or the prior o_0), the shared frame.

The forking token is, by definition, the position where re-sampling a DIFFERENT
next token would divert the outcome — i.e. where the alternates' o_{t,w} diverge
most. ``most_divergent_branch_index`` localizes exactly that, which is the right
trunk for a tree: a position whose alternates all agree makes a degenerate tree.
"""

from __future__ import annotations

from src.common.math import l2_distance

from .forking_path_types import AltTokenRollouts, ForkingTrajectory
from .forking_tree_model import (
    BRANCH_LEVEL,
    ROOT_LEVEL,
    TRUNK_LEVEL,
    BranchingTree,
    BranchingTreeNode,
)


def _clean_token(text: str) -> str:
    """Render a token for a node/edge label: visible newline, non-empty fallback."""
    return text.replace("\n", "\\n").strip() or "·"


def _branch_divergence(position) -> float:
    """Max pairwise L2 between a position's supported alternates' outcomes o_{t,w}.

    Zero when the position has fewer than two supported alternates or they all
    agree (no fork); large when re-sampling a different token genuinely diverts the
    outcome — the forking-paths signature of a decision token.
    """
    supported = [a.conditional_histogram for a in position.alternates if a.rollout_labels]
    return max(
        (l2_distance(x, y) for i, x in enumerate(supported) for y in supported[i + 1 :]),
        default=0.0,
    )


def most_divergent_branch_index(traj: ForkingTrajectory) -> int:
    """Position whose alternate continuations diverge most in outcome — the fork token.

    This is the model-agnostic, noise-robust trunk for the branching tree: unlike
    the consecutive-barycenter Δ_t (which a tiny model's per-token jitter dominates
    at t=0), it directly measures where a re-sampled token flips the answer.
    """
    if not traj.positions:
        return -1
    return max(range(len(traj.positions)), key=lambda i: _branch_divergence(traj.positions[i]))


def _ranked_branches(alternates: list[AltTokenRollouts], max_branches: int) -> list[AltTokenRollouts]:
    """Top branch continuations at the trunk: the base token plus the highest-mass alternates.

    We keep the realized base token (so the tree always shows the path the model
    took) and the most-probable *divergent* alternates, capped at ``max_branches``
    so the tree stays readable — mirroring the paper's 2-3 labelled branches.
    """
    supported = [a for a in alternates if a.rollout_labels]
    base = [a for a in supported if a.is_base_token]
    alts = sorted(
        (a for a in supported if not a.is_base_token),
        key=lambda a: a.token_prob,
        reverse=True,
    )
    ranked = base + alts
    return ranked[:max_branches] if max_branches > 0 else ranked


def _root_position(traj: ForkingTrajectory, trunk_index: int, root_back: int) -> tuple[str, list[float]]:
    """The root frame: a base-path position ``root_back`` tokens before the trunk.

    Falls back to the full-resample prior o_0 when the trunk sits at (or near) the
    start of the base path, so the root is always a well-defined shared frame.
    """
    root_pos = trunk_index - max(1, root_back)
    if 0 <= root_pos < len(traj.positions):
        pos = traj.positions[root_pos]
        return _clean_token(pos.base_token_text), pos.outcome_histogram
    return "prior (o₀)", traj.prior_histogram


def build_branching_tree(
    traj: ForkingTrajectory,
    trunk_index: int,
    max_branches: int = 3,
    root_back: int = 4,
) -> BranchingTree:
    """Assemble the root -> trunk -> branch tree for one trajectory's fork.

    ``trunk_index`` is the decision-token position (the detected forking token);
    ``max_branches`` caps how many alternate continuations become branches (the
    base token is always kept); ``root_back`` is how many positions before the
    trunk the shared root frame is read from.
    """
    trunk = traj.positions[trunk_index]
    nodes: list[BranchingTreeNode] = []

    # Root: an earlier shared frame (or the prior), the conditioning context.
    root_label, root_hist = _root_position(traj, trunk_index, root_back)
    nodes.append(
        BranchingTreeNode(
            level=ROOT_LEVEL, label=root_label, edge_token="", edge_weight=1.0,
            outcome_histogram=root_hist, n_rollouts=0, parent_index=-1,
        )
    )

    # Trunk: the decision token, its barycenter o_trunk (Eq. 2).
    trunk_idx = len(nodes)
    nodes.append(
        BranchingTreeNode(
            level=TRUNK_LEVEL, label=_clean_token(trunk.base_token_text),
            edge_token="", edge_weight=1.0,
            outcome_histogram=trunk.outcome_histogram,
            n_rollouts=sum(len(a.rollout_labels) for a in trunk.alternates),
            parent_index=0,
        )
    )

    # Branches: the top alternate continuations w, each with its o_{t,w} (Eq. 1).
    for alt in _ranked_branches(trunk.alternates, max_branches):
        suffix = " (base)" if alt.is_base_token else ""
        nodes.append(
            BranchingTreeNode(
                level=BRANCH_LEVEL,
                label=f"{_clean_token(alt.token_text)}{suffix}",
                edge_token=_clean_token(alt.token_text),
                edge_weight=alt.token_prob,
                outcome_histogram=alt.conditional_histogram,
                n_rollouts=len(alt.rollout_labels),
                parent_index=trunk_idx,
            )
        )

    caption = (f"fork at token t={trunk_index} “{_clean_token(trunk.base_token_text)}”"
               f" — root → trunk → branches")
    return BranchingTree(
        item_question_id=traj.item_question_id,
        model=traj.model,
        outcome_labels=list(traj.outcome_set.labels),
        caption=caption,
        nodes=nodes,
    )
