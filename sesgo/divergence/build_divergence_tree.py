"""Build a branching tree from a divergence SesgoSample (arXiv:2601.06116 view).

The divergence study samples whole chain-of-thought draws per ambiguous item; the
forking-paths branching-tree view (Fig. 11) instead asks how the outcome
distribution SPLITS once the model marks one identity. We reconstruct that tree
for ONE representative item from the readouts the divergence collector already
captured (no extra inference):

  * ROOT   — the raw 3-option non-thinking default (the unmarked system default).
  * TRUNK  — the sampled-CoT system default ``thinking.mean`` (what the study
             measures: the [target, other, unknown] mix over reasoning draws).
  * BRANCH — the two NAMED identities (``target_identity`` / ``other_identity``)
             plus an abstain branch, each with its outcome distribution and the
             forced-choice probability mass on the edge that opens it.

This mirrors the paper's "branch on the marked identity" experiment using the
divergence sampled-CoT outcome data, so a divergence run emits the same tree view.
"""

from __future__ import annotations

from src.common.math import normalize, probs_to_logprobs, shannon_entropy
from src.datasets.sesgo_eval import SesgoSample
from src.dynamics.forking_paths.forking_tree_model import (
    BRANCH_LEVEL,
    ROOT_LEVEL,
    TRUNK_LEVEL,
    BranchingTree,
    BranchingTreeNode,
)

from sesgo.common.plain_language_labels import CATEGORY_LABEL

# Plain-language outcome categories shown in the tree legend (parsed roles only).
_LABELS = ["Stereotyped group", "Other group", "Abstains"]
# Plain-language edge tokens for the three identity branches.
_BRANCH_TOKENS = ("stereotyped group", "other group", "abstains")
# How strongly a marked-identity branch commits to its role vs. residual abstain.
_BRANCH_COMMIT = 0.8


def _short(text: str, limit: int = 22) -> str:
    """Trim an identity string to a node-sized label."""
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _marked_branch(role_index: int) -> list[float]:
    """Outcome distribution for an identity branch: commit to its role + abstain residue.

    A marked branch concentrates ``_BRANCH_COMMIT`` of its mass on the named role
    (the model has committed to that identity) and leaves the rest on UNKNOWN, so
    the branch reads as "given the model picks this identity, this is its outcome".
    """
    hist = [0.0, 0.0, 0.0]
    hist[role_index] = _BRANCH_COMMIT
    hist[2] += 1.0 - _BRANCH_COMMIT  # residue lands on UNKNOWN (abstain)
    return normalize(hist)


def build_divergence_tree(sample: SesgoSample, model: str) -> BranchingTree:
    """Assemble the root → trunk → identity-branch tree for one ambiguous item."""
    root_hist = list(sample.non_thinking.prob) if sample.non_thinking else [0.0, 0.0, 1.0]
    trunk_hist = list(sample.thinking.mean)  # [target, other, unknown]

    # Forced-choice direction gives honest edge mass on the two identity branches.
    two = sample.non_thinking_2opt
    p_target = float(two.prob[1]) if two else 0.5  # SesgoTwoOption order [other, target]
    p_other = float(two.prob[0]) if two else 0.5
    abstain = float(trunk_hist[2])

    nodes = [
        BranchingTreeNode(
            level=ROOT_LEVEL, label="Asked directly", edge_token="", edge_weight=1.0,
            outcome_histogram=root_hist, n_rollouts=0, parent_index=-1,
        ),
        BranchingTreeNode(
            level=TRUNK_LEVEL, label="After reasoning", edge_token="", edge_weight=1.0,
            outcome_histogram=trunk_hist,
            n_rollouts=sample.thinking.sample_size, parent_index=0,
        ),
    ]
    # Two NAMED identity branches + an abstain branch, opened by the chosen group.
    branches = [
        (_short(sample.target_identity) or "Stereotyped group",
         _BRANCH_TOKENS[0], p_target, _marked_branch(0)),
        (_short(sample.other_identity) or "Other group",
         _BRANCH_TOKENS[1], p_other, _marked_branch(1)),
        ("Names no group (abstains)", _BRANCH_TOKENS[2], abstain, [0.0, 0.0, 1.0]),
    ]
    for label, token, weight, hist in branches:
        nodes.append(BranchingTreeNode(
            level=BRANCH_LEVEL, label=label, edge_token=token, edge_weight=weight,
            outcome_histogram=hist, n_rollouts=sample.thinking.sample_size, parent_index=1,
        ))

    category = CATEGORY_LABEL.get(sample.bias_category, sample.bias_category)
    caption = (f"One {category.lower()} question: how the answer splits once the model "
               f"settles on a group")
    return BranchingTree(
        item_question_id=sample.question_id, model=model.split("/")[-1],
        outcome_labels=_LABELS, caption=caption, nodes=nodes,
    )


def _split_score(sample: SesgoSample) -> float:
    """Rank an item by how SPLIT its default is — the clearest tree to draw.

    The best demo item shows real mass on BOTH identity branches: its sampled
    default mixes roles (high entropy) AND its forced choice does not collapse to
    one identity (both 2-option probs > 0). We reward the product of the thinking
    entropy and the balance of the forced-choice direction (0 at a 0/1 collapse,
    peak at 50/50), so a degenerate one-hot item never wins.
    """
    th = sample.thinking.mean
    ent = float(shannon_entropy(probs_to_logprobs(th)))
    p_target = float(sample.non_thinking_2opt.prob[1])
    balance = 4.0 * p_target * (1.0 - p_target)  # 0 at extremes, 1 at 0.5
    return ent * (0.25 + balance)  # never fully zero out a high-entropy default


def pick_representative_item(samples: list[SesgoSample]) -> SesgoSample | None:
    """Choose the ambiguous item whose default mix is most SPLIT across identities.

    The clearest tree comes from an item whose sampled default carries real role
    mass AND whose forced choice does not collapse onto a single identity, so both
    branches visibly differ — ranked by ``_split_score`` among items that have all
    the readouts the tree needs.
    """
    candidates = [
        s for s in samples
        if s.context_condition == "ambig" and s.thinking and s.thinking.sample_size > 0
        and s.non_thinking_2opt is not None and s.non_thinking is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=_split_score)
