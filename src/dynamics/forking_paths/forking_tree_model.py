"""Flat BaseSchema records for the left-to-right forking BRANCHING TREE.

The branching-tree view (arXiv:2601.06116, Fig. 11/22) renders a forking decision
as a left-to-right tree whose NODES are outcome distributions and whose EDGES are
labelled by the alternate continuation token that opens each branch. We keep the
tree as a FLAT node list (each node names its parent by index, never nesting a
``dict``/``list`` deeper than 1-D) so the whole structure roundtrips through
BaseSchema and crosses the capture/plot boundary like every other study record.

Three node levels mirror the paper's vocabulary:
  * ROOT  — an early base-path position, the shared conditioning frame o_root.
  * TRUNK — the decision/forking token position, its barycenter o_trunk.
  * BRANCH — one alternate continuation w at the trunk, its o_{t,w} and edge mass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common.base_schema import BaseSchema

# Node-level tags (the paper's root -> trunk -> branch vocabulary).
ROOT_LEVEL = "root"
TRUNK_LEVEL = "trunk"
BRANCH_LEVEL = "branch"


@dataclass
class BranchingTreeNode(BaseSchema):
    """One node of the branching tree: a labelled outcome distribution.

    ``outcome_histogram`` is the node's distribution over the outcome categories
    (same order as the tree's ``outcome_labels``); it is what the horizontal
    stacked bar at the node draws. ``edge_token`` is the continuation token that
    opens this node from its parent (empty for root/trunk); ``edge_weight`` is the
    probability mass on that edge (next-token p, used for edge thickness).
    ``parent_index`` is -1 for the root, else this node's parent in the flat list.
    """

    level: str  # ROOT_LEVEL / TRUNK_LEVEL / BRANCH_LEVEL
    label: str  # human label shown in the node (e.g. "his partner")
    edge_token: str  # the alternate token text on the incoming edge
    edge_weight: float  # p(x_t = edge_token): incoming-edge probability mass
    outcome_histogram: list[float]  # node distribution over outcome_labels
    n_rollouts: int  # support behind the histogram (sample count)
    parent_index: int  # index into the flat node list, -1 for the root


@dataclass
class BranchingTree(BaseSchema):
    """A flat left-to-right branching tree of outcome-distribution nodes.

    ``nodes`` is the flat list (root first, then trunk, then branches); each node
    points at its parent by index. ``outcome_labels`` fixes the bar/legend order
    so every node's ``outcome_histogram`` indexes the same categories. ``caption``
    is the one-line frame (the item / token the fork sits on).
    """

    item_question_id: str
    model: str
    outcome_labels: list[str]
    caption: str
    nodes: list[BranchingTreeNode] = field(default_factory=list)

    @property
    def root_index(self) -> int:
        """Index of the (single) root node, or -1 if the tree is empty."""
        for i, node in enumerate(self.nodes):
            if node.parent_index < 0:
                return i
        return -1

    def children_of(self, index: int) -> list[int]:
        """Indices of the nodes whose parent is ``index`` (left-to-right order)."""
        return [i for i, node in enumerate(self.nodes) if node.parent_index == index]
