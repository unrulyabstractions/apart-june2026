"""Ternary choice runner for 3-way (N-way) preference experiments.

Extends ModelRunner with a teacher-forced 3-way scoring method. This is the
binary recipe (BinaryChoiceRunner.choose / _run_pair) generalized from 2 to 3
labels: instead of one fork comparing two logprobs, we softmax over three
divergent option-label logprobs.
"""

from __future__ import annotations


from ..inference.model_runner import ModelRunner
from ..inference import GeneratedTrajectory
from ..binary_choice.choice_utils import encode_into_trajectory_ids
from ..common.choice import TernaryChoice
from ..common.profiler import profile


def _divergent_logprobs(
    trajs: list[GeneratedTrajectory],
) -> tuple[float, float, float]:
    """Read each trajectory's conditional logprob at the divergence position.

    The three forced continuations share `effective_prefix`, so their token-id
    sequences are identical up to the option-label token. The divergence
    position is the FIRST index where the three sequences are not all equal —
    that index holds each label's option-label token, and traj.logprobs[i] is
    its conditional logprob P(label_token | prompt + prefix).
    """
    id_seqs = [t.token_ids for t in trajs]
    min_len = min(len(s) for s in id_seqs)

    div_pos = min_len  # fallback: identical continuations (shouldn't happen)
    for i in range(min_len):
        first = id_seqs[0][i]
        if any(seq[i] != first for seq in id_seqs[1:]):
            div_pos = i
            break

    # logprobs[div_pos] is conditioned on the shared prefix, so the three are
    # directly comparable as P(label) and can be softmaxed.
    return (
        float(trajs[0].logprobs[div_pos]),
        float(trajs[1].logprobs[div_pos]),
        float(trajs[2].logprobs[div_pos]),
    )


class TernaryChoiceRunner(ModelRunner):
    """High-level runner for 3-way teacher-forced choice experiments.

    Inherits all ModelRunner functionality (no own __init__, like
    BinaryChoiceRunner). choose3() runs three forced-continuation trajectories
    (one per label) in a single batched forward pass and returns a
    TernaryChoice with a softmax distribution over the three options.
    """

    @profile("run_ternary_choice")
    def choose3(
        self,
        prompt: str,
        choice_prefix: str,
        labels: tuple[str, str, str],
    ) -> TernaryChoice:
        """Score which of three labels the model assigns highest probability.

        The model is NOT allowed to reason: skip_thinking_prefix auto-suppresses
        <think> for reasoning models, and each label is teacher-forced.

        Args:
            prompt:        The task / question text.
            choice_prefix: Shared response prefix, e.g. "Answer: ".
            labels:        Three candidate labels, ideally single-token-distinct
                           like ("0", "1", "2") or ("a", "b", "c").

        Returns:
            TernaryChoice with the labels and three divergent conditional
            logprobs (softmax → probs, argmax → choice_idx).
        """
        templated_prompt = self.apply_chat_template(prompt)

        # Auto-prepend skip thinking prefix for reasoning models (binary parity).
        effective_prefix = self.skip_thinking_prefix + choice_prefix

        # One forced continuation per label; jointly encode prompt+continuation
        # via the SAME helper binary uses (preserves BPE merges / BOS handling).
        ids_list = [
            encode_into_trajectory_ids(self, templated_prompt, effective_prefix + label)
            for label in labels
        ]

        trajs = self._run_triple(ids_list)
        logprobs = _divergent_logprobs(trajs)

        return TernaryChoice(labels=tuple(labels), logprobs=logprobs)

    @profile("_run_triple")
    def _run_triple(
        self, ids_list: list[list[int]]
    ) -> list[GeneratedTrajectory]:
        """Run three teacher-forced trajectories in one batched forward pass."""
        # Single batched call → 3 teacher-forced forward passes (binary uses 2).
        return self.compute_trajectories_batch(ids_list)
