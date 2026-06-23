"""Plan every (position t, alternate token w) branch for one prompt's base path.

Decodes the greedy base thinking path once (recording each position's full-vocab
logits), reads the top-K alternate tokens per position, and builds the forced
branch PREFIX string for each (t, w): the chat-templated prompt + base tokens
x*_{<t} with the t-th token replaced by w. The capture driver then samples S
continuations from each prefix. Per-position sample counts can be inflated near a
suspected change point (the highest-entropy base token) per the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from src.common import BaseSchema
from src.common.math import shannon_entropy, probs_to_logprobs, normalize, q_diversity
from src.datasets.prompt import SesgoPromptSample
from src.datasets.prompt.sesgo_prompt_localization import sesgo_answer_cue
from src.inference import ModelRunner
from src.inference.answer_parser import answer_segment, parse_answer

from .forking_top_k_tokens import AltToken, alternates_at_position

# Reasonable default cap on a forked continuation's reasoning before it must commit (keeps a
# resampled thought from running away; the base path itself comes pre-capped from the readout).
DEFAULT_FORK_MAX_NEW_TOKENS = 512


@dataclass
class BranchPlan:
    """All branches for one prompt: per-position (alt, prefix, n_samples) rows."""

    base_path_text: str
    base_token_ids: list[int]
    base_token_texts: list[str]
    prompt_token_count: int
    # rows_per_position[t] = list of (AltToken, branch_prefix_str, n_samples)
    rows_per_position: list[list[tuple[AltToken, str, int]]]


def samples_per_position(
    base_logits: list[torch.Tensor], near_window: int, n_samples: int
) -> list[int]:
    """Per-position sample budget: +50% within ``near_window`` of the peak-entropy token.

    The paper samples more continuations near suspected change points; we proxy
    "suspected" by the base-path position whose next-token distribution has the
    highest Shannon entropy (the model was most uncertain there).
    """
    n = len(base_logits)
    counts = [n_samples] * n
    if near_window <= 0 or n == 0:
        return counts
    ents = [
        shannon_entropy(probs_to_logprobs(normalize(torch.softmax(row.float(), dim=-1).tolist())))
        for row in base_logits
    ]
    peak = max(range(n), key=lambda i: ents[i])
    for i in range(max(0, peak - near_window), min(n, peak + near_window + 1)):
        counts[i] = int(round(n_samples * 1.5))
    return counts


def build_branch_plan(
    runner: ModelRunner,
    sample: SesgoPromptSample,
    near_window: int,
    n_samples: int,
    base_max_new_tokens: int = 256,
    max_positions: int = 0,
) -> BranchPlan:
    """Greedy-decode the base path and enumerate every (t, w) branch prefix.

    ``max_positions`` (0 == all) caps how many leading base-path positions are
    branched — the EXPENSIVE part — while the full base path is still decoded for
    the text strip. This is the local-pilot cost knob; the cloud run uses 0.
    """
    # Greedy base thinking decode to get the realized token ids + prefill split.
    # (KV-cached generation returns logprobs only, so full_logits is None here.)
    gen = runner.generate_trajectory_from_prompt(
        sample.text, max_new_tokens=base_max_new_tokens, temperature=0.0
    )
    prefill_len = gen.prefill_length  # where the generated base path begins
    full_ids = list(gen.token_ids)
    base_ids = full_ids[prefill_len:]
    if max_positions > 0:
        base_ids = base_ids[:max_positions]
    return _enumerate_branches(runner, full_ids, prefill_len, base_ids, near_window, n_samples)


def build_branch_plan_from_text(
    runner: ModelRunner,
    templated_prompt: str,
    base_path_text: str,
    near_window: int,
    n_samples: int,
    max_positions: int = 0,
) -> BranchPlan:
    """Build the branch plan from an ALREADY-decoded base path (e.g. the ``response_text`` a
    stability/readout run already produced) — NO re-generation. The expensive greedy decode is
    skipped; we just re-encode the (templated) prompt + the stored response and teacher-force
    once for the per-position fork logits. This is the cheap, item-parallel path: every stored
    response is an independent base path, so a whole sweep of items forks without re-decoding.

    `templated_prompt` is the prompt EXACTLY as the model saw it (the readout's `prompt_text`),
    and `base_path_text` is that run's `response_text`."""
    prefill_ids = runner.encode_ids(templated_prompt, add_special_tokens=True)
    base_ids = runner.encode_ids(base_path_text, add_special_tokens=False)
    if max_positions > 0:
        base_ids = base_ids[:max_positions]
    prefill_len = len(prefill_ids)
    full_ids = list(prefill_ids) + list(base_ids)
    return _enumerate_branches(runner, full_ids, prefill_len, list(base_ids), near_window, n_samples)


def _enumerate_branches(
    runner: ModelRunner, full_ids: list[int], prefill_len: int,
    base_ids: list[int], near_window: int, n_samples: int,
) -> BranchPlan:
    """Teacher-force the realized prompt+base sequence ONCE and enumerate every (t, w) branch
    prefix. Shared by the generate (`build_branch_plan`) and read-from-text
    (`build_branch_plan_from_text`) entry points — only the base-id source differs."""
    # One teacher-forced pass over the realized sequence recovers the per-position full-vocab
    # logits (the top-K source). full_logits[i] PREDICTS token i; for base position t the
    # predicting row is at absolute index prefill_len + t.
    traj = runner.compute_trajectory(full_ids[: prefill_len + len(base_ids)])
    pred_rows = [traj.full_logits[prefill_len + t] for t in range(len(base_ids))]

    base_token_texts = [runner.decode_ids([tid]) for tid in base_ids]
    counts = samples_per_position(pred_rows, near_window, n_samples)

    # The backend's batched decode re-encodes the prefix string with add_special_tokens=True,
    # so a leading BOS in the decoded text would be duplicated; drop it here and let
    # re-encoding restore exactly one.
    bos = runner.bos_token_id

    rows_per_position: list[list[tuple[AltToken, str, int]]] = []
    for t, tid in enumerate(base_ids):
        alts = alternates_at_position(pred_rows[t], tid, runner.decode_ids)
        # Forced prefix = templated prompt + base tokens up to t (exclusive) + w.
        head_ids = list(full_ids[: prefill_len + t])
        if bos is not None and head_ids and head_ids[0] == bos:
            head_ids = head_ids[1:]
        rows: list[tuple[AltToken, str, int]] = []
        for alt in alts:
            prefix = runner.decode_ids(head_ids + [alt.token_id])
            rows.append((alt, prefix, counts[t]))
        rows_per_position.append(rows)

    return BranchPlan(
        base_path_text=runner.decode_ids(base_ids),
        base_token_ids=list(base_ids),
        base_token_texts=base_token_texts,
        prompt_token_count=prefill_len,
        rows_per_position=rows_per_position,
    )


@dataclass
class ImmediateCommit(BaseSchema):
    """What the model would commit to if it STOPPED reasoning at base position ``position``."""

    position: int
    label: str            # option marker, e.g. "c)" ("" if none parsed)
    choice: str           # target / other / unknown / invalid
    label_prob: float     # probability of the committed answer token | the force-closed context
    vocab_diversity: float  # effective number of next-token choices there (exp Shannon entropy)


def immediate_commits_from_text(
    runner: ModelRunner,
    templated_prompt: str,
    base_path_text: str,
    sample: SesgoPromptSample,
    max_answer_tokens: int = 64,
) -> list[ImmediateCommit]:
    """At EVERY base-path position t, force-close the reasoning and greedily read what the model
    would commit to IMMEDIATELY if it stopped thinking right there — WITH the same measurements
    the readout reports (label_prob, vocab_diversity), so each per-position commit is directly
    comparable to the model's final answer. The deterministic "answer-so-far" curve that
    complements the sampled O_t: it shows how the committed answer AND its confidence firm up as
    reasoning unfolds. `templated_prompt` + `base_path_text` are the readout's prompt_text +
    response_text. One batched greedy decode for all positions, then a per-position teacher-forced
    read at the committed answer token (identical to run_greedy_readout's measurement, incl. the
    full_logits[pos] alignment)."""
    close_ids = runner.encode_ids(
        "\n" + (runner.reasoning_close_marker or "</think>") + "\n\n", add_special_tokens=False)
    prefill_ids = runner.encode_ids(templated_prompt, add_special_tokens=True)
    base_ids = runner.encode_ids(base_path_text, add_special_tokens=False)
    cue = sesgo_answer_cue(sample.language)
    roles = [r.value if hasattr(r, "value") else r for r in sample.position_labels]

    prompt_ids_per_t = [list(prefill_ids) + list(base_ids[:t]) + list(close_ids)
                        for t in range(len(base_ids))]
    answers = runner.continue_from_text_batch(
        [runner.decode_ids(p) for p in prompt_ids_per_t],
        max_new_tokens=max_answer_tokens, temperature=0.0,
    )

    commits: list[ImmediateCommit] = []
    for t, (pids, ans) in enumerate(zip(prompt_ids_per_t, answers)):
        label, choice, off = parse_answer(ans, sample.option_labels, roles, cue)
        ans_ids = runner.encode_ids(ans, add_special_tokens=False)
        full_ids = list(pids) + list(ans_ids)
        # answer-token position: parsed offset, else the first post-think token
        prefix = ans[:off] if off >= 0 else ans[: len(ans) - len(answer_segment(ans))]
        pos = len(pids) + len(runner.encode_ids(prefix, add_special_tokens=False))
        pos = min(max(pos, len(pids)), len(full_ids) - 1)
        ct = runner.compute_trajectory(full_ids[: pos + 1])
        dist = torch.log_softmax(ct.full_logits[pos], dim=-1)
        commits.append(ImmediateCommit(
            position=t, label=label, choice=choice,
            label_prob=math.exp(float(ct.logprobs[pos])),
            vocab_diversity=float(q_diversity(dist, 1.0)),
        ))
    return commits
