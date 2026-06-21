"""Left-padding + attention-mask helpers shared by every batched inference path.

A batched forward pass over variable-length sequences must tell the model which
tokens are padding, or those pad tokens leak into attention and corrupt the real
logits. We LEFT-pad (not right-pad) for two reasons:

  1. Generation / next-token scoring reads position ``-1``; left-padding keeps the
     last REAL token at ``-1`` for every sample, so one shared index works.
  2. Structural-position indices (geometry) computed on the UNPADDED sequence map
     into the padded tensor by adding a per-sample left-pad offset — a single
     additive shift, no re-tokenization.

These helpers are the single source of truth for that padding so the backend, the
runner, and geometry all agree on the mask and the offsets.
"""

from __future__ import annotations

import torch

# Default micro-batch width for padded forward passes. ~32 keeps padded waste low
# while still amortizing kernel-launch / Python overhead across the batch.
DEFAULT_BATCH_SIZE = 32


def left_pad_batch(
    token_ids_batch: list[list[int]],
    pad_token_id: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Left-pad a ragged batch and build its attention mask + per-sample offsets.

    Args:
        token_ids_batch: variable-length token-id sequences.
        pad_token_id: id to pad with (model never attends to it; mask is 0 there).
        device: device for the returned tensors.

    Returns:
        (input_ids, attention_mask, offsets) where
          input_ids       [batch, max_len] left-padded token ids,
          attention_mask  [batch, max_len] 1 for real tokens, 0 for left pad,
          offsets         offsets[i] = max_len - len(seq_i): add to an UNPADDED
                          position to get the index of that token in row i.
    """
    max_len = max(len(ids) for ids in token_ids_batch)
    input_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    offsets: list[int] = []
    for ids in token_ids_batch:
        pad = max_len - len(ids)
        offsets.append(pad)
        input_rows.append([pad_token_id] * pad + list(ids))
        mask_rows.append([0] * pad + [1] * len(ids))
    input_ids = torch.tensor(input_rows, device=device)
    attention_mask = torch.tensor(mask_rows, device=device)
    return input_ids, attention_mask, offsets


def unpad_row(
    batched: torch.Tensor,
    row: int,
    seq_len: int,
    offset: int,
) -> torch.Tensor:
    """Slice one sample's REAL slice out of a left-padded batched tensor.

    Works for any tensor whose dim 0 is batch and dim 1 is the padded sequence
    (logits [batch, max_len, vocab] or activations [batch, max_len, d_model]).

    Args:
        batched: [batch, max_len, ...] tensor from a padded forward pass.
        row: which batch row this sample occupies.
        seq_len: original (unpadded) length of this sample.
        offset: this sample's left-pad offset (from ``left_pad_batch``).

    Returns:
        [seq_len, ...] slice for the real tokens, in original order.
    """
    return batched[row, offset : offset + seq_len]
