"""Build + crash-safely persist the per-position RAW rollout dump.

Companion to ``forking_path_capture``: as each base-path position's branches are
sliced out of the flat rollout batch, we assemble a ``ForkingPositionDump`` (every
alternate's raw continuation text + parsed label + token info) and write it
IMMEDIATELY to ``<dump_dir>/pos_<NNN>.json`` via ``save_json_atomic``. Writing one
file per position as we go means a mid-run crash keeps every position completed so
far, and a reader never sees a torn file.
"""

from __future__ import annotations

from pathlib import Path

from src.common.file_io import save_json_atomic

from .forking_rollout_dump import ForkingPositionDump, RolloutDumpEntry


def build_position_dump(
    position: int,
    base_token_id: int,
    base_token_text: str,
    plan_rows: list[tuple],
    per_alt_texts: list[list[str]],
    per_alt_labels: list[list[str]],
) -> ForkingPositionDump:
    """Assemble one position's dump from its alternates' raw texts + labels.

    ``plan_rows[a]`` is ``(AltToken, prefix, n_samples)`` for alternate a;
    ``per_alt_texts[a]`` / ``per_alt_labels[a]`` are that alternate's parallel raw
    continuation texts and parsed outcome labels (same order, same length).
    """
    entries: list[RolloutDumpEntry] = []
    for alt_index, ((alt, _prefix, _n), texts, labels) in enumerate(
        zip(plan_rows, per_alt_texts, per_alt_labels)
    ):
        for sample_index, (text, label) in enumerate(zip(texts, labels)):
            entries.append(
                RolloutDumpEntry(
                    alt_index=alt_index,
                    token_id=alt.token_id,
                    token_text=alt.token_text,
                    token_prob=alt.prob,
                    is_base_token=alt.is_base,
                    sample_index=sample_index,
                    outcome_label=label,
                    raw_text=text,
                )
            )
    return ForkingPositionDump(
        position=position,
        base_token_id=base_token_id,
        base_token_text=base_token_text,
        entries=entries,
    )


def write_position_dump(dump_dir: Path, dump: ForkingPositionDump) -> Path:
    """Atomically write ONE position's dump to ``pos_<NNN>.json`` (crash-safe)."""
    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    out_path = dump_dir / f"pos_{dump.position:03d}.json"
    save_json_atomic(dump.to_dict(), out_path)
    return out_path
