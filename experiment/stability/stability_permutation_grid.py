"""Role-order (permutation) signatures recovered from rendered SESGO prompts.

A stability item is shown in 18 superficial variants: 3 LABEL styles crossed with
the 6 ROLE orders of its three options. The permutation axis is the order the
three "<marker> <option text>" lines appear in; stripping the marker leaves the
option texts in displayed order, which is a label-style-independent signature of
that role order.

Both readers here FAIL LOUDLY rather than guessing: a prompt that does not parse
to exactly 3 options, or a group that does not expose all 6 role orders, means the
prompt format drifted and the per-axis flip buckets would be silently mis-counted.
"""

from __future__ import annotations

from collections import defaultdict

from src.datasets.sesgo_eval import SesgoDataset, SesgoSample


def _option_texts(s: SesgoSample) -> list[str]:
    """The three "<marker> <option text>" option texts, in displayed order.

    A SESGO item always renders exactly three options, so anything else means the
    prompt format drifted and the role-order signature would be unreliable.
    """
    opts: list[str] = []
    for ln in (x.strip() for x in s.prompt_text.splitlines()):
        head, _, rest = ln.partition(" ")
        if rest and len(head) <= 3 and head.endswith(")"):
            opts.append(rest.strip())
    if len(opts) != 3:
        raise ValueError(
            f"Expected exactly 3 option lines in prompt_text, parsed {len(opts)} "
            f"for question_id={s.question_id} label_style={s.label_style!r}. "
            "The 'marker) text' option format changed; the role-order signature "
            "can no longer be recovered from prompt_text."
        )
    return opts


def permutation_signature(s: SesgoSample) -> str:
    """Label-style-independent signature of the role->position ordering.

    Drops the option markers and joins the option texts in displayed order. Fails
    loudly (never falls back) if the option lines cannot be parsed.
    """
    return " | ".join(_option_texts(s))


def assert_permutation_grid(dataset: SesgoDataset) -> None:
    """Fail loudly unless every group exposes exactly 6 role-order signatures.

    A stability group (question_id, condition, polarity) crosses 3 label styles
    with the 6 role permutations of 3 options. Holding the label style fixed, the
    role axis must therefore present all 6 permutations; fewer means the grid is
    incomplete and the per-axis flip buckets would silently under-count. Validated
    once per dataset so the assertion never re-runs inside the flip loops.
    """
    by_fixed: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for s in dataset.samples:
        key = (s.question_id, s.context_condition, s.question_polarity, s.label_style)
        by_fixed[key].add(permutation_signature(s))
    for (qid, cond, pol, style), sigs in by_fixed.items():
        if len(sigs) != 6:
            raise ValueError(
                f"Expected exactly 6 role-order permutations per group, found "
                f"{len(sigs)} for question_id={qid} condition={cond} "
                f"polarity={pol} label_style={style!r}. The role-permutation grid "
                "is incomplete; per-axis invariance would be mis-counted."
            )
