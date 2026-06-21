"""Seeded train/test split over SESGO contrastive question_ids.

Each of the 231 question_ids forms one balanced (scaffold vs no-scaffold) pair.
We split the PAIRS — by question_id — so a pair never straddles the train/test
boundary: the steering vector is fitted on TRAIN pairs and the causal claim is
tested on the held-out TEST pairs the vector never saw. The split is seeded and
sorted-then-shuffled, so the same (seed, fraction) always yields the same split,
and it is persisted in the vector bundle for Run/Verify to reuse verbatim.
"""

from __future__ import annotations

import random


def split_question_ids(
    question_ids: list[str], train_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Deterministically split unique question_ids into (train, test) lists.

    Sorts first (so input order can't perturb the split), shuffles with a seeded
    RNG, then takes the leading ``train_fraction`` as train. Returns both lists
    sorted for stable, diff-friendly persistence.
    """
    unique = sorted(set(question_ids))
    rng = random.Random(seed)
    shuffled = unique[:]
    rng.shuffle(shuffled)
    n_train = round(len(shuffled) * train_fraction)
    train = sorted(shuffled[:n_train])
    test = sorted(shuffled[n_train:])
    return train, test
