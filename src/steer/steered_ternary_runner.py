"""A TernaryChoiceRunner that runs the 3-option readout UNDER a steering hook.

The only change versus the base runner is the ONE call inside ``_run_triple``:
``compute_trajectories_batch`` becomes the existing
``compute_trajectories_batch_with_intervention(ids_list, iv)``. ``_divergent_scores``
and ``choose3`` are inherited unchanged, so the steered readout is identical to
the unsteered one except for the add-mode resid_post hook. We DO NOT write a new
hook — the add(alpha*v) intervention is fully implemented in the inference stack;
we only set the active intervention before calling the inherited ``choose3``.
"""

from __future__ import annotations

from src.inference import GeneratedTrajectory
from src.inference.interventions import Intervention
from src.ternary_choice import TernaryChoiceRunner


class SteeredTernaryChoiceRunner(TernaryChoiceRunner):
    """TernaryChoiceRunner whose teacher-forced triple runs under an intervention.

    Set ``active_intervention`` (or leave it None for the unsteered baseline);
    ``choose3`` then routes its single batched forward through the steered path.
    Inherits the runner with no own ``__init__`` (mirrors BinaryChoiceRunner /
    TernaryChoiceRunner), so it loads weights exactly like the base querier.
    """

    active_intervention: Intervention | None = None

    def _run_triple(self, ids_list: list[list[int]]) -> list[GeneratedTrajectory]:
        """Run the three teacher-forced trajectories, steered when an iv is set."""
        if self.active_intervention is None:
            return self.compute_trajectories_batch(ids_list)
        return self.compute_trajectories_batch_with_intervention(
            ids_list, self.active_intervention
        )
