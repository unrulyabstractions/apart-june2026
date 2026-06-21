"""The three dynamic states (pull / drift / potential) over the O_t trajectory.

Per Rios-Sialer App. H, the forking-paths O_t IS the system barycenter (pull),
so the three states fall out of the captured outcome histograms directly:

  pull_t      = ||O_t||_Λ                 strength of the attractor at position t
  drift_t     = ||O_t - O_0||_θ           accumulated deviation from the prior o_0
  potential_t = ||O_T - O_t||_θ           deviance still required to reach the end
  Δ_t         = ||O_t - O_{t-1}||_θ       forking magnitude (consecutive 1st diff)

All magnitudes are the dimension-normalized L2 norms (||·||/sqrt(dim)); we reuse
the shared src.dynamics metrics so the normalization matches the homogenization
study exactly. The most-forking position is argmax_t Δ_t (Eq. H.9).
"""

from __future__ import annotations

from src.dynamics import drift, normalized_norm, potential, pull

from .forking_path_types import DynamicStatesSeries


def compute_dynamic_states(
    prior_histogram: list[float],
    outcome_histograms: list[list[float]],
    final_histogram: list[float],
) -> DynamicStatesSeries:
    """Pull / drift / potential / forking-magnitude per base-path position."""
    pulls = [pull(o_t) for o_t in outcome_histograms]
    drifts = [drift(o_t, prior_histogram) for o_t in outcome_histograms]
    potentials = [potential(final_histogram, o_t) for o_t in outcome_histograms]

    # Forking magnitude Δ_t = ||O_t - O_{t-1}||_θ; the t=0 reference is o_0.
    forking_mag: list[float] = []
    prev = prior_histogram
    for o_t in outcome_histograms:
        diff = [a - b for a, b in zip(o_t, prev)]
        forking_mag.append(normalized_norm(diff))
        prev = o_t

    most = forking_mag.index(max(forking_mag)) if forking_mag else -1
    return DynamicStatesSeries(
        pull=pulls,
        drift=drifts,
        potential=potentials,
        forking_magnitude=forking_mag,
        most_forking_index=most,
    )
