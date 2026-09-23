"""CS-DPP-NI §3.1: occupancy filtering then greedy UCB + logdet(K)."""
from dataclasses import dataclass
import math
import numpy as np
from .metrics import logdet_subset, psd_matrix
from .spec import SpecGap


@dataclass
class QualityEstimate:
    count: int = 0
    total: float = 0.0

    def observe(self, quality):
        if not math.isfinite(quality) or not 0 <= quality <= 1:
            raise ValueError("quality outside [0,1]")
        self.count += 1
        self.total += quality

    def ucb(self, t, exploration):
        if self.count == 0:
            raise SpecGap("zero-visit UCB initialization must be specified")
        return self.total / self.count + exploration * math.sqrt(math.log(t) / self.count)


def quality_weighted_kernel(similarity, weights):
    """Explicit D S D constructor; callers must justify how weights are obtained."""
    s = psd_matrix(similarity)
    w = np.asarray(weights, dtype=float)
    if w.shape != (len(s),) or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("finite nonnegative kernel weights required")
    return w[:, None] * s * w[None, :]


def rbf_ucb_kernel(features, ucbs, bandwidth):
    """Explicit inherited 3.2 kernel: UCB_i UCB_j exp(-gamma ||phi_i-phi_j||²).

    The caller supplies the feature definition and current 3.4 UCB estimates.
    No diagonal jitter or identity term changes the log-det objective.
    """
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or not len(x) or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError("finite nonempty feature matrix required")
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("positive RBF bandwidth required")
    squared_distance = ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)
    return quality_weighted_kernel(np.exp(-bandwidth * squared_distance), ucbs)


def select_niches(ids, populations, densities, estimates, kernel, *, t, k,
                  population_threshold, rho0, alpha, diversity_weight, exploration):
    ids = list(ids)
    if len(ids) != len(set(ids)) or t < 1 or k < 1 or population_threshold < 1:
        raise ValueError("unique ids, t>=1, k>=1 and positive population threshold required")
    if not all(math.isfinite(v) for v in (rho0, alpha, diversity_weight, exploration)) or rho0 <= 0 or alpha <= 0 or diversity_weight < 0 or exploration < 0:
        raise ValueError("invalid search parameters")
    matrix = psd_matrix(kernel)
    if len(matrix) != len(ids):
        raise ValueError("kernel/id shape mismatch")
    if any(not math.isfinite(densities[i]) or not 0 <= densities[i] <= 1 for i in ids):
        raise ValueError("density must be occupied-neighbor fraction")
    rho = rho0 * t ** (-alpha)
    active = [i for i in ids if populations[i] >= population_threshold]
    pool = [i for i in active if densities[i] <= rho]
    if len(pool) < k:
        raise SpecGap(f"candidate pool {len(pool)} < k={k}; no automatic k reduction")
    positions = {n: j for j, n in enumerate(ids)}
    ucbs = {i: estimates[i].ucb(t, exploration) for i in pool}
    selected, trace = [], []
    for _ in range(k):
        old = logdet_subset(matrix, [positions[i] for i in selected])
        scores = {}
        for i in pool:
            if i not in selected:
                gain = logdet_subset(matrix, [positions[j] for j in selected + [i]]) - old
                scores[i] = {"ucb": ucbs[i], "logdet_gain": gain,
                             "score": ucbs[i] + diversity_weight * gain}
        # Exact ties resolve by input order, which is frozen in the event.
        winner = max(scores, key=lambda i: scores[i]["score"])
        trace.append({"selected_before": list(selected), "scores": scores, "winner": winner})
        selected.append(winner)
    return {"selected": selected, "active": active, "pool": pool, "rho": rho, "trace": trace}
