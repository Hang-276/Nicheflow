"""CA-SBMS estimator (§3.3). Undefined allocation is a required dependency."""
from collections import deque
from dataclasses import dataclass
import math
from .spec import SpecGap, IntegrityError


def curvature_ratio(gamma):
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must lie in [0,1]; no silent clipping")
    return 1.0 if gamma == 0 else -math.expm1(-gamma) / gamma


@dataclass(frozen=True)
class MarginalObservation:
    id: str
    element: str
    conditioning_set: tuple[str, ...]
    marginal: float
    empty_marginal: float
    utility_definition: str
    receipt_ids: tuple[str, ...]


class CurvatureWindow:
    def __init__(self, size, delta, confidence_constant, epsilon_max, utility_definition, boundary_policy="error"):
        if size < 1 or not 0 < delta < 1 or confidence_constant <= 0 or epsilon_max < 0:
            raise ValueError("invalid window/confidence parameters")
        if not utility_definition:
            raise SpecGap("shared utility and paired-marginal protocol must be defined")
        self.window = deque(maxlen=size)
        self.delta, self.constant, self.epsilon_max = delta, confidence_constant, epsilon_max
        self.utility_definition, self.seen = utility_definition, {}
        if boundary_policy not in {"error", "cap_at_one"}:
            raise ValueError("unknown curvature boundary policy")
        self.boundary_policy = boundary_policy

    def observe(self, observation, paid_receipts):
        o = observation
        if o.utility_definition != self.utility_definition or o.element in o.conditioning_set:
            raise ValueError("incompatible marginal observation")
        if not o.receipt_ids or not set(o.receipt_ids) <= set(paid_receipts):
            raise IntegrityError("marginals need auditable paid observation receipts")
        if not math.isfinite(o.empty_marginal) or o.empty_marginal <= 0:
            raise ValueError("empty-set marginal must be strictly positive")
        if not math.isfinite(o.marginal) or not 0 <= o.marginal <= o.empty_marginal:
            raise ValueError("observed marginals violate monotone diminishing-return domain")
        if o.id in self.seen:
            if self.seen[o.id] != o:
                raise IntegrityError("marginal id changed")
            return False
        if any(set(old.receipt_ids) == set(o.receipt_ids) for old in self.seen.values()):
            raise IntegrityError("renaming the same paid evidence cannot increase effective sample count")
        self.seen[o.id] = o
        self.window.append(o)
        return True

    def estimate(self, t):
        if t < 1:
            raise ValueError("t must be positive")
        m = len(self.window)
        if m == 0:
            return {"gamma": 1.0, "epsilon": None, "m": 0, "fallback": True,
                    "reason": "no_valid_marginal_observations", "h": curvature_ratio(1)}
        epsilon = self.constant * math.sqrt(math.log(t / self.delta) / m)
        empirical = max(1 - o.marginal / o.empty_marginal for o in self.window)
        fallback = epsilon > self.epsilon_max
        gamma = 1.0 if fallback else empirical + epsilon
        capped = gamma > 1 and self.boundary_policy == "cap_at_one"
        if capped:
            gamma = 1.0
        if not 0 <= gamma <= 1:
            raise SpecGap("empirical curvature plus confidence exceeds [0,1]; boundary rule unspecified")
        return {"gamma": gamma, "empirical": empirical, "epsilon": epsilon, "m": m,
                "fallback": fallback, "capped_at_one": capped, "h": curvature_ratio(gamma),
                "observation_ids": [o.id for o in self.window]}


class CASBMS:
    def __init__(self, estimator, allocation_policy=None, policy_source=None):
        self.estimator, self.policy = estimator, allocation_policy
        self.policy_source = policy_source

    def decide(self, t, public_context, remaining_budget):
        if self.policy is None or not self.policy_source:
            raise SpecGap("§3.3.7 does not specify h(gamma), context -> budget allocation")
        estimate = self.estimator.estimate(t)
        allocation = self.policy(dict(public_context), dict(estimate), remaining_budget)
        if set(allocation) != {"search", "deploy"}:
            raise ValueError("allocation must contain search and deploy proportions")
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in allocation.values()):
            raise ValueError("invalid budget allocation")
        if not math.isclose(sum(allocation.values()), 1, abs_tol=1e-10):
            raise ValueError("budget proportions must sum to one")
        return {"allocation": allocation, "estimate": estimate,
                "context": dict(public_context), "policy_source": self.policy_source}
