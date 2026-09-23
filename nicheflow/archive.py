"""100-cell archive primitives. Acceptance and descriptor policies are explicit inputs."""
from dataclasses import dataclass, replace
import math
import numpy as np
from .spec import IntegrityError, SpecGap

SHAPE = (5, 5, 4)


def cell_id(cell):
    if len(cell) != 3 or any(type(x) is not int or not 0 <= x < n for x, n in zip(cell, SHAPE)):
        raise ValueError("invalid 5x5x4 niche")
    return cell[0] * 20 + cell[1] * 4 + cell[2]


def heterogeneity_bin(strong_nodes, total_nodes):
    if not 0 <= strong_nodes <= total_nodes or total_nodes <= 0:
        raise ValueError("invalid node counts")
    ratio = strong_nodes / total_nodes
    return 0 if ratio == 0 else 1 if ratio <= .3 else 2 if ratio <= .7 else 3


def cost_boundaries(observed_usd_costs):
    x = np.asarray(observed_usd_costs, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("measured, finite nonnegative USD costs required")
    return np.quantile(x, [.2, .4, .6, .8]).tolist()


def cost_bin(usd_cost, boundaries):
    if len(boundaries) != 4 or sorted(boundaries) != list(boundaries):
        raise ValueError("four ordered quantile boundaries required")
    if not all(math.isfinite(x) and x >= 0 for x in [usd_cost, *boundaries]):
        raise ValueError("invalid costs")
    # Ties stay in the same cell; empty quantile bins are allowed and recorded.
    return int(np.searchsorted(boundaries, usd_cost, side="left"))


def occupancy_density(cell, occupied, neighbors):
    """Neighborhood must be defined by the source-backed descriptor policy."""
    cell_id(cell)
    local = tuple(neighbors(cell))
    if not local or len(local) != len(set(local)):
        raise ValueError("nonempty unique neighborhood required")
    for n in local:
        cell_id(n)
    return sum(n in occupied for n in local) / len(local)


@dataclass(frozen=True)
class Evaluation:
    workflow: str
    cell: tuple[int, int, int]
    quality: float
    usd_cost: float
    receipt_ids: tuple[str, ...]
    split_role: str
    descriptor_version: str


def pareto_admission(candidate, incumbent):
    """3.2 Algorithm 1: empty cell or strict Pareto dominance (q up, USD down).

    Exposed as an explicit policy, not silently bound to a new source version.
    An incomparable candidate stays in the population but replaces no elite.
    """
    if incumbent is None:
        return True
    if candidate.cell != incumbent.cell:
        raise ValueError("Pareto admission compares entries within one niche")
    return (candidate.quality >= incumbent.quality and candidate.usd_cost <= incumbent.usd_cost
            and (candidate.quality > incumbent.quality or candidate.usd_cost < incumbent.usd_cost))


class Archive:
    def __init__(self):
        self.population = {i: {} for i in range(100)}
        self.elites, self.events = {}, []
        self.evaluations = {}

    def rebin(self, describe, acceptance):
        """Rebuild dynamic quantile cells in original evaluation order."""
        population, elites, evaluations, changes = {i: {} for i in range(100)}, {}, {}, []
        for version, old in self.evaluations.items():
            cell, descriptor_version = describe(old)
            new = replace(old, cell=tuple(cell), descriptor_version=descriptor_version)
            i = cell_id(new.cell)
            evaluations[version] = new
            population[i][version] = new
            if acceptance(new, elites.get(i)):
                elites[i] = new
            if old.cell != new.cell:
                changes.append({"workflow": version, "from": list(old.cell), "to": list(new.cell)})
        self.population, self.elites, self.evaluations = population, elites, evaluations
        return changes

    def evaluate(self, result, *, paid_receipts, acceptance=None, policy_source=None):
        i = cell_id(result.cell)
        if result.split_role not in {"development", "calibration"}:
            raise IntegrityError("held-out labels cannot update archive")
        if not result.receipt_ids or not set(result.receipt_ids) <= set(paid_receipts):
            raise IntegrityError("archive evaluation needs real execution receipts")
        if not math.isfinite(result.quality) or not 0 <= result.quality <= 1 or not math.isfinite(result.usd_cost) or result.usd_cost < 0:
            raise ValueError("invalid quality/cost")
        if not result.descriptor_version:
            raise ValueError("descriptor version required")
        if result.workflow in self.population[i]:
            if self.population[i][result.workflow] != result:
                raise IntegrityError("conflicting immutable evaluation")
            return None
        if acceptance is None or not policy_source:
            raise SpecGap("elite acceptance policy must be source-backed")
        current = self.elites.get(i)
        keep = acceptance(result, current)
        if type(keep) is not bool:
            raise ValueError("acceptance must return bool")
        self.population[i][result.workflow] = result
        self.evaluations[result.workflow] = result
        if keep:
            self.elites[i] = result
        event = {"cell": i, "candidate": result.workflow, "previous": current.workflow if current else None,
                 "accepted": keep, "policy_source": policy_source, "receipts": result.receipt_ids}
        self.events.append(event)
        return event
