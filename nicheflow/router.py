"""Independent per-arm quality/cost ridge models and OFUL/Chebyshev §3.2."""
from __future__ import annotations
import math
import numpy as np
from .spec import SpecGap, IntegrityError


class RidgeHead:
    def __init__(self, dimension, regularization):
        if dimension < 1 or regularization <= 0:
            raise ValueError("positive dimension and regularization required")
        self.regularization = float(regularization)
        self.v = regularization * np.eye(dimension)
        self.b = np.zeros(dimension)
        self.count = 0

    def vector(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != self.b.shape or not np.isfinite(x).all():
            raise ValueError("invalid feature vector")
        return x

    def update(self, x, reward):
        x = self.vector(x)
        if not math.isfinite(reward):
            raise ValueError("non-finite reward")
        self.v += np.outer(x, x)
        self.b += x * reward
        self.count += 1

    def predict(self, x, beta):
        x = self.vector(x)
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("invalid confidence radius")
        mean = float(x @ np.linalg.solve(self.v, self.b))
        width = float(beta * math.sqrt(max(0, x @ np.linalg.solve(self.v, x))))
        return {"mean": mean, "width": width, "ucb": mean + width, "lcb": mean - width}

    def oful_beta(self, *, noise_bound, parameter_bound, delta):
        if not 0 < delta < 1 or noise_bound < 0 or parameter_bound < 0:
            raise ValueError("invalid OFUL assumptions")
        logdet = np.linalg.slogdet(self.v)[1] - len(self.b) * math.log(self.regularization)
        return noise_bound * math.sqrt(max(0, logdet + 2 * math.log(1 / delta))) + math.sqrt(self.regularization) * parameter_bound

    def state(self):
        return {"v": self.v.tolist(), "b": self.b.tolist(), "count": self.count,
                "regularization": self.regularization}


class BiGLinUCB:
    def __init__(self, dimension, regularization):
        self.dimension, self.regularization = dimension, regularization
        self.arms, self.receipts, self.epoch = {}, {}, 0

    def add_arm(self, version_id):
        if version_id in self.arms:
            return False
        self.arms[version_id] = (RidgeHead(self.dimension, self.regularization),
                                 RidgeHead(self.dimension, self.regularization))
        self.epoch += 1
        return True

    def choose(self, features, *, weights, beta_quality, beta_cost,
               cost_reward_definition, fallback_threshold=None, safe_arm=None):
        if not cost_reward_definition:
            raise SpecGap("cost-head reward direction requires a source-backed definition")
        if len(weights) != 2 or any(not math.isfinite(w) or w <= 0 for w in weights):
            raise ValueError("two positive objective weights required")
        if set(features) != set(self.arms) or not self.arms:
            raise ValueError("features must cover the current, nonempty arm set")
        scores = {}
        for arm, (q, c) in self.arms.items():
            bq = beta_quality[arm] if isinstance(beta_quality, dict) else beta_quality
            bc = beta_cost[arm] if isinstance(beta_cost, dict) else beta_cost
            pq, pc = q.predict(features[arm], bq), c.predict(features[arm], bc)
            scores[arm] = {"quality": pq, "cost_reward": pc,
                           "score": min(weights[0] * pq["ucb"], weights[1] * pc["ucb"])}
        chosen = max(scores, key=lambda a: scores[a]["score"])
        fallback = fallback_threshold is not None and scores[chosen]["score"] < fallback_threshold
        if fallback:
            if safe_arm not in self.arms:
                raise SpecGap("low-confidence fallback needs a valid specified safe workflow")
            chosen = safe_arm
        return {"arm": chosen, "epoch": self.epoch, "scores": scores, "fallback": fallback,
                "weights": list(weights), "cost_reward_definition": cost_reward_definition}

    def update(self, arm, features, quality, cost_reward, receipt_id):
        if not receipt_id or not 0 <= quality <= 1 or not math.isfinite(cost_reward):
            raise ValueError("valid receipt and rewards required")
        signature = (arm, tuple(features), quality, cost_reward)
        if receipt_id in self.receipts:
            if self.receipts[receipt_id] != signature:
                raise IntegrityError("receipt reused with different feedback")
            return False
        q, c = self.arms[arm]
        q.vector(features)
        q.update(features, quality)
        c.update(features, cost_reward)
        self.receipts[receipt_id] = signature
        return True

    def state(self):
        return {"epoch": self.epoch, "arms": {a: {"quality": q.state(), "cost": c.state()}
                for a, (q, c) in self.arms.items()}, "receipt_ids": sorted(self.receipts)}
