"""Small CPU MLP proxy. Hyperparameters/screening threshold are explicit inputs."""
import numpy as np
from .spec import IntegrityError


class MLPProxy:
    def __init__(self, dimension, hidden, seed):
        if dimension < 1 or hidden < 1:
            raise ValueError("positive dimensions required")
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, 1 / np.sqrt(dimension), (dimension, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, 1 / np.sqrt(hidden), hidden)
        self.b2 = 0.
        self.training_receipts = ()

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape[-1] != len(self.w1) or not np.isfinite(x).all():
            raise ValueError("invalid proxy features")
        h = np.tanh(x @ self.w1 + self.b1)
        # Stable logistic output, bounding predictions but never observed scores.
        z = h @ self.w2 + self.b2
        return np.exp(-np.logaddexp(0, -z))

    def fit(self, x, y, *, receipt_ids, roles, paid_receipts, steps, learning_rate):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if x.ndim != 2 or x.shape[1] != len(self.w1) or y.shape != (len(x),) or len(x) == 0:
            raise ValueError("invalid proxy training shape")
        if not np.isfinite(x).all() or not np.isfinite(y).all() or (y < 0).any() or (y > 1).any():
            raise ValueError("invalid proxy training values")
        if len(receipt_ids) != len(x) or len(set(receipt_ids)) != len(x) or not set(receipt_ids) <= set(paid_receipts):
            raise IntegrityError("proxy labels need unique paid observations")
        if len(roles) != len(x) or any(role not in {"development", "calibration"} for role in roles):
            raise IntegrityError("held-out labels cannot fit proxy")
        if steps < 1 or not 0 < learning_rate <= 1:
            raise ValueError("invalid optimizer configuration")
        for _ in range(steps):
            h = np.tanh(x @ self.w1 + self.b1)
            p = self.predict(x)
            dz = (p - y) / len(x)  # Bernoulli cross-entropy gradient
            dh = dz[:, None] * self.w2 * (1 - h * h)
            self.w2 -= learning_rate * (h.T @ dz)
            self.b2 -= learning_rate * dz.sum()
            self.w1 -= learning_rate * (x.T @ dh)
            self.b1 -= learning_rate * dh.sum(axis=0)
        self.training_receipts = tuple(receipt_ids)

    def screen(self, candidates, features, threshold):
        if not self.training_receipts:
            raise ValueError("unfitted proxy; bootstrap screening rule not supplied")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold outside [0,1]")
        predictions = self.predict(features)
        if len(predictions) != len(candidates):
            raise ValueError("candidate-feature mismatch")
        return [{"candidate": c, "predicted_quality": float(p), "selected": bool(p >= threshold),
                 "training_receipts": self.training_receipts} for c, p in zip(candidates, predictions)]

    def screen_fraction(self, candidates, features, fraction):
        """Inherited 3.2 Algorithm 1 exposes screening proportion n% as an input."""
        import math
        if not self.training_receipts or not 0 < fraction <= 1:
            raise ValueError("fitted proxy and fraction in (0,1] required")
        predictions = self.predict(features)
        if len(predictions) != len(candidates):
            raise ValueError("candidate-feature mismatch")
        selected = set(sorted(range(len(candidates)), key=lambda i: -predictions[i])[:math.ceil(fraction * len(candidates))])
        return [{"candidate": c, "predicted_quality": float(p), "selected": i in selected,
                 "fraction": fraction, "training_receipts": self.training_receipts}
                for i, (c, p) in enumerate(zip(candidates, predictions))]
