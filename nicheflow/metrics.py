"""Observable metrics. No empirical maximum is presented as a true oracle."""
import math
import numpy as np


def psd_matrix(matrix):
    k = np.asarray(matrix, dtype=float)
    if k.ndim != 2 or k.shape[0] != k.shape[1] or not np.isfinite(k).all():
        raise ValueError("kernel must be square and finite")
    if not np.allclose(k, k.T, rtol=0, atol=1e-10):
        raise ValueError("kernel must be symmetric")
    if len(k) and np.linalg.eigvalsh(k).min() < -1e-10:
        raise ValueError("kernel must be positive semidefinite")
    return k


def logdet_subset(matrix, indices):
    k = psd_matrix(matrix)
    indices = list(indices)
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate subset index")
    if not indices:
        return 0.0  # det of the empty principal minor = 1
    sub = k[np.ix_(indices, indices)]
    if np.linalg.eigvalsh(sub).min() <= 0:
        raise ValueError("singular/non-positive principal minor; no jitter or I+K substitution")
    sign, value = np.linalg.slogdet(sub)
    if sign <= 0 or not math.isfinite(value):
        raise ValueError("undefined log determinant")
    return float(value)


def vendi_score(similarity):
    k = psd_matrix(similarity)
    if not len(k):
        return None
    if not np.allclose(np.diag(k), 1.0, rtol=0, atol=1e-9):
        raise ValueError("Vendi requires a normalized similarity kernel, not quality weights")
    values = np.linalg.eigvalsh(k / len(k))
    values = values[values > 0]  # exclude zero and numerical negative roundoff
    return float(np.exp(-np.sum(values * np.log(values))))


def hypervolume(points, reference):
    """Exact 2D HV, BOTH coordinates maximize; conversion must precede this call."""
    ref = np.asarray(reference, dtype=float)
    if ref.shape != (2,) or not np.isfinite(ref).all():
        raise ValueError("finite 2D reference required")
    clean = []
    for p in points:
        p = np.asarray(p, dtype=float)
        if p.shape != (2,) or not np.isfinite(p).all():
            raise ValueError("finite 2D points required")
        if np.all(p >= ref):
            clean.append(tuple(p))
    area, previous_x = 0.0, ref[0]
    for x in sorted({p[0] for p in clean}):
        height = max(p[1] for p in clean if p[0] >= x) - ref[1]
        area += (x - previous_x) * height
        previous_x = x
    return float(area)


def pass_at_k(n, c, k):
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("require 0<=correct<=samples and 1<=k<=samples")
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))
