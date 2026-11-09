"""Data-drift monitoring using the Population Stability Index (PSI)."""
from __future__ import annotations

import numpy as np


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI < 0.1 stable, 0.1-0.25 moderate drift, > 0.25 significant drift."""
    edges = np.unique(np.percentile(reference, np.linspace(0, 100, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_p = np.histogram(reference, edges)[0] / len(reference)
    cur_p = np.histogram(current, edges)[0] / max(len(current), 1)
    ref_p, cur_p = np.clip(ref_p, 1e-6, None), np.clip(cur_p, 1e-6, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def drift_report(reference_matrix: np.ndarray, current_matrix: np.ndarray, names):
    out = {}
    for j, name in enumerate(names):
        score = psi(reference_matrix[:, j], current_matrix[:, j])
        out[name] = {
            "psi": round(score, 4),
            "status": "stable" if score < 0.1 else "moderate" if score < 0.25 else "drift",
        }
    return out
