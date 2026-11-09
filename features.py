"""Hand-crafted features that capture common theft signatures in consumption series."""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "mean", "std", "cv", "min", "max", "p10", "p90",
    "zero_frac", "low_frac", "trend_ratio", "max_week_drop",
    "lag1_autocorr", "weekly_autocorr", "skew", "diff_abs_mean",
]


def _autocorr(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag + 1 or np.std(x) < 1e-9:
        return 0.0
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def extract(series) -> np.ndarray:
    x = np.nan_to_num(np.asarray(series, dtype=float), nan=0.0)
    n = len(x)
    mean, std = x.mean(), x.std()
    third = max(n // 3, 1)
    first, last = x[:third].mean(), x[-third:].mean()

    week_means = np.array([x[i:i + 7].mean() for i in range(0, n - 6, 7)]) if n >= 7 else x
    max_drop = 0.0
    if len(week_means) > 1:
        max_drop = float(np.max(-np.diff(week_means)) / (mean + 1e-9))

    skew = float(np.mean(((x - mean) / (std + 1e-9)) ** 3))
    feats = [
        mean, std, std / (mean + 1e-9), x.min(), x.max(),
        np.percentile(x, 10), np.percentile(x, 90),
        float(np.mean(x <= 1e-6)),
        float(np.mean(x < 0.3 * mean)) if mean > 0 else 1.0,
        last / (first + 1e-9),
        max_drop,
        _autocorr(x, 1), _autocorr(x, 7), skew,
        float(np.mean(np.abs(np.diff(x)))) / (mean + 1e-9),
    ]
    return np.array(feats, dtype=float)


def extract_matrix(rows) -> np.ndarray:
    return np.vstack([extract(r) for r in rows])
