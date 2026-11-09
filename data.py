"""Data utilities: synthetic smart-meter generator + loader for the SGCC dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic(n_consumers: int = 800, days: int = 150,
                       theft_frac: float = 0.12, seed: int = 0):
    """Return (X, y). X: (n_consumers, days) daily kWh. y: 1 = theft.

    Normal consumers follow a weekly cycle + seasonal drift + noise.
    Theft is injected using three patterns seen in real data:
      1. sudden persistent drop after a random day (meter tampering / bypass)
      2. intermittent zero-reading days
      3. erratic scaling of daily readings
    """
    rng = np.random.default_rng(seed)
    t = np.arange(days)
    X = np.zeros((n_consumers, days))
    y = (rng.random(n_consumers) < theft_frac).astype(int)

    for i in range(n_consumers):
        base = rng.uniform(4, 30)
        weekly = 1 + 0.15 * np.sin(2 * np.pi * t / 7 + rng.uniform(0, 6.28))
        season = 1 + 0.2 * np.sin(2 * np.pi * t / 365 + rng.uniform(0, 6.28))
        series = base * weekly * season * rng.normal(1, 0.08, days)

        if y[i]:
            kind = rng.integers(0, 3)
            if kind == 0:
                start = rng.integers(days // 4, days - 20)
                series[start:] *= rng.uniform(0.15, 0.6)
            elif kind == 1:
                mask = rng.random(days) < rng.uniform(0.15, 0.4)
                series[mask] = 0
            else:
                series *= rng.uniform(0.2, 1.4, days)
        X[i] = np.clip(series, 0, None)
    return X, y


def load_sgcc(csv_path: str):
    """Load the SGCC electricity-theft CSV (rows = consumers, FLAG column, date columns).

    Dataset: State Grid Corporation of China smart-meter data (public on Kaggle/GitHub).
    Missing values are linearly interpolated per consumer.
    """
    df = pd.read_csv(csv_path)
    y = df["FLAG"].to_numpy().astype(int)
    feats = df.drop(columns=[c for c in ["CONS_NO", "FLAG"] if c in df.columns])
    feats = feats.apply(pd.to_numeric, errors="coerce")
    feats = feats.T.interpolate(limit_direction="both").T.fillna(0)
    return feats.to_numpy(dtype=float), y
