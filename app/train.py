"""Train + evaluate the theft detector.

Usage:
    python -m app.train                       # synthetic data
    python -m app.train --sgcc data/data.csv  # real SGCC data
"""
from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from app.data import generate_synthetic, load_sgcc
from app.features import FEATURE_NAMES, extract_matrix


def run(out_dir: str = "models", sgcc: str | None = None, n_consumers: int = 800,
        seed: int = 0) -> dict:
    X, y = load_sgcc(sgcc) if sgcc else generate_synthetic(n_consumers=n_consumers, seed=seed)
    F = extract_matrix(X)

    # train / validation (threshold tuning) / test
    F_tr, F_tmp, y_tr, y_tmp = train_test_split(F, y, test_size=0.4, stratify=y, random_state=seed)
    F_val, F_te, y_val, y_te = train_test_split(F_tmp, y_tmp, test_size=0.5, stratify=y_tmp,
                                                random_state=seed)

    clf = GradientBoostingClassifier(random_state=seed).fit(F_tr, y_tr)

    # Unsupervised detector trained on normal consumers only (flags unseen patterns)
    iso = IsolationForest(n_estimators=200, random_state=seed).fit(F_tr[y_tr == 0])

    # Tune the decision threshold on validation data (never on test data)
    val_proba = clf.predict_proba(F_val)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 37)
    best_th = float(thresholds[int(np.argmax([f1_score(y_val, val_proba >= t) for t in thresholds]))])

    proba = clf.predict_proba(F_te)[:, 1]
    pred = proba >= best_th
    anomaly = -iso.score_samples(F_te)

    metrics = {
        "threshold": round(best_th, 3),
        "precision": round(float(precision_score(y_te, pred)), 4),
        "recall": round(float(recall_score(y_te, pred)), 4),
        "f1": round(float(f1_score(y_te, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_te, proba)), 4),
        "isolation_forest_roc_auc": round(float(roc_auc_score(y_te, anomaly)), 4),
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "theft_rate": round(float(y.mean()), 4),
        "data_source": "SGCC" if sgcc else "synthetic",
    }

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump({
        "clf": clf, "iso": iso, "threshold": best_th,
        "feature_names": FEATURE_NAMES, "reference": F_tr,
        "importances": dict(zip(FEATURE_NAMES, clf.feature_importances_.tolist())),
        "version": "1.0.0",
    }, os.path.join(out_dir, "model.joblib"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sgcc", default=None, help="path to SGCC csv")
    ap.add_argument("--out", default="models")
    ap.add_argument("--n", type=int, default=800)
    args = ap.parse_args()
    print(json.dumps(run(args.out, args.sgcc, args.n), indent=2))
