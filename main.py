"""FastAPI inference service for electricity-theft detection."""
from __future__ import annotations

import json
import os
import time
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.features import extract, extract_matrix
from app.monitor import drift_report

app = FastAPI(title="GridGuard", version="1.0.0",
              description="Electricity theft & anomaly detection from smart-meter data")

_cache: dict = {}
_stats = {"requests": 0, "flagged": 0, "latencies_ms": []}


def model_dir() -> str:
    return os.environ.get("MODEL_DIR", "models")


def bundle() -> dict:
    path = os.path.join(model_dir(), "model.joblib")
    if path not in _cache:
        if not os.path.exists(path):
            raise HTTPException(503, "Model not trained. Run: python -m app.train")
        _cache[path] = joblib.load(path)
    return _cache[path]


class MeterSeries(BaseModel):
    consumer_id: str
    daily_kwh: List[float] = Field(..., min_length=28, description="Daily kWh readings (>= 28 days)")


class Prediction(BaseModel):
    consumer_id: str
    theft_probability: float
    anomaly_score: float
    flagged: bool
    top_signals: dict


@app.get("/health")
def health():
    return {"status": "ok",
            "model_loaded": os.path.exists(os.path.join(model_dir(), "model.joblib"))}


@app.post("/predict", response_model=Prediction)
def predict(body: MeterSeries):
    t0 = time.perf_counter()
    b = bundle()
    feats = extract(body.daily_kwh).reshape(1, -1)
    prob = float(b["clf"].predict_proba(feats)[0, 1])
    anomaly = float(-b["iso"].score_samples(feats)[0])
    flagged = prob >= b["threshold"]

    # Explain: the most important model features, with this consumer's values
    top = sorted(b["importances"], key=b["importances"].get, reverse=True)[:3]
    signals = {k: round(float(feats[0, b["feature_names"].index(k)]), 4) for k in top}

    _stats["requests"] += 1
    _stats["flagged"] += int(flagged)
    _stats["latencies_ms"].append((time.perf_counter() - t0) * 1000)
    _stats["latencies_ms"] = _stats["latencies_ms"][-1000:]
    return Prediction(consumer_id=body.consumer_id, theft_probability=round(prob, 4),
                      anomaly_score=round(anomaly, 4), flagged=flagged, top_signals=signals)


@app.post("/drift")
def drift(batch: List[MeterSeries]):
    """Compare live series against the training distribution (PSI per feature)."""
    if len(batch) < 30:
        raise HTTPException(400, "Send at least 30 consumers for a meaningful drift check")
    b = bundle()
    current = extract_matrix([s.daily_kwh for s in batch])
    return drift_report(b["reference"], current, b["feature_names"])


@app.get("/metrics")
def metrics():
    path = os.path.join(model_dir(), "metrics.json")
    train_metrics = json.load(open(path)) if os.path.exists(path) else {}
    lat = _stats["latencies_ms"]
    return {
        "training": train_metrics,
        "serving": {
            "requests": _stats["requests"],
            "flagged": _stats["flagged"],
            "p50_ms": round(float(np.percentile(lat, 50)), 2) if lat else None,
            "p95_ms": round(float(np.percentile(lat, 95)), 2) if lat else None,
        },
    }
