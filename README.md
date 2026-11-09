# GridGuard: Electricity Theft Detection API

Detects suspicious smart-meter consumption (meter tampering, bypass, intermittent zero readings)
and serves predictions through a production-style FastAPI service with drift monitoring, tests,
Docker packaging and CI.

India's distribution utilities lose a large share of energy to theft and billing losses (AT&C losses).
GridGuard shows how an ML detector can be shipped as a monitored service rather than a notebook.

## Features
- **Two detectors**: supervised Gradient Boosting + unsupervised Isolation Forest (trained on normal users only)
- **15 engineered features** targeting theft signatures (sudden drops, zero days, weekly-pattern loss, trend ratio)
- **Threshold tuned on a validation split**, metrics reported on a held-out test split
- **Explainable output**: top contributing signals returned with each prediction
- **Drift monitoring** via Population Stability Index (`POST /drift`)
- **Serving metrics** (request count, flag rate, p50/p95 latency) at `GET /metrics`
- Pytest suite, Dockerfile, GitHub Actions CI

## Quick start
```bash
pip install -r requirements.txt
python -m app.train                  # trains on synthetic data
uvicorn app.main:app --reload        # http://localhost:8000/docs
pytest -q
```

Train on the real SGCC dataset (download it separately and place the CSV at `data/data.csv`):
```bash
python -m app.train --sgcc data/data.csv
```

Docker:
```bash
docker build -t grid-guard . && docker run -p 8000:8000 grid-guard
```

## API
```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"consumer_id":"C123","daily_kwh":[12,11,13,12,10,11,12,12,11,13,12,10,11,12,12,11,13,12,10,11,12,3,2,3,2,3,2,3]}'
```

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Theft probability, anomaly score, flag, top signals |
| `POST /drift` | PSI drift report for a batch (>= 30 consumers) |
| `GET /metrics` | Training metrics + serving latency stats |
| `GET /health` | Liveness / model-loaded check |

## Results
Fill this table with YOUR measured numbers after running on SGCC (do not copy synthetic-data numbers
as real-world results; synthetic data is much easier than real data).

| Dataset | Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| SGCC | Gradient Boosting | | | | |

Serving latency (measure with `GET /metrics` after a load test): p95 = ___ ms

## Limitations and ethics
- Flagging a consumer is a *lead for inspection*, not proof of theft; a human must verify before any action.
- Models trained on one region's data can drift or be biased on another's; monitor PSI and retrain.
- Synthetic data is for development and tests only.

## Roadmap
LSTM autoencoder for sequence anomalies, short-term load forecasting, model registry, Prometheus export.
