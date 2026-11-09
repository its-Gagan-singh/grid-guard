import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main
from app.data import generate_synthetic
from app.features import FEATURE_NAMES, extract
from app.monitor import psi
from app.train import run


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    run(out_dir=str(d), n_consumers=400)
    os.environ["MODEL_DIR"] = str(d)
    main._cache.clear()
    return TestClient(main.app)


def test_features_shape():
    assert extract(np.ones(60)).shape == (len(FEATURE_NAMES),)


def test_psi_identical_is_near_zero():
    x = np.random.default_rng(0).normal(size=1000)
    assert psi(x, x) < 0.01


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_theft_scores_higher_than_normal(client):
    X, y = generate_synthetic(n_consumers=200, seed=42)
    probs = {0: [], 1: []}
    for row, label in zip(X[:80], y[:80]):
        r = client.post("/predict", json={"consumer_id": "c", "daily_kwh": row.tolist()})
        assert r.status_code == 200
        probs[int(label)].append(r.json()["theft_probability"])
    assert np.mean(probs[1]) > np.mean(probs[0])


def test_rejects_short_series(client):
    r = client.post("/predict", json={"consumer_id": "c", "daily_kwh": [1.0] * 5})
    assert r.status_code == 422


def test_metrics_endpoint(client):
    assert "training" in client.get("/metrics").json()
