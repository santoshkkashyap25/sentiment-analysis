"""Adversarial stress and edge-case test suite for API routes, SQLite concurrency, and monitoring."""

import inspect
import sqlite3
import sys
import tempfile
import threading
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
import scipy.sparse as sp
from fastapi.testclient import TestClient

import src.api.app as app_module
from src.api.app import app
from src.monitoring.drift_detection import ModelMonitor, RetrainingStrategy


@pytest.fixture
def client():
    return TestClient(app)


def test_adversarial_sqlite_wal_high_concurrency(tmp_path):
    """Stress test: 30 concurrent writer threads and 15 concurrent reader threads on SQLite WAL mode."""
    db_path = str(tmp_path / "adv_wal_stress.db")
    monitor = ModelMonitor(db_path=db_path)
    errors = []

    def writer(idx):
        try:
            for seq in range(5):
                monitor.log_prediction(
                    input_text=f"Adversarial concurrent feedback sample {idx}-{seq}",
                    prediction=idx % 3,
                    confidence=0.88,
                    response_time=0.012
                )
        except Exception as e:
            errors.append((idx, "write", e))

    def reader(idx):
        try:
            for _ in range(3):
                _ = monitor.get_monitoring_dashboard_data(days=1)
        except Exception as e:
            errors.append((idx, "read", e))

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(30)]
    threads.extend([threading.Thread(target=reader, args=(i,)) for i in range(15)])

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Encountered SQLite lock contention errors: {errors}"

    conn = sqlite3.connect(db_path, timeout=30.0)
    total_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    conn.close()
    assert total_count == 150  # 30 writers * 5 records


def test_adversarial_api_http_concurrency(tmp_path, monkeypatch, client):
    """Stress test: Concurrent HTTP POST /predict and GET /api/telemetry requests without thread exhaustion."""
    db_path = str(tmp_path / "adv_api_concurrency.db")
    monitor = ModelMonitor(db_path=db_path)
    monkeypatch.setattr(app_module, "monitor", monitor)

    errors = []

    def send_predict(idx):
        try:
            resp = client.post("/predict", json={"text": f"Concurrent review number {idx} is exceptional!"})
            if resp.status_code != 200:
                errors.append((idx, "predict_status", resp.status_code))
        except Exception as e:
            errors.append((idx, "predict_exc", e))

    def send_telemetry(idx):
        try:
            resp = client.get("/api/telemetry")
            if resp.status_code != 200:
                errors.append((idx, "telemetry_status", resp.status_code))
        except Exception as e:
            errors.append((idx, "telemetry_exc", e))

    threads = [threading.Thread(target=send_predict, args=(i,)) for i in range(20)]
    threads.extend([threading.Thread(target=send_telemetry, args=(i,)) for i in range(10)])

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Encountered HTTP concurrency errors: {errors}"

    conn = sqlite3.connect(db_path, timeout=30.0)
    total_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    conn.close()
    assert total_count == 20


@pytest.mark.parametrize("exc_type,msg", [
    (RuntimeError, "Fatal inference engine failure"),
    (ValueError, "Corrupted feature dimensions"),
    (MemoryError, "Out of inference memory"),
])
def test_adversarial_predict_error_propagation_and_db_integrity(tmp_path, monkeypatch, client, exc_type, msg):
    """Verify that varied inference exceptions return HTTP 500 and write zero records to SQLite."""
    db_path = str(tmp_path / "adv_err_integrity.db")
    monitor = ModelMonitor(db_path=db_path)
    monkeypatch.setattr(app_module, "monitor", monitor)

    def mock_broken(text):
        raise exc_type(msg)

    monkeypatch.setattr(app_module.api, "predict_sentiment", mock_broken)

    resp = client.post("/predict", json={"text": "Authentic review text"})
    assert resp.status_code == 500
    assert msg in resp.json()["detail"]

    conn = sqlite3.connect(db_path, timeout=30.0)
    count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    conn.close()
    assert count == 0


def test_adversarial_input_validation_boundary_matrix(client):
    """Verify boundary cases in input payload handling (empty, whitespace, malformed, huge, unicode)."""
    # 1. Empty string
    r1 = client.post("/predict", json={"text": ""})
    assert r1.status_code in [400, 422]

    # 2. Pure whitespace and special escapes
    r2 = client.post("/predict", json={"text": "   \t\r\n   "})
    assert r2.status_code == 400
    assert "Text must be a non-empty string" in r2.json()["detail"]

    # 3. Missing 'text' attribute
    r3 = client.post("/predict", json={"message": "Great product!"})
    assert r3.status_code == 422

    # 4. Wrong data type: integer
    r4 = client.post("/predict", json={"text": 98765})
    assert r4.status_code == 422

    # 5. Wrong data type: null / None
    r5 = client.post("/predict", json={"text": None})
    assert r5.status_code == 422

    # 6. HTML-only payload that cleans to empty
    r6 = client.post("/predict", json={"text": "<p><br><span></span></p>"})
    assert r6.status_code == 200
    assert r6.json()["sentiment"] == "Neutral"

    # 7. Multilingual, emoji and unicode symbols
    r7 = client.post("/predict", json={"text": "✨ Excellent service! Très rapide et fiable! 素晴らしい体験でした! 🚀🎉"})
    assert r7.status_code == 200
    assert r7.json()["sentiment"] in ["Positive", "Neutral", "Negative"]

    # 8. Extremely large payload (20,000 characters)
    huge_text = "Highly recommended product with durable build. " * 500
    r8 = client.post("/predict", json={"text": huge_text})
    assert r8.status_code == 200
    assert "response_time_ms" in r8.json()


def test_adversarial_cpu_routes_synchronous_event_loop_safety():
    """Verify CPU-bound routes are non-coroutine standard functions so threadpool execution prevents loop starvation."""
    routes_to_check = {
        "/predict": "predict",
        "/api/simulate-drift": "simulate_drift",
        "/api/check-drift": "check_live_drift"
    }

    found = {}
    for r in app.routes:
        path = getattr(r, "path", None)
        if path in routes_to_check:
            found[path] = r.endpoint
            assert not inspect.iscoroutinefunction(r.endpoint), (
                f"Route {path} must be a synchronous 'def' function, found coroutine."
            )

    assert set(found.keys()) == set(routes_to_check.keys())


def test_adversarial_drift_detection_edge_cases(tmp_path):
    """Stress test drift detection against identical, shifted, constant, and sparse distributions."""
    db_path = str(tmp_path / "adv_drift.db")
    monitor = ModelMonitor(db_path=db_path)

    # 1. Identical distributions -> 0% drift, KS=0, p=1.0
    ref = np.random.normal(loc=0.0, scale=1.0, size=(80, 50))
    curr_same = ref.copy()
    res_same = monitor.detect_data_drift(ref, curr_same)
    assert res_same["overall_drift"] is False
    assert res_same["drift_percentage"] == 0.0

    # 2. Shifted distributions -> drift detected
    curr_shifted = np.random.normal(loc=10.0, scale=1.0, size=(80, 50))
    res_shifted = monitor.detect_data_drift(ref, curr_shifted)
    assert res_shifted["overall_drift"] is True
    assert res_shifted["drift_percentage"] > 0.5

    # 3. Constant value features (all zeros) handled without division by zero
    ref_zeros = np.zeros((50, 50))
    curr_zeros = np.zeros((50, 50))
    res_zeros = monitor.detect_data_drift(ref_zeros, curr_zeros)
    assert res_zeros["overall_drift"] is False
    assert res_zeros["drift_percentage"] == 0.0

    # 4. Sparse CSR matrix features
    ref_sp = sp.csr_matrix(ref)
    curr_sp = sp.csr_matrix(curr_same)
    res_sp = monitor.detect_data_drift(ref_sp, curr_sp)
    assert res_sp["overall_drift"] is False


def test_adversarial_prediction_drift_and_retraining_strategy(tmp_path):
    """Verify prediction drift and retraining strategy workflows."""
    db_path = str(tmp_path / "adv_retrain.db")
    monitor = ModelMonitor(db_path=db_path)

    # Varied class distribution
    ref_preds = np.array([0] * 30 + [1] * 40 + [2] * 50)
    curr_preds = np.array([0] * 35 + [1] * 35 + [2] * 50)
    p_res = monitor.detect_prediction_drift(ref_preds, curr_preds)
    assert "chi2_statistic" in p_res
    assert "p_value" in p_res
    assert isinstance(p_res["drift_detected"], (bool, np.bool_))

    # Retraining strategy
    strat = RetrainingStrategy({})
    plan = strat.create_retraining_plan(new_data_size=3000, current_model_performance={"f1": 0.85})
    assert plan["data_requirements"]["minimum_samples"] == 3000
    assert plan["validation_strategy"]["cross_validation_folds"] == 5


def test_adversarial_packaging_and_deployment_files():
    """Verify deployment configurations, requirements, and manifest integrity."""
    # 1. requirements-dev.txt contains onnx>=1.14.0
    req_dev = (PROJECT_ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    assert "onnx>=1.14.0" in req_dev

    # 2. pyproject.toml contains all required ML dependencies
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for dep in ["onnx", "tokenizers", "onnxruntime", "transformers", "torch"]:
        assert dep in pyproject, f"Missing {dep} in pyproject.toml"

    # 3. .env.example contains KAGGLE_API_TOKEN placeholder
    env_example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "KAGGLE_API_TOKEN" in env_example

    # 4. Dockerfile configuration
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "EXPOSE 5000" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "uvicorn src.api.app:app" in dockerfile
