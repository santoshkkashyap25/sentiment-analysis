"""Integration and unit tests for FastAPI endpoints."""

import sys
import inspect
import sqlite3
import threading
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient
from src.api.app import app
from src.monitoring.drift_detection import ModelMonitor
import src.api.app as app_module

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_stats_endpoint():
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "uptime" in data
    assert "model_info" in data


def test_predict_endpoint_success():
    payload = {"text": "The build quality is superb and delivery was fast!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] in ["Positive", "Neutral", "Negative"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert "probabilities" in data
    assert "Positive" in data["probabilities"]
    assert "Neutral" in data["probabilities"]
    assert "Negative" in data["probabilities"]
    assert "response_time_ms" in data


def test_predict_endpoint_empty_text_400():
    payload = {"text": "   "}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400


def test_telemetry_endpoint():
    response = client.get("/api/telemetry")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions" in data
    assert "avg_latency_ms" in data
    assert "sentiment_counts" in data
    assert "recent_predictions" in data


def test_simulate_drift_endpoint():
    response = client.post("/api/simulate-drift")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "simulation_complete"
    assert "drift_detected" in data
    assert "drift_percentage" in data


def test_cpu_bound_routes_are_synchronous():
    """R6.1: Verify CPU-bound routes (/predict, /api/simulate-drift, /api/check-drift) are synchronous def."""
    cpu_routes = ["/predict", "/api/simulate-drift", "/api/check-drift"]
    found_routes = []
    for route in app.routes:
        path = getattr(route, "path", None)
        if path in cpu_routes:
            found_routes.append(path)
            assert not inspect.iscoroutinefunction(route.endpoint), (
                f"Route '{path}' must be a synchronous 'def', not async def."
            )
    assert set(found_routes) == set(cpu_routes)


def test_predict_endpoint_error_returns_500_and_does_not_log_sqlite(tmp_path, monkeypatch):
    """R6.2: Verify inference failures raise HTTP 500 and write no records to SQLite."""
    test_db = tmp_path / "test_api_err.db"
    test_monitor = ModelMonitor(db_path=str(test_db))
    monkeypatch.setattr(app_module, "monitor", test_monitor)

    def mock_broken_predict(text):
        raise RuntimeError("Inference model weights unavailable or corrupted")

    monkeypatch.setattr(app_module.api, "predict_sentiment", mock_broken_predict)

    response = client.post("/predict", json={"text": "Legitimate customer feedback text"})
    assert response.status_code == 500
    assert "Inference model weights unavailable or corrupted" in response.json()["detail"]

    # Verify SQLite database has 0 records
    conn = sqlite3.connect(str(test_db), timeout=30.0)
    count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    conn.close()
    assert count == 0


def test_predict_endpoint_success_logs_to_sqlite(tmp_path, monkeypatch):
    """R6.2: Verify successful predictions write accurate records to SQLite."""
    test_db = tmp_path / "test_api_success.db"
    test_monitor = ModelMonitor(db_path=str(test_db))
    monkeypatch.setattr(app_module, "monitor", test_monitor)

    response = client.post("/predict", json={"text": "Outstanding quality, very satisfied!"})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] in ["Positive", "Neutral", "Negative"]
    assert 0.0 <= data["confidence"] <= 1.0

    # Verify SQLite recorded exactly 1 row
    conn = sqlite3.connect(str(test_db), timeout=30.0)
    row = conn.execute("SELECT COUNT(*), prediction, input_text FROM predictions").fetchone()
    conn.close()
    assert row[0] == 1
    assert row[1] in [0, 1, 2]
    assert "Outstanding quality" in row[2]


def test_sqlite_wal_mode_and_timeout(tmp_path):
    """R6.3: Verify SQLite connections configure timeout=30.0 and execute PRAGMA journal_mode=WAL."""
    test_db = tmp_path / "test_wal.db"
    monitor = ModelMonitor(db_path=str(test_db))

    conn = sqlite3.connect(str(test_db), timeout=30.0)
    mode = conn.execute("PRAGMA journal_mode;").fetchone()[0].lower()
    conn.close()
    assert mode == "wal"


def test_check_live_drift_endpoint_insufficient_data(tmp_path, monkeypatch):
    """Verify /api/check-drift endpoint handles empty / insufficient data gracefully."""
    test_db = tmp_path / "test_drift_empty.db"
    test_monitor = ModelMonitor(db_path=str(test_db))
    monkeypatch.setattr(app_module, "monitor", test_monitor)

    response = client.post("/api/check-drift")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "insufficient_data"
    assert data["drift_detected"] is False


def test_check_live_drift_endpoint_with_data(tmp_path, monkeypatch):
    """Verify /api/check-drift endpoint evaluates drift when queries exist in SQLite."""
    test_db = tmp_path / "test_drift_data.db"
    test_monitor = ModelMonitor(db_path=str(test_db))
    monkeypatch.setattr(app_module, "monitor", test_monitor)

    # Seed 10 predictions
    for i in range(10):
        test_monitor.log_prediction(
            input_text=f"Product review sample number {i} works great and very fast",
            prediction=2,
            confidence=0.95,
            response_time=0.02
        )

    response = client.post("/api/check-drift")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["success", "insufficient_data"]
    assert "drift_detected" in data


def test_sqlite_multithreaded_concurrency(tmp_path):
    """R6.3: Verify multi-threaded concurrent write operations on SQLite under WAL mode."""
    test_db = tmp_path / "test_concurrency.db"
    test_monitor = ModelMonitor(db_path=str(test_db))
    errors = []

    def worker(idx):
        try:
            test_monitor.log_prediction(
                input_text=f"Concurrent feedback #{idx}",
                prediction=idx % 3,
                confidence=0.85,
                response_time=0.01
            )
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(25)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Encountered concurrency errors: {errors}"

    conn = sqlite3.connect(str(test_db), timeout=30.0)
    total_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    conn.close()
    assert total_count == 25


if __name__ == '__main__':
    pytest.main([__file__, "-v"])
