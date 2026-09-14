"""Integration and unit tests for FastAPI endpoints."""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient
from src.api.app import app

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


if __name__ == '__main__':
    print("Running API tests directly...", flush=True)
    test_health_endpoint()
    print("test_health_endpoint: PASSED", flush=True)
    test_stats_endpoint()
    print("test_stats_endpoint: PASSED", flush=True)
    test_predict_endpoint_success()
    print("test_predict_endpoint_success: PASSED", flush=True)
    test_predict_endpoint_empty_text_400()
    print("test_predict_endpoint_empty_text_400: PASSED", flush=True)
    test_telemetry_endpoint()
    print("test_telemetry_endpoint: PASSED", flush=True)
    test_simulate_drift_endpoint()
    print("test_simulate_drift_endpoint: PASSED", flush=True)
    print("ALL API TESTS PASSED!", flush=True)
