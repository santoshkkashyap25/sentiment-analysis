"""Unit tests for SentiPulse Python SDK Client."""

import pytest
from unittest.mock import MagicMock, patch
from sentipulse.client import SentiPulse


def test_sdk_empty_text():
    """Verify SentiPulse client returns default neutral dict on empty text."""
    sp = SentiPulse(api_url="http://mock-api.local")
    res = sp.analyze("")
    assert res["sentiment"] == "Neutral"
    assert res["confidence"] == 0.33
    assert res["escalation_recommended"] is False


def test_sdk_remote_client_mock():
    """Verify SentiPulse client correctly dispatches HTTP request to remote API."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "sentiment": "Negative",
        "confidence": 0.94,
        "probabilities": {"Negative": 0.94, "Neutral": 0.04, "Positive": 0.02},
        "response_time_ms": 28.5,
        "model_version": "3.0-ONNX-INT8"
    }
    mock_resp.raise_for_status.return_value = None

    with patch("requests.post", return_value=mock_resp) as mock_post:
        sp = SentiPulse(api_url="https://api.sentipulse.io")
        res = sp.analyze("Worst purchase ever, completely broke after 1 day.")

        mock_post.assert_called_once_with(
            "https://api.sentipulse.io/predict",
            json={"text": "Worst purchase ever, completely broke after 1 day."},
            timeout=15.0
        )
        assert res["sentiment"] == "Negative"
        assert res["confidence"] == 0.94
        assert res["escalation_recommended"] is True


def test_sdk_batch_inference():
    """Verify SentiPulse analyze_batch iterates over list of texts."""
    sp = SentiPulse(api_url="https://api.sentipulse.io")
    with patch.object(sp, "analyze") as mock_analyze:
        mock_analyze.side_effect = [
            {"sentiment": "Positive", "confidence": 0.9},
            {"sentiment": "Negative", "confidence": 0.95}
        ]
        results = sp.analyze_batch(["Great", "Terrible"])
        assert len(results) == 2
        assert results[0]["sentiment"] == "Positive"
        assert results[1]["sentiment"] == "Negative"


def test_sdk_health_and_drift_remote():
    """Verify health and check_drift methods query corresponding remote endpoints."""
    mock_health = MagicMock()
    mock_health.json.return_value = {"status": "healthy", "model_version": "3.0-ONNX-INT8"}
    mock_health.raise_for_status.return_value = None

    mock_drift = MagicMock()
    mock_drift.json.return_value = {"status": "success", "drift_detected": False}
    mock_drift.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_health), patch("requests.post", return_value=mock_drift):
        sp = SentiPulse(api_url="https://api.sentipulse.io")
        h = sp.health()
        assert h["status"] == "healthy"

        d = sp.check_drift()
        assert d["drift_detected"] is False

