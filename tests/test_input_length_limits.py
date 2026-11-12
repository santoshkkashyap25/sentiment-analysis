"""
Tests for input length limits, Head+Tail smart truncation, and payload validation.
"""
import pytest
from fastapi.testclient import TestClient
from src.api.app import app
from sentipulse import SentiPulse

client = TestClient(app)

def test_api_accepts_normal_text_not_truncated():
    """Verify standard short text is not truncated and returns word_count and char_count."""
    payload = {"text": "Excellent build quality and very fast shipping!"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["sentiment"] == "Positive"
    assert data["truncated"] is False
    assert data["word_count"] == 7
    assert data["char_count"] == len(payload["text"])

def test_api_head_tail_truncation_on_long_text():
    """Verify long text exceeding 128 tokens triggers Head+Tail truncation flag and captures concluding verdict."""
    long_review = (
        "I ordered this product with high hopes. "
        + ("The delivery arrived on schedule and the exterior packaging looked fine. " * 20)
        + "However, it caught fire on the first day! Horrible customer service, refused refund, absolute disaster."
    )
    resp = client.post("/predict", json={"text": long_review})
    assert resp.status_code == 200
    data = resp.json()
    assert data["truncated"] is True
    assert data["word_count"] > 120
    # Head+Tail captures the concluding negative verdict that head-only would have missed
    assert data["sentiment"] == "Negative"
    assert data["confidence"] >= 0.85

def test_api_hard_limit_rejection_over_25000_chars():
    """Verify payloads exceeding 25,000 characters are rejected with 422 Unprocessable Entity."""
    oversized = "A" * 25001
    resp = client.post("/predict", json={"text": oversized})
    assert resp.status_code == 422

def test_sdk_local_oversized_rejection():
    """Verify SDK raises ValueError if input text exceeds 25,000 characters."""
    sp = SentiPulse()
    with pytest.raises(ValueError, match="exceeds maximum allowed length"):
        sp.analyze("B" * 25001)

def test_sdk_analyze_includes_length_metadata():
    """Verify SDK returns word_count, char_count and truncated metadata."""
    sp = SentiPulse(api_url="http://127.0.0.1:5000")
    res = sp.analyze("Great product, five stars!")
    assert res["sentiment"] == "Positive"
    assert "word_count" in res
    assert "char_count" in res
    assert res["truncated"] is False
