"""Unit tests for SentiPulse Support Triage and Webhook Dispatcher."""

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.integrations.webhooks import WebhookDispatcher
from src.api.app import app

client = TestClient(app)


def test_webhook_dispatcher_escalation_trigger():
    """Verify WebhookDispatcher triggers HTTP POST when Negative confidence >= threshold."""
    dispatcher = WebhookDispatcher(webhook_url="https://hooks.example.com/alerts", min_negative_conf=0.85)

    with patch.object(dispatcher, "_send") as mock_send:
        # 1. Negative with 90% confidence -> SHOULD trigger
        triggered = dispatcher.dispatch_escalation(
            text="Customer service ignored my refund request",
            sentiment="Negative",
            confidence=0.90
        )
        assert triggered is True

        # 2. Negative with 70% confidence (< 85%) -> SHOULD NOT trigger
        triggered_low = dispatcher.dispatch_escalation(
            text="Mild disappointment",
            sentiment="Negative",
            confidence=0.70
        )
        assert triggered_low is False

        # 3. Positive with 99% confidence -> SHOULD NOT trigger
        triggered_pos = dispatcher.dispatch_escalation(
            text="Loved it!",
            sentiment="Positive",
            confidence=0.99
        )
        assert triggered_pos is False


def test_webhook_slack_payload_format():
    """Verify Slack webhook URLs format payloads with Slack attachments."""
    dispatcher = WebhookDispatcher(webhook_url="https://hooks.slack.com/services/T00/B00/X00")
    payload = dispatcher.format_payload(
        text="App keeps crashing on launch!",
        sentiment="Negative",
        confidence=0.95
    )
    assert "attachments" in payload
    assert payload["attachments"][0]["color"] == "#e11d48"
    assert "SentiPulse Customer Escalation Alert" in payload["text"]


def test_webhook_generic_payload_format():
    """Verify generic webhook URLs format payloads as clean JSON schemas."""
    dispatcher = WebhookDispatcher(webhook_url="https://api.internal.com/triage")
    payload = dispatcher.format_payload(
        text="Broken part received.",
        sentiment="Negative",
        confidence=0.92
    )
    assert payload["event"] == "customer_escalation"
    assert payload["source"] == "sentipulse-engine"
    assert payload["severity"] == "high"
    assert payload["confidence"] == 0.92


def test_api_webhook_config_endpoints():
    """Verify /api/webhook/config GET and POST endpoints."""
    # 1. GET config
    res_get = client.get("/api/webhook/config")
    assert res_get.status_code == 200
    data_get = res_get.json()
    assert "webhook_url" in data_get
    assert "min_negative_conf" in data_get

    # 2. POST update config
    res_post = client.post(
        "/api/webhook/config",
        json={"webhook_url": "https://hooks.slack.com/services/TEST", "min_negative_conf": 0.88, "enabled": True}
    )
    assert res_post.status_code == 200
    data_post = res_post.json()
    assert data_post["status"] == "updated"
    assert data_post["webhook_url"] == "https://hooks.slack.com/services/TEST"
    assert data_post["min_negative_conf"] == 0.88
    assert data_post["enabled"] is True

    # 3. Clean up by disabling webhook URL
    client.post("/api/webhook/config", json={"webhook_url": None, "min_negative_conf": 0.85, "enabled": False})
