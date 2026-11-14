"""
Tests for Multi-Tenant Isolation, API Key Authentication, and Partitioned Telemetry.
"""
import pytest
from fastapi.testclient import TestClient
from src.api.app import app
import src.api.app as app_module
from src.monitoring.drift_detection import ModelMonitor
from sentipulse import SentiPulse

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_tenant_db(tmp_path, monkeypatch):
    """Ensure every multi-tenancy test runs with an isolated database."""
    test_db = tmp_path / "test_multi_tenancy.db"
    test_mon = ModelMonitor(db_path=str(test_db))
    monkeypatch.setattr(app, "monitor", test_mon)
    monkeypatch.setattr(app_module, "monitor", test_mon)
    yield test_mon


def test_auth_create_and_list_api_keys():
    """Verify generating and listing API keys for a tenant."""
    # 1. Create key for tenant 'acme'
    res = client.post("/api/auth/keys", json={"tenant_id": "acme", "name": "Acme Production"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "created"
    assert data["tenant_id"] == "acme"
    api_key = data["api_key"]
    assert api_key.startswith("sp_live_acme_")

    # 2. List keys
    list_res = client.get("/api/auth/keys")
    assert list_res.status_code == 200
    keys = list_res.json()["keys"]
    assert any(k["tenant_id"] == "acme" for k in keys)


def test_auth_invalid_api_key_rejected():
    """Verify invalid or forged API keys are rejected with HTTP 401."""
    res = client.post(
        "/predict",
        json={"text": "Test review text"},
        headers={"X-API-Key": "sp_live_invalid_forged_key_12345"}
    )
    assert res.status_code == 401
    assert "Invalid API Key" in res.json()["detail"]


def test_backward_compatibility_no_key_defaults():
    """Verify requests without an API key default to 'default' tenant with HTTP 200."""
    res = client.post("/predict", json={"text": "Standard feedback without auth header"})
    assert res.status_code == 200
    data = res.json()
    assert data["tenant_id"] == "default"


def test_tenant_data_and_telemetry_isolation():
    """Verify that predictions from Tenant A are completely isolated from Tenant B."""
    # Create keys for two distinct tenants
    key_a = client.post("/api/auth/keys", json={"tenant_id": "tenant_alpha"}).json()["api_key"]
    key_b = client.post("/api/auth/keys", json={"tenant_id": "tenant_beta"}).json()["api_key"]

    # Submit 3 Positive reviews for Tenant Alpha
    for i in range(3):
        res_a = client.post(
            "/predict",
            json={"text": f"Alpha positive review number {i} - superb experience!"},
            headers={"X-API-Key": key_a}
        )
        assert res_a.status_code == 200
        assert res_a.json()["tenant_id"] == "tenant_alpha"

    # Submit 2 Negative reviews for Tenant Beta
    for j in range(2):
        res_b = client.post(
            "/predict",
            json={"text": f"Beta negative review number {j} - broken device, refund denied."},
            headers={"X-API-Key": key_b}
        )
        assert res_b.status_code == 200
        assert res_b.json()["tenant_id"] == "tenant_beta"

    # Query telemetry for Tenant Alpha
    telem_a = client.get("/api/telemetry", headers={"X-API-Key": key_a}).json()
    assert telem_a["tenant_id"] == "tenant_alpha"
    # All recent predictions for Alpha must belong to Alpha
    assert len(telem_a["recent_predictions"]) >= 3
    assert all(r["tenant_id"] == "tenant_alpha" for r in telem_a["recent_predictions"])
    assert all("Alpha" in r["text"] for r in telem_a["recent_predictions"])

    # Query telemetry for Tenant Beta
    telem_b = client.get("/api/telemetry", headers={"X-API-Key": key_b}).json()
    assert telem_b["tenant_id"] == "tenant_beta"
    assert len(telem_b["recent_predictions"]) >= 2
    assert all(r["tenant_id"] == "tenant_beta" for r in telem_b["recent_predictions"])
    assert all("Beta" in r["text"] for r in telem_b["recent_predictions"])


def test_sdk_multi_tenant_client():
    """Verify Python SDK properly attaches X-API-Key header to remote requests."""
    from unittest.mock import MagicMock, patch

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "sentiment": "Positive",
        "confidence": 0.95,
        "probabilities": {"Negative": 0.01, "Neutral": 0.04, "Positive": 0.95},
        "response_time_ms": 15.2,
        "tenant_id": "enterprise_corp"
    }
    mock_resp.raise_for_status.return_value = None

    with patch("requests.post", return_value=mock_resp) as mock_post:
        sp = SentiPulse(api_url="https://api.sentipulse.io", api_key="sp_live_enterprise_corp_12345")
        res = sp.analyze("Enterprise grade service, highly pleased!")
        
        mock_post.assert_called_once_with(
            "https://api.sentipulse.io/predict",
            json={"text": "Enterprise grade service, highly pleased!"},
            headers={"X-API-Key": "sp_live_enterprise_corp_12345"},
            timeout=15.0
        )
        assert res["sentiment"] == "Positive"
        assert res["tenant_id"] == "enterprise_corp"

