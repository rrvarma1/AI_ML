"""Integration tests for health, inference, and error contracts."""

# Summary:
# 1. Purpose: Verifies health, prediction, batch, and API validation behavior.
# 2. What it does: Sends requests to the FastAPI TestClient and checks response contracts.
# 3. Invoked by: Pytest during automated API verification.
# 4. Main functions/classes: assert_prediction and the health, prediction, and error tests.
# 5. Validations/controls: Checks probabilities, request IDs, artifact identities, and rejected inputs.

from __future__ import annotations

import pytest


def assert_prediction(response, model_id: str, classes: set[str]) -> None:
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model_id"] == model_id
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert body["result"]["prediction"] in classes
    probabilities = body["result"]["probabilities"]
    assert set(probabilities) == classes
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_liveness(client) -> None:
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_readiness_and_model_health(client) -> None:
    response = client.get("/health/ready")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ready"
    assert set(body["models"]) == {"claim-v3", "risk-v5"}
    assert all(model["ready"] for model in body["models"].values())
    assert all(len(model["artifact_sha256"]) == 64 for model in body["models"].values())
    assert all(len(model["category_contract_sha256"]) == 64 for model in body["models"].values())
    assert all(model["categorical_feature_count"] == 7 for model in body["models"].values())


def test_claim_prediction(client, claim_payload, monkeypatch) -> None:
    audit_events = []
    monkeypatch.setattr("app.main.log_prediction", lambda **event: audit_events.append(event))
    response = client.post(
        "/v1/predictions/claim-outcome",
        json=claim_payload,
        headers={"X-Request-ID": "claim-test"},
    )
    assert_prediction(response, "claim-v3", {"Paid", "Pending", "Rejected"})
    assert response.headers["X-Request-ID"] == "claim-test"
    assert len(audit_events) == 1
    assert audit_events[0]["request_id"] == "claim-test"
    assert audit_events[0]["model_id"] == "claim-v3"
    assert len(audit_events[0]["artifact_sha256"]) == 64
    assert len(audit_events[0]["feature_hash"]) == 64
    assert not set(claim_payload).intersection(audit_events[0])


def test_risk_prediction(client, risk_payload) -> None:
    response = client.post("/v1/predictions/visit-risk", json=risk_payload)
    assert_prediction(response, "risk-v5", {"High", "Low", "Medium"})


def test_batch_prediction(client, claim_payload, monkeypatch) -> None:
    audit_events = []
    monkeypatch.setattr("app.main.log_prediction", lambda **event: audit_events.append(event))
    response = client.post(
        "/v1/predictions/claim-outcome/batch",
        json={"instances": [claim_payload, claim_payload]},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 2
    assert len(body["results"]) == 2
    assert [event["record_index"] for event in audit_events] == [0, 1]
    assert all(event["batch_size"] == 2 for event in audit_events)


def test_unknown_field_is_rejected(client, claim_payload) -> None:
    claim_payload["unexpected"] = "value"
    response = client.post("/v1/predictions/claim-outcome", json=claim_payload)
    assert response.status_code == 422
    body = response.json()["error"]
    assert body["code"] == "validation_error"
    assert body["request_id"] == response.headers["X-Request-ID"]


def test_missing_field_is_rejected(client, claim_payload) -> None:
    del claim_payload["doctor_id"]
    response = client.post("/v1/predictions/claim-outcome", json=claim_payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_inconsistent_quarter_is_rejected(client, claim_payload) -> None:
    claim_payload["visit_quarter"] = 1
    response = client.post("/v1/predictions/claim-outcome", json=claim_payload)
    assert response.status_code == 422
    details = response.json()["error"]["details"]
    assert "visit_quarter must be 2" in details[0]["msg"]


def test_approved_amount_above_bill_is_rejected(client, risk_payload) -> None:
    risk_payload["approved_amount"] = risk_payload["billed_amount"] + 1
    response = client.post("/v1/predictions/visit-risk", json=risk_payload)
    assert response.status_code == 422
    details = response.json()["error"]["details"]
    assert "approved_amount cannot exceed billed_amount" in details[0]["msg"]


def test_new_operational_category_is_accepted_with_default_warning(client, claim_payload) -> None:
    claim_payload["city"] = "New Hospital City"
    response = client.post("/v1/predictions/claim-outcome", json=claim_payload)
    assert response.status_code == 200, response.text
