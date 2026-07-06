"""Tests for encrypted Oracle monitoring payloads and configuration."""

# Summary:
# 1. Purpose: Verifies encrypted telemetry storage without requiring a live Oracle test database.
# 2. Functions/validations: Tests confidentiality, round trips, integrity checks, retention, and key rotation.

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from cryptography.fernet import Fernet

from app.oracle_monitoring_store import MonitoringStoreError, OracleMonitoringStore


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def setinputsizes(self, **_):
        pass

    def executemany(self, _sql, rows):
        self.connection.inserted.extend(dict(row) for row in rows)
        self.rowcount = len(rows)

    def execute(self, _sql, _binds=None):
        self.rowcount = self.connection.delete_count

    def fetchall(self):
        return self.connection.fetch_rows


class FakeConnection:
    def __init__(self):
        self.inserted = []
        self.fetch_rows = []
        self.delete_count = 0
        self.commits = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1


def test_encrypted_telemetry_round_trip_hides_features() -> None:
    connection = FakeConnection()
    key = Fernet.generate_key()
    store = OracleMonitoringStore(
        connection_factory=lambda: connection,
        encryption_key=key,
        retention_days=90,
        schema="MONITORING",
    )
    now = datetime.now(timezone.utc)
    stored = store.store_predictions(
        model_id="claim-v3",
        artifact_sha256="a" * 64,
        request_id="private-request-id",
        recorded_at=now,
        records=[{"age": 53, "city": "Hyderabad"}],
        results=[{"prediction": "Rejected", "probabilities": {"Rejected": 0.7, "Paid": 0.3}}],
        input_hashes=["b" * 64],
    )
    assert stored == 1
    row = connection.inserted[0]
    assert b"Hyderabad" not in row["encrypted_payload"]
    assert b"private-request-id" not in row["encrypted_payload"]
    connection.fetch_rows = [
        (
            row["telemetry_id"],
            row["recorded_at"],
            row["key_id"],
            row["encrypted_payload"],
            row["payload_sha256"],
        )
    ]
    records = store.fetch_predictions(
        model_id="claim-v3",
        artifact_sha256="a" * 64,
        start_at=now,
        end_at=now.replace(year=now.year + 1),
    )
    assert records[0]["features"]["city"] == "Hyderabad"
    assert records[0]["prediction"] == "Rejected"


def test_ciphertext_integrity_and_key_rotation() -> None:
    old_key = Fernet.generate_key()
    new_key = Fernet.generate_key()
    old_store = OracleMonitoringStore(
        connection_factory=lambda: FakeConnection(), encryption_key=old_key
    )
    ciphertext = old_store._encrypt({"features": {"age": 53}})
    old_key_id = old_store._primary_key_id
    rotated_store = OracleMonitoringStore(
        connection_factory=lambda: FakeConnection(),
        encryption_key=new_key,
        decryption_keys=(old_key,),
    )
    assert rotated_store._decrypt(old_key_id, ciphertext)["features"]["age"] == 53
    with pytest.raises(MonitoringStoreError, match="authentication failed"):
        rotated_store._decrypt(old_key_id, ciphertext[:-1] + b"x")


def test_schema_and_retention_are_validated() -> None:
    key = Fernet.generate_key()
    with pytest.raises(ValueError, match="schema"):
        OracleMonitoringStore(
            connection_factory=lambda: FakeConnection(),
            encryption_key=key,
            schema="bad.schema",
        )
    with pytest.raises(ValueError, match="retention_days"):
        OracleMonitoringStore(
            connection_factory=lambda: FakeConnection(),
            encryption_key=key,
            retention_days=0,
        )


def test_api_sends_validated_prediction_to_monitoring_store(
    client, claim_payload, monkeypatch
) -> None:
    import app.main as main_module

    calls = []

    class CaptureStore:
        def store_predictions(self, **kwargs):
            calls.append(kwargs)
            return len(kwargs["records"])

    monkeypatch.setattr(main_module, "monitoring_store", CaptureStore())
    response = client.post(
        "/v1/predictions/claim-outcome",
        json=claim_payload,
        headers={"X-Request-ID": "telemetry-api-test"},
    )
    assert response.status_code == 200, response.text
    assert calls[0]["request_id"] == "telemetry-api-test"
    assert calls[0]["model_id"] == "claim-v3"
    assert calls[0]["records"][0] == claim_payload
    assert calls[0]["results"][0]["prediction"] in {"Paid", "Pending", "Rejected"}


def test_api_fails_closed_when_required_telemetry_write_fails(
    client, claim_payload, monkeypatch
) -> None:
    import app.main as main_module

    class FailingStore:
        def store_predictions(self, **_):
            raise MonitoringStoreError("Oracle unavailable")

    monkeypatch.setattr(main_module, "monitoring_store", FailingStore())
    response = client.post("/v1/predictions/claim-outcome", json=claim_payload)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "monitoring_store_unavailable"
