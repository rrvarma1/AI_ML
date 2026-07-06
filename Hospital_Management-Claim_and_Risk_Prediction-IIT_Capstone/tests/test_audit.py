"""Unit tests for privacy-preserving prediction audit events."""

# Summary:
# 1. Purpose: Verifies privacy-safe prediction hashing and JSON audit events.
# 2. What it does: Tests deterministic hashes and captures emitted prediction audit records.
# 3. Invoked by: Pytest during automated audit-control verification.
# 4. Main functions/classes: Feature-hash and audit-log test functions.
# 5. Validations/controls: Confirms raw features are absent and timestamps and hashes are present.

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from app.audit import AUDIT_LOGGER, input_feature_hash, log_prediction


SECRET = b"audit-test-secret-that-is-at-least-32-bytes"


def test_feature_hash_is_canonical_and_sensitive_to_changes() -> None:
    first = {"age": 53, "city": "Hyderabad", "chronic_flag": 0}
    reordered = {"chronic_flag": 0, "city": "Hyderabad", "age": 53}
    changed = {"age": 54, "city": "Hyderabad", "chronic_flag": 0}

    assert input_feature_hash(first, SECRET) == input_feature_hash(reordered, SECRET)
    assert input_feature_hash(first, SECRET) != input_feature_hash(changed, SECRET)
    assert input_feature_hash(first, SECRET) != input_feature_hash(first, SECRET + b"-other")


def test_audit_log_is_json_and_contains_no_raw_features() -> None:
    records: list[logging.LogRecord] = []

    class CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = CaptureHandler()
    AUDIT_LOGGER.addHandler(handler)
    try:
        log_prediction(
            timestamp=datetime(2026, 7, 2, 10, 15, tzinfo=timezone.utc),
            request_id="audit-test",
            model_id="claim-v3",
            model_name="Claim model",
            artifact_sha256="a" * 64,
            feature_hash=input_feature_hash({"age": 53}, SECRET),
            prediction="Rejected",
            probabilities={"Paid": 0.1, "Pending": 0.2, "Rejected": 0.7},
            duration_ms=12.345,
            record_index=0,
            batch_size=1,
        )
    finally:
        AUDIT_LOGGER.removeHandler(handler)

    event = json.loads(records[0].getMessage())
    assert event["timestamp"] == "2026-07-02T10:15:00Z"
    assert event["model_id"] == "claim-v3"
    assert event["artifact_sha256"] == "a" * 64
    assert len(event["input_feature_hash"]) == 64
    assert event["duration_ms"] == 12.35
    assert "age" not in event
