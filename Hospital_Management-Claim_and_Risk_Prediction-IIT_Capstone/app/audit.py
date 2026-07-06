"""Privacy-preserving structured audit events for successful predictions."""

# Summary:
# 1. Purpose: Creates privacy-safe audit logs for successful model predictions.
# 2. What it does: Hashes validated inputs and writes one structured JSON event per prediction.
# 3. Invoked by: app/main.py after Claim V3 or Risk V5 returns a prediction.
# 4. Main functions/classes: input_feature_hash, build_prediction_event, and log_prediction.
# 5. Validations/controls: Uses canonical JSON and HMAC-SHA256; raw input features are not logged.

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from datetime import datetime
from typing import Any


AUDIT_LOGGER = logging.getLogger("prediction_audit")
AUDIT_LOGGER.setLevel(logging.INFO)
AUDIT_LOGGER.propagate = False

if not any(getattr(handler, "is_prediction_audit_handler", False) for handler in AUDIT_LOGGER.handlers):
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.is_prediction_audit_handler = True  # type: ignore[attr-defined]
    AUDIT_LOGGER.addHandler(handler)


def input_feature_hash(record: dict[str, Any], secret: bytes) -> str:
    """Return an HMAC over a deterministic representation of validated features."""
    canonical_input = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hmac.new(secret, canonical_input, hashlib.sha256).hexdigest()


def build_prediction_event(
    *,
    timestamp: datetime,
    request_id: str,
    model_id: str,
    model_name: str,
    artifact_sha256: str,
    feature_hash: str,
    prediction: str,
    probabilities: dict[str, float],
    duration_ms: float,
    record_index: int,
    batch_size: int,
) -> dict[str, Any]:
    return {
        "event": "model_prediction",
        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
        "request_id": request_id,
        "model_id": model_id,
        "model_name": model_name,
        "artifact_sha256": artifact_sha256,
        "input_feature_hash": feature_hash,
        "prediction": prediction,
        "probabilities": probabilities,
        "record_index": record_index,
        "batch_size": batch_size,
        "duration_ms": round(duration_ms, 2),
    }


def log_prediction(**event_fields: Any) -> None:
    """Write one compact JSON event without raw feature values."""
    event = build_prediction_event(**event_fields)
    AUDIT_LOGGER.info(json.dumps(event, separators=(",", ":"), ensure_ascii=True))
