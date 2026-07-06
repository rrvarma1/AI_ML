"""Runtime configuration resolved from environment variables."""

# Summary:
# 1. Purpose: Loads API paths, secrets, batch limits, and data-quality policies.
# 2. What it does: Builds one immutable Settings object from environment variables.
# 3. Invoked by: app/main.py and app/model_registry.py during application startup.
# 4. Main functions/classes: Settings and Settings.from_environment.
# 5. Validations/controls: Requires a 32-byte hash key and validates batch and policy values.

# Drift monitoring change:
# 1. Purpose: Enables encrypted Oracle telemetry and controls whether monitoring failures stop predictions.
# 2. Functions/validations: Loads MONITORING_ENABLED and validates fail-closed or warning behavior.

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path
    claim_model_path: Path
    risk_model_path: Path
    prediction_hash_key: bytes = field(repr=False)
    max_batch_size: int = 100
    missing_value_policy: str = "warn"
    unseen_category_policy: str = "warn"
    monitoring_enabled: bool = False
    monitoring_failure_policy: str = "fail_closed"

    @classmethod
    def from_environment(cls) -> "Settings":
        default_root = Path(__file__).resolve().parents[1]
        project_root = Path(os.getenv("CAPSTONE_PROJECT_ROOT", default_root)).expanduser().resolve()
        claim_model_path = Path(
            os.getenv(
                "CLAIM_MODEL_PATH",
                project_root / "models" / "claim" / "claim_random_forest_hypertuned_v3.pkl",
            )
        ).expanduser().resolve()
        risk_model_path = Path(
            os.getenv(
                "RISK_MODEL_PATH",
                project_root / "models" / "risk" / "risk_random_forest_smote_undersampling_v5.pkl",
            )
        ).expanduser().resolve()
        prediction_hash_key_value = os.getenv("PREDICTION_HASH_KEY")
        if not prediction_hash_key_value:
            raise ValueError("PREDICTION_HASH_KEY is required")
        prediction_hash_key = prediction_hash_key_value.encode("utf-8")
        if len(prediction_hash_key) < 32:
            raise ValueError("PREDICTION_HASH_KEY must contain at least 32 bytes")
        max_batch_size = int(os.getenv("MAX_PREDICTION_BATCH_SIZE", "100"))
        if max_batch_size < 1 or max_batch_size > 10_000:
            raise ValueError("MAX_PREDICTION_BATCH_SIZE must be between 1 and 10000")
        missing_value_policy = os.getenv("MISSING_VALUE_POLICY", "warn").lower()
        unseen_category_policy = os.getenv("UNSEEN_CATEGORY_POLICY", "warn").lower()
        if missing_value_policy not in {"warn", "reject"}:
            raise ValueError("MISSING_VALUE_POLICY must be 'warn' or 'reject'")
        if unseen_category_policy not in {"warn", "reject"}:
            raise ValueError("UNSEEN_CATEGORY_POLICY must be 'warn' or 'reject'")
        monitoring_enabled = _parse_boolean(os.getenv("MONITORING_ENABLED", "false"))
        monitoring_failure_policy = os.getenv(
            "MONITORING_FAILURE_POLICY", "fail_closed"
        ).lower()
        if monitoring_failure_policy not in {"fail_closed", "warn"}:
            raise ValueError("MONITORING_FAILURE_POLICY must be 'fail_closed' or 'warn'")
        return cls(
            project_root=project_root,
            claim_model_path=claim_model_path,
            risk_model_path=risk_model_path,
            prediction_hash_key=prediction_hash_key,
            max_batch_size=max_batch_size,
            missing_value_policy=missing_value_policy,
            unseen_category_policy=unseen_category_policy,
            monitoring_enabled=monitoring_enabled,
            monitoring_failure_policy=monitoring_failure_policy,
        )


def _parse_boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("MONITORING_ENABLED must be true or false")


settings = Settings.from_environment()
