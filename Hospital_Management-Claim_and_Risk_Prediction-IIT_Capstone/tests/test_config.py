"""Tests for security-sensitive runtime configuration."""

# Summary:
# 1. Purpose: Verifies security-sensitive API configuration rules.
# 2. What it does: Tests required secrets, minimum lengths, and data-quality policy loading.
# 3. Invoked by: Pytest during automated configuration verification.
# 4. Main functions/classes: Test functions for Settings.from_environment.
# 5. Validations/controls: Confirms secrets stay out of repr and invalid policies fail fast.

# Drift monitoring change:
# 1. Purpose: Verifies that Oracle monitoring enablement and failure behavior are configured safely.
# 2. Functions/validations: Tests accepted booleans and rejects unsupported monitoring failure policies.

from __future__ import annotations

import pytest

from app.config import Settings


def test_prediction_hash_key_is_required(monkeypatch) -> None:
    monkeypatch.delenv("PREDICTION_HASH_KEY", raising=False)
    with pytest.raises(ValueError, match="PREDICTION_HASH_KEY is required"):
        Settings.from_environment()


def test_prediction_hash_key_has_minimum_length(monkeypatch) -> None:
    monkeypatch.setenv("PREDICTION_HASH_KEY", "too-short")
    with pytest.raises(ValueError, match="at least 32 bytes"):
        Settings.from_environment()


def test_prediction_hash_key_is_excluded_from_settings_repr(monkeypatch) -> None:
    secret = "a-valid-secret-that-is-longer-than-32-bytes"
    monkeypatch.setenv("PREDICTION_HASH_KEY", secret)
    assert secret not in repr(Settings.from_environment())


def test_data_quality_policies_are_validated(monkeypatch) -> None:
    monkeypatch.setenv("MISSING_VALUE_POLICY", "invalid")
    with pytest.raises(ValueError, match="MISSING_VALUE_POLICY"):
        Settings.from_environment()


def test_data_quality_policies_are_loaded(monkeypatch) -> None:
    monkeypatch.setenv("MISSING_VALUE_POLICY", "reject")
    monkeypatch.setenv("UNSEEN_CATEGORY_POLICY", "reject")
    settings = Settings.from_environment()
    assert settings.missing_value_policy == "reject"
    assert settings.unseen_category_policy == "reject"


def test_monitoring_settings_are_loaded(monkeypatch) -> None:
    monkeypatch.setenv("MONITORING_ENABLED", "true")
    monkeypatch.setenv("MONITORING_FAILURE_POLICY", "warn")
    settings = Settings.from_environment()
    assert settings.monitoring_enabled is True
    assert settings.monitoring_failure_policy == "warn"


def test_invalid_monitoring_settings_fail_fast(monkeypatch) -> None:
    monkeypatch.setenv("MONITORING_ENABLED", "sometimes")
    with pytest.raises(ValueError, match="MONITORING_ENABLED"):
        Settings.from_environment()
