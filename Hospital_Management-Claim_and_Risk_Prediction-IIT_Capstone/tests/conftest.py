"""Shared API fixtures using the real persisted model artifacts."""

# Summary:
# 1. Purpose: Provides shared configuration and test data for the API test suite.
# 2. What it does: Sets safe test settings and creates the API client and valid payload fixtures.
# 3. Invoked by: Pytest before tests in the tests directory run.
# 4. Main functions/classes: client, claim_payload, and risk_payload fixtures.
# 5. Validations/controls: Uses a test-only hash key and the real project model artifacts.

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


CAPSTONE_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CAPSTONE_PROJECT_ROOT", str(CAPSTONE_ROOT))
os.environ.setdefault("PREDICTION_HASH_KEY", "test-only-prediction-hash-key-32-bytes-minimum")

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def claim_payload() -> dict:
    return {
        "age": 53,
        "chronic_flag": 0,
        "length_of_stay_hours": 3.48,
        "billed_amount": 23577.37,
        "days_from_visit_to_billing": 10,
        "days_since_registration_at_visit": 200,
        "billing_month": 6,
        "billing_quarter": 2,
        "billing_day_of_week": 2,
        "billing_is_weekend": 0,
        "visit_month": 6,
        "visit_quarter": 2,
        "visit_day_of_week": 1,
        "visit_is_weekend": 0,
        "gender": "M",
        "city": "Hyderabad",
        "insurance_provider": "SecureLife",
        "department": "Cardiology",
        "visit_type": "ER",
        "risk_score": "Low",
        "doctor_id": 169,
    }


@pytest.fixture
def risk_payload() -> dict:
    return {
        "age": 53,
        "chronic_flag": 0,
        "length_of_stay_hours": 3.48,
        "billed_amount": 23577.37,
        "approved_amount": 0.0,
        "payment_days": 16.0,
        "days_since_registration_at_visit": 200,
        "visit_year": 2025,
        "visit_month": 6,
        "visit_quarter": 2,
        "visit_day_of_week": 1,
        "visit_is_weekend": 0,
        "gender": "M",
        "city": "Hyderabad",
        "insurance_provider": "SecureLife",
        "department": "Cardiology",
        "visit_type": "ER",
        "claim_status": "Rejected",
        "doctor_id": 169,
    }
