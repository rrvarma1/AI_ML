"""Tests for chronological training baseline reconstruction."""

# Summary:
# 1. Purpose: Verifies that baseline data uses the notebook-compatible earliest 80% split.
# 2. Functions/validations: Tests table joins, derived date features, source metadata, and feature contracts.

from __future__ import annotations

import pandas as pd

from app.data_validation import EXPECTED_COLUMNS
from monitoring.training_baseline import build_training_frame_from_tables


def source_tables():
    patients = pd.DataFrame(
        {
            "patient_id": range(1, 11),
            "age": range(31, 41),
            "gender": ["F", "M"] * 5,
            "city": ["Hyderabad"] * 10,
            "insurance_provider": ["SecureLife"] * 10,
            "chronic_flag": [0, 1] * 5,
            "registration_date": pd.date_range("2024-01-01", periods=10),
        }
    )
    visits = pd.DataFrame(
        {
            "visit_id": range(1, 11),
            "patient_id": range(1, 11),
            "visit_date": pd.date_range("2025-01-01", periods=10),
            "department": ["Cardiology"] * 10,
            "visit_type": ["ER"] * 10,
            "length_of_stay_hours": [4.0] * 10,
            "risk_score": ["Low", "Medium"] * 5,
            "doctor_id": range(100, 110),
        }
    )
    billing = pd.DataFrame(
        {
            "bill_id": range(1, 11),
            "visit_id": range(1, 11),
            "billed_amount": range(1000, 1010),
            "approved_amount": range(900, 910),
            "claim_status": ["Paid", "Rejected"] * 5,
            "payment_days": [5.0] * 10,
            "billing_date": pd.date_range("2025-01-02", periods=10),
        }
    )
    return patients, visits, billing


def test_claim_and_risk_training_frames_follow_80_percent_split() -> None:
    patients, visits, billing = source_tables()
    source = {"type": "oracle", "tables": {}}
    for model_id in ("claim-v3", "risk-v5"):
        frame, metadata = build_training_frame_from_tables(
            patients,
            visits,
            billing,
            model_id=model_id,
            source=source,
        )
        assert len(frame) == 8
        assert frame.columns.tolist() == EXPECTED_COLUMNS[model_id]
        assert metadata["source"] is source
        assert metadata["period"]["start"].startswith("2025-01")


def test_derived_calendar_and_lag_features_are_correct() -> None:
    patients, visits, billing = source_tables()
    frame, _ = build_training_frame_from_tables(
        patients,
        visits,
        billing,
        model_id="claim-v3",
        source={"type": "oracle", "tables": {}},
    )
    assert frame.iloc[0]["days_from_visit_to_billing"] == 1
    assert frame.iloc[0]["visit_month"] == 1
    assert frame.iloc[0]["billing_quarter"] == 1
