"""Tests for feature and prediction baseline and drift calculations."""

# Summary:
# 1. Purpose: Verifies stable populations stay healthy and material shifts become critical.
# 2. Functions/validations: Tests numeric PSI, categorical divergence, unseen values, and prediction drift.

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from monitoring.drift_metrics import calculate_drift_report, create_training_baseline


def make_baseline():
    frame = pd.DataFrame(
        {
            "age": list(range(20, 70)) * 2,
            "city": ["Hyderabad", "Delhi"] * 50,
        }
    )
    predictions = ["Paid", "Rejected"] * 50
    probabilities = np.asarray([[0.7, 0.3], [0.3, 0.7]] * 50)
    baseline = create_training_baseline(
        model_id="claim-v3",
        artifact_sha256="a" * 64,
        category_contract_sha256="b" * 64,
        training_frame=frame,
        predictions=predictions,
        probabilities=probabilities,
        classes=["Paid", "Rejected"],
        training_period={"start": "2025-01-01", "end": "2025-06-30"},
        source_files={"training.csv": "c" * 64},
    )
    return baseline, frame, predictions, probabilities


def records_from(frame, predictions, probabilities):
    return [
        {
            "features": row,
            "prediction": prediction,
            "probabilities": {"Paid": float(probs[0]), "Rejected": float(probs[1])},
            "max_probability": float(max(probs)),
        }
        for row, prediction, probs in zip(frame.to_dict("records"), predictions, probabilities)
    ]


def test_matching_population_has_no_critical_drift() -> None:
    baseline, frame, predictions, probabilities = make_baseline()
    now = datetime.now(timezone.utc)
    report = calculate_drift_report(
        baseline=baseline,
        telemetry_records=records_from(frame, predictions, probabilities),
        window_start=now - timedelta(days=1),
        window_end=now,
    )
    assert report["overall_status"] == "ok"
    assert report["feature_drift"]["age"]["psi"] == 0.0
    assert report["prediction_drift"]["class_js_divergence"] == 0.0


def test_shifted_features_and_predictions_are_critical() -> None:
    baseline, frame, _, _ = make_baseline()
    shifted = frame.copy()
    shifted["age"] = shifted["age"] + 70
    shifted["city"] = "New City"
    predictions = ["Rejected"] * len(shifted)
    probabilities = np.asarray([[0.01, 0.99]] * len(shifted))
    now = datetime.now(timezone.utc)
    report = calculate_drift_report(
        baseline=baseline,
        telemetry_records=records_from(shifted, predictions, probabilities),
        window_start=now - timedelta(days=1),
        window_end=now,
    )
    assert report["overall_status"] == "critical"
    assert report["feature_drift"]["age"]["status"] == "critical"
    assert report["feature_drift"]["city"]["unseen_category_rate"] == 1.0
    assert report["prediction_drift"]["status"] == "critical"
