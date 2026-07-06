"""Training baseline profiles and production drift calculations."""

# Summary:
# 1. Purpose: Builds aggregate training profiles and compares encrypted production records over time.
# 2. Functions/validations: Calculates PSI, Jensen-Shannon, missing, unseen, mean, and confidence drift safely.

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.data_validation import NUMERIC_RULES


MISSING_CATEGORY = "__MISSING__"
STATUS_ORDER = {"ok": 0, "warning": 1, "critical": 2}


@dataclass(frozen=True)
class DriftThresholds:
    psi_warning: float = 0.10
    psi_critical: float = 0.25
    js_warning: float = 0.10
    js_critical: float = 0.20
    missing_delta_warning: float = 0.02
    missing_delta_critical: float = 0.05
    unseen_rate_warning: float = 0.01
    unseen_rate_critical: float = 0.05
    standardized_mean_warning: float = 0.25
    standardized_mean_critical: float = 0.50
    confidence_delta_warning: float = 0.05
    confidence_delta_critical: float = 0.10


def create_training_baseline(
    *,
    model_id: str,
    artifact_sha256: str,
    category_contract_sha256: str,
    training_frame: pd.DataFrame,
    predictions: list[str],
    probabilities: np.ndarray,
    classes: list[str],
    training_period: dict[str, str],
    source_files: dict[str, str],
) -> dict[str, Any]:
    if len(training_frame) == 0:
        raise ValueError("Training baseline requires at least one record")
    if len(predictions) != len(training_frame) or probabilities.shape[0] != len(training_frame):
        raise ValueError("Training predictions must align with training_frame")
    numeric_features = set(NUMERIC_RULES[model_id])
    feature_profiles = {}
    for feature in training_frame.columns:
        if feature in numeric_features:
            feature_profiles[feature] = _numeric_profile(training_frame[feature])
        else:
            feature_profiles[feature] = _categorical_profile(training_frame[feature])
    max_probabilities = probabilities.max(axis=1)
    return {
        "schema_version": 1,
        "model_id": model_id,
        "artifact_sha256": artifact_sha256,
        "category_contract_sha256": category_contract_sha256,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "baseline_population": "earliest_80_percent_chronological_training_split",
        "training_period": training_period,
        "record_count": len(training_frame),
        "source_file_sha256": source_files,
        "feature_profiles": feature_profiles,
        "prediction_profile": {
            "class_distribution": _distribution(pd.Series(predictions), classes),
            "confidence": _numeric_profile(pd.Series(max_probabilities)),
            "mean_max_probability": float(np.mean(max_probabilities)),
            "low_confidence_rate": float(np.mean(max_probabilities < 0.50)),
        },
    }


def calculate_drift_report(
    *,
    baseline: dict[str, Any],
    telemetry_records: list[dict[str, Any]],
    window_start: datetime,
    window_end: datetime,
    thresholds: DriftThresholds | None = None,
) -> dict[str, Any]:
    if not telemetry_records:
        raise ValueError("At least one telemetry record is required for drift calculation")
    thresholds = thresholds or DriftThresholds()
    frame = pd.DataFrame([record["features"] for record in telemetry_records])
    feature_metrics: dict[str, Any] = {}
    for feature, profile in baseline["feature_profiles"].items():
        if feature not in frame.columns:
            feature_metrics[feature] = {
                "type": profile["type"],
                "status": "critical",
                "error": "feature_missing_from_production_telemetry",
            }
            continue
        if profile["type"] == "numeric":
            feature_metrics[feature] = _numeric_drift(frame[feature], profile, thresholds)
        else:
            feature_metrics[feature] = _categorical_drift(frame[feature], profile, thresholds)

    predictions = pd.Series([record["prediction"] for record in telemetry_records], dtype="string")
    max_probabilities = pd.Series(
        [float(record["max_probability"]) for record in telemetry_records], dtype=float
    )
    baseline_prediction = baseline["prediction_profile"]
    current_distribution = _distribution(predictions)
    baseline_distribution = baseline_prediction["class_distribution"]
    prediction_js = jensen_shannon_divergence(baseline_distribution, current_distribution)
    prediction_psi = population_stability_index(
        list(baseline_distribution.values()),
        [current_distribution.get(label, 0.0) for label in baseline_distribution],
    )
    confidence = _numeric_drift(max_probabilities, baseline_prediction["confidence"], thresholds)
    mean_confidence = float(max_probabilities.mean())
    confidence_delta = mean_confidence - float(baseline_prediction["mean_max_probability"])
    low_confidence_rate = float((max_probabilities < 0.50).mean())
    prediction_status = _max_status(
        _threshold_status(prediction_js, thresholds.js_warning, thresholds.js_critical),
        _threshold_status(prediction_psi, thresholds.psi_warning, thresholds.psi_critical),
        _threshold_status(
            abs(confidence_delta),
            thresholds.confidence_delta_warning,
            thresholds.confidence_delta_critical,
        ),
        confidence["status"],
    )
    feature_statuses = [metric["status"] for metric in feature_metrics.values()]
    overall_status = _max_status(*feature_statuses, prediction_status)
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model_id": baseline["model_id"],
        "artifact_sha256": baseline["artifact_sha256"],
        "baseline_created_at": baseline["created_at"],
        "window": {
            "start": _iso_utc(window_start),
            "end": _iso_utc(window_end),
        },
        "record_count": len(telemetry_records),
        "overall_status": overall_status,
        "summary": {
            "warning_features": sum(status == "warning" for status in feature_statuses),
            "critical_features": sum(status == "critical" for status in feature_statuses),
            "prediction_status": prediction_status,
        },
        "thresholds": asdict(thresholds),
        "feature_drift": feature_metrics,
        "prediction_drift": {
            "status": prediction_status,
            "class_distribution_baseline": baseline_distribution,
            "class_distribution_current": current_distribution,
            "class_js_divergence": prediction_js,
            "class_psi": prediction_psi,
            "mean_max_probability_baseline": baseline_prediction["mean_max_probability"],
            "mean_max_probability_current": mean_confidence,
            "mean_max_probability_delta": confidence_delta,
            "low_confidence_rate_baseline": baseline_prediction["low_confidence_rate"],
            "low_confidence_rate_current": low_confidence_rate,
            "confidence_distribution": confidence,
        },
    }


def population_stability_index(expected: list[float], actual: list[float]) -> float:
    if len(expected) != len(actual) or not expected:
        raise ValueError("PSI distributions must be non-empty and have equal lengths")
    expected_array = _normalized(np.asarray(expected, dtype=float))
    actual_array = _normalized(np.asarray(actual, dtype=float))
    epsilon = 1e-6
    expected_array = np.clip(expected_array, epsilon, None)
    actual_array = np.clip(actual_array, epsilon, None)
    return float(np.sum((actual_array - expected_array) * np.log(actual_array / expected_array)))


def jensen_shannon_divergence(
    expected: dict[str, float], actual: dict[str, float]
) -> float:
    labels = sorted(set(expected) | set(actual))
    p = _normalized(np.asarray([expected.get(label, 0.0) for label in labels], dtype=float))
    q = _normalized(np.asarray([actual.get(label, 0.0) for label in labels], dtype=float))
    midpoint = 0.5 * (p + q)
    return float(0.5 * _kl_divergence(p, midpoint) + 0.5 * _kl_divergence(q, midpoint))


def _numeric_profile(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    missing_rate = float(1.0 - len(finite) / max(len(series), 1))
    if finite.empty:
        return {
            "type": "numeric",
            "missing_rate": missing_rate,
            "mean": None,
            "std": None,
            "minimum": None,
            "maximum": None,
            "median": None,
            "cut_points": [],
            "bin_proportions": [1.0],
        }
    cut_points = sorted(
        set(float(value) for value in finite.quantile(np.arange(0.1, 1.0, 0.1)).tolist())
    )
    return {
        "type": "numeric",
        "missing_rate": missing_rate,
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=0)),
        "minimum": float(finite.min()),
        "maximum": float(finite.max()),
        "median": float(finite.median()),
        "cut_points": cut_points,
        "bin_proportions": _histogram_proportions(finite, cut_points),
    }


def _categorical_profile(series: pd.Series) -> dict[str, Any]:
    normalized = _categorical_series(series)
    return {
        "type": "categorical",
        "missing_rate": float((normalized == MISSING_CATEGORY).mean()),
        "distribution": _distribution(normalized),
        "known_categories": sorted(
            value for value in normalized.unique().tolist() if value != MISSING_CATEGORY
        ),
    }


def _numeric_drift(
    series: pd.Series, baseline: dict[str, Any], thresholds: DriftThresholds
) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    missing_rate = float(1.0 - len(finite) / max(len(series), 1))
    missing_delta = missing_rate - float(baseline["missing_rate"])
    current_bins = _histogram_proportions(finite, baseline["cut_points"])
    psi = population_stability_index(baseline["bin_proportions"], current_bins)
    current_mean = float(finite.mean()) if not finite.empty else None
    baseline_mean = baseline["mean"]
    baseline_std = baseline["std"]
    standardized_mean_shift = None
    if current_mean is not None and baseline_mean is not None:
        scale = float(baseline_std) if baseline_std and baseline_std > 1e-12 else 1.0
        standardized_mean_shift = abs(current_mean - float(baseline_mean)) / scale
    statuses = [
        _threshold_status(psi, thresholds.psi_warning, thresholds.psi_critical),
        _threshold_status(
            abs(missing_delta),
            thresholds.missing_delta_warning,
            thresholds.missing_delta_critical,
        ),
    ]
    if standardized_mean_shift is not None:
        statuses.append(
            _threshold_status(
                standardized_mean_shift,
                thresholds.standardized_mean_warning,
                thresholds.standardized_mean_critical,
            )
        )
    return {
        "type": "numeric",
        "status": _max_status(*statuses),
        "psi": psi,
        "missing_rate_baseline": baseline["missing_rate"],
        "missing_rate_current": missing_rate,
        "missing_rate_delta": missing_delta,
        "mean_baseline": baseline_mean,
        "mean_current": current_mean,
        "standardized_mean_shift": standardized_mean_shift,
    }


def _categorical_drift(
    series: pd.Series, baseline: dict[str, Any], thresholds: DriftThresholds
) -> dict[str, Any]:
    normalized = _categorical_series(series)
    current_distribution = _distribution(normalized)
    js = jensen_shannon_divergence(baseline["distribution"], current_distribution)
    known = set(baseline["known_categories"])
    unseen_rate = float(
        normalized.map(lambda value: value != MISSING_CATEGORY and value not in known).mean()
    )
    missing_rate = float((normalized == MISSING_CATEGORY).mean())
    missing_delta = missing_rate - float(baseline["missing_rate"])
    return {
        "type": "categorical",
        "status": _max_status(
            _threshold_status(js, thresholds.js_warning, thresholds.js_critical),
            _threshold_status(
                unseen_rate,
                thresholds.unseen_rate_warning,
                thresholds.unseen_rate_critical,
            ),
            _threshold_status(
                abs(missing_delta),
                thresholds.missing_delta_warning,
                thresholds.missing_delta_critical,
            ),
        ),
        "js_divergence": js,
        "unseen_category_rate": unseen_rate,
        "missing_rate_baseline": baseline["missing_rate"],
        "missing_rate_current": missing_rate,
        "missing_rate_delta": missing_delta,
    }


def _categorical_series(series: pd.Series) -> pd.Series:
    def normalize(value: Any) -> str:
        if pd.isna(value) or (isinstance(value, str) and not value.strip()):
            return MISSING_CATEGORY
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
            return str(int(value))
        return str(value).strip()

    return series.map(normalize).astype("string")


def _distribution(series: pd.Series, labels: list[str] | None = None) -> dict[str, float]:
    counts = series.astype("string").value_counts(dropna=False)
    values = {str(label): float(count / max(len(series), 1)) for label, count in counts.items()}
    if labels:
        return {label: values.get(label, 0.0) for label in labels}
    return dict(sorted(values.items()))


def _histogram_proportions(series: pd.Series, cut_points: list[float]) -> list[float]:
    if series.empty:
        return [0.0] * (len(cut_points) + 1)
    indices = np.searchsorted(np.asarray(cut_points), series.to_numpy(dtype=float), side="right")
    counts = np.bincount(indices, minlength=len(cut_points) + 1)
    return (counts / counts.sum()).astype(float).tolist()


def _normalized(values: np.ndarray) -> np.ndarray:
    total = values.sum()
    if total <= 0:
        return np.full(len(values), 1.0 / len(values))
    return values / total


def _kl_divergence(first: np.ndarray, second: np.ndarray) -> float:
    mask = first > 0
    return float(np.sum(first[mask] * np.log(first[mask] / np.clip(second[mask], 1e-12, None))))


def _threshold_status(value: float, warning: float, critical: float) -> str:
    if value >= critical:
        return "critical"
    if value >= warning:
        return "warning"
    return "ok"


def _max_status(*statuses: str) -> str:
    return max(statuses or ("ok",), key=lambda status: STATUS_ORDER[status])


def _iso_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("Drift window timestamps must include a timezone")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
