"""Shared model-input quality checks for API and offline monitoring jobs."""

# Summary:
# 1. Purpose: Provides shared input-quality checks for live and offline model data.
# 2. What it does: Checks columns, missing values, numbers, categories, and business rules.
# 3. Invoked by: app/model_registry.py and both scripts in the monitoring package.
# 4. Main functions/classes: ValidationReport, validate_frame, validate_records, and contract helpers.
# 5. Validations/controls: Hides raw unseen values and emits privacy-safe JSON quality events.

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
import pandas as pd


CLAIM_MODEL_ID = "claim-v3"
RISK_MODEL_ID = "risk-v5"
ValidationPolicy = Literal["warn", "reject"]

EXPECTED_CLAIM_COLUMNS = [
    "age",
    "chronic_flag",
    "length_of_stay_hours",
    "billed_amount",
    "days_from_visit_to_billing",
    "days_since_registration_at_visit",
    "billing_month",
    "billing_quarter",
    "billing_day_of_week",
    "billing_is_weekend",
    "visit_month",
    "visit_quarter",
    "visit_day_of_week",
    "visit_is_weekend",
    "gender",
    "city",
    "insurance_provider",
    "department",
    "visit_type",
    "risk_score",
    "doctor_id",
]

EXPECTED_RISK_COLUMNS = [
    "age",
    "chronic_flag",
    "length_of_stay_hours",
    "billed_amount",
    "approved_amount",
    "payment_days",
    "days_since_registration_at_visit",
    "visit_year",
    "visit_month",
    "visit_quarter",
    "visit_day_of_week",
    "visit_is_weekend",
    "gender",
    "city",
    "insurance_provider",
    "department",
    "visit_type",
    "claim_status",
    "doctor_id",
]

EXPECTED_COLUMNS = {
    CLAIM_MODEL_ID: EXPECTED_CLAIM_COLUMNS,
    RISK_MODEL_ID: EXPECTED_RISK_COLUMNS,
}


@dataclass(frozen=True)
class NumericRule:
    minimum: float
    maximum: float
    integer: bool = False


NUMERIC_RULES = {
    CLAIM_MODEL_ID: {
        "age": NumericRule(0, 120, True),
        "chronic_flag": NumericRule(0, 1, True),
        "length_of_stay_hours": NumericRule(0, 100_000),
        "billed_amount": NumericRule(0, 1_000_000_000),
        "days_from_visit_to_billing": NumericRule(-36_500, 36_500, True),
        "days_since_registration_at_visit": NumericRule(0, 36_500, True),
        "billing_month": NumericRule(1, 12, True),
        "billing_quarter": NumericRule(1, 4, True),
        "billing_day_of_week": NumericRule(0, 6, True),
        "billing_is_weekend": NumericRule(0, 1, True),
        "visit_month": NumericRule(1, 12, True),
        "visit_quarter": NumericRule(1, 4, True),
        "visit_day_of_week": NumericRule(0, 6, True),
        "visit_is_weekend": NumericRule(0, 1, True),
        "doctor_id": NumericRule(1, 10_000_000, True),
    },
    RISK_MODEL_ID: {
        "age": NumericRule(0, 120, True),
        "chronic_flag": NumericRule(0, 1, True),
        "length_of_stay_hours": NumericRule(0, 100_000),
        "billed_amount": NumericRule(0, 1_000_000_000),
        "approved_amount": NumericRule(0, 1_000_000_000),
        "payment_days": NumericRule(0, 36_500),
        "days_since_registration_at_visit": NumericRule(0, 36_500, True),
        "visit_year": NumericRule(2000, 2100, True),
        "visit_month": NumericRule(1, 12, True),
        "visit_quarter": NumericRule(1, 4, True),
        "visit_day_of_week": NumericRule(0, 6, True),
        "visit_is_weekend": NumericRule(0, 1, True),
        "doctor_id": NumericRule(1, 10_000_000, True),
    },
}

# These values are governed enums. Other categorical features may legitimately
# gain new operational values and therefore follow the configured policy.
CONTROLLED_CATEGORIES = {
    CLAIM_MODEL_ID: {"gender", "risk_score"},
    RISK_MODEL_ID: {"gender", "claim_status"},
}

DATA_QUALITY_LOGGER = logging.getLogger("data_quality_validation")
DATA_QUALITY_LOGGER.setLevel(logging.INFO)
DATA_QUALITY_LOGGER.propagate = False
if not any(getattr(handler, "is_data_quality_handler", False) for handler in DATA_QUALITY_LOGGER.handlers):
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.is_data_quality_handler = True  # type: ignore[attr-defined]
    DATA_QUALITY_LOGGER.addHandler(handler)


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    severity: Literal["warning", "error"]
    feature: str | None
    count: int
    rate: float
    message: str
    distinct_count: int | None = None


@dataclass
class ValidationReport:
    model_id: str
    generated_at: str
    row_count: int
    expected_columns: list[str]
    observed_columns: list[str]
    category_contract_sha256: str
    issues: list[ValidationIssue]
    artifact_sha256: str | None = None

    @property
    def error_count(self) -> int:
        return sum(issue.severity == "error" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity == "warning" for issue in self.issues)

    @property
    def valid(self) -> bool:
        return self.error_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "generated_at": self.generated_at,
            "artifact_sha256": self.artifact_sha256,
            "category_contract_sha256": self.category_contract_sha256,
            "row_count": self.row_count,
            "expected_columns": self.expected_columns,
            "observed_columns": self.observed_columns,
            "valid": self.valid,
            "summary": {
                "error_count": self.error_count,
                "warning_count": self.warning_count,
            },
            "issues": [asdict(issue) for issue in self.issues],
        }

    def client_errors(self) -> list[dict[str, Any]]:
        return [asdict(issue) for issue in self.issues if issue.severity == "error"]


def extract_category_contract(pipeline: Any) -> dict[str, list[str]]:
    """Extract fitted OneHotEncoder categories from a persisted pipeline."""
    if not hasattr(pipeline, "named_steps") or "preprocessor" not in pipeline.named_steps:
        raise ValueError("Pipeline does not contain a fitted 'preprocessor' step")
    preprocessor = pipeline.named_steps["preprocessor"]
    transformer = None
    columns = None
    for name, candidate, candidate_columns in preprocessor.transformers_:
        if name == "cat":
            transformer = candidate
            columns = list(candidate_columns)
            break
    if transformer is None or columns is None:
        raise ValueError("Preprocessor does not contain a fitted 'cat' transformer")
    encoder = transformer
    if hasattr(transformer, "steps"):
        encoders = [step for _, step in transformer.steps if hasattr(step, "categories_")]
        if not encoders:
            raise ValueError("Categorical transformer does not contain a fitted encoder")
        encoder = encoders[-1]
    if not hasattr(encoder, "categories_"):
        raise ValueError("Categorical transformer does not expose fitted categories")
    if len(columns) != len(encoder.categories_):
        raise ValueError("Categorical feature and category array counts do not match")
    return {
        feature: sorted({_canonical_category(value) for value in values})
        for feature, values in zip(columns, encoder.categories_)
    }


def category_contract_sha256(contract: dict[str, list[str]]) -> str:
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frame(
    frame: pd.DataFrame,
    *,
    model_id: str,
    category_contract: dict[str, list[str]],
    missing_policy: ValidationPolicy = "warn",
    unseen_policy: ValidationPolicy = "warn",
    artifact_sha256: str | None = None,
) -> ValidationReport:
    if model_id not in EXPECTED_COLUMNS:
        raise ValueError(f"Unsupported model_id: {model_id}")
    if missing_policy not in {"warn", "reject"}:
        raise ValueError("missing_policy must be 'warn' or 'reject'")
    if unseen_policy not in {"warn", "reject"}:
        raise ValueError("unseen_policy must be 'warn' or 'reject'")

    expected = EXPECTED_COLUMNS[model_id]
    issues: list[ValidationIssue] = []
    row_count = len(frame)
    denominator = max(row_count, 1)
    observed = [str(column) for column in frame.columns]

    if row_count == 0:
        issues.append(ValidationIssue("empty_dataset", "error", None, 0, 0.0, "Input contains no records"))

    for feature in sorted(set(expected) - set(observed)):
        issues.append(
            ValidationIssue(
                "missing_column",
                "error",
                feature,
                row_count,
                1.0 if row_count else 0.0,
                f"Required feature column '{feature}' is missing",
            )
        )
    for feature in sorted(set(observed) - set(expected)):
        issues.append(
            ValidationIssue(
                "unexpected_column",
                "error",
                feature,
                row_count,
                1.0 if row_count else 0.0,
                f"Unexpected feature column '{feature}' is not in the model contract",
            )
        )

    for feature in expected:
        if feature not in frame.columns:
            continue
        series = frame[feature]
        missing_mask = _missing_mask(series)
        missing_count = int(missing_mask.sum())
        if missing_count:
            issues.append(
                ValidationIssue(
                    "missing_value",
                    "error" if missing_policy == "reject" else "warning",
                    feature,
                    missing_count,
                    missing_count / denominator,
                    f"Feature '{feature}' contains missing or blank values",
                )
            )

        if feature in NUMERIC_RULES[model_id]:
            _validate_numeric_feature(
                series,
                missing_mask,
                feature,
                NUMERIC_RULES[model_id][feature],
                denominator,
                issues,
            )

        if feature in category_contract:
            _validate_categorical_feature(
                series,
                missing_mask,
                model_id,
                feature,
                category_contract[feature],
                unseen_policy,
                denominator,
                issues,
            )

    _validate_business_rules(frame, model_id, denominator, issues)
    return ValidationReport(
        model_id=model_id,
        generated_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        artifact_sha256=artifact_sha256,
        category_contract_sha256=category_contract_sha256(category_contract),
        row_count=row_count,
        expected_columns=expected.copy(),
        observed_columns=observed,
        issues=issues,
    )


def validate_records(
    records: list[dict[str, Any]],
    **kwargs: Any,
) -> ValidationReport:
    return validate_frame(pd.DataFrame.from_records(records), **kwargs)


def log_validation_report(report: ValidationReport, request_id: str) -> None:
    if not report.issues:
        return
    event = {
        "event": "data_quality_validation",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "request_id": request_id,
        "model_id": report.model_id,
        "artifact_sha256": report.artifact_sha256,
        "category_contract_sha256": report.category_contract_sha256,
        "row_count": report.row_count,
        "valid": report.valid,
        "summary": {"error_count": report.error_count, "warning_count": report.warning_count},
        "issues": [asdict(issue) for issue in report.issues],
    }
    DATA_QUALITY_LOGGER.info(json.dumps(event, separators=(",", ":"), ensure_ascii=True))


def _missing_mask(series: pd.Series) -> pd.Series:
    mask = series.isna()
    if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
        mask = mask | series.astype("string").str.strip().eq("").fillna(False)
    return mask


def _canonical_category(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _validate_numeric_feature(
    series: pd.Series,
    missing_mask: pd.Series,
    feature: str,
    rule: NumericRule,
    denominator: int,
    issues: list[ValidationIssue],
) -> None:
    present = ~missing_mask
    numeric = pd.to_numeric(series, errors="coerce")
    invalid_type = present & numeric.isna()
    _append_mask_issue(
        issues,
        invalid_type,
        "invalid_numeric_type",
        feature,
        denominator,
        f"Feature '{feature}' contains values that are not numeric",
    )
    finite = pd.Series(np.isfinite(numeric.fillna(0).to_numpy(dtype=float)), index=series.index)
    non_finite = present & numeric.notna() & ~finite
    _append_mask_issue(
        issues,
        non_finite,
        "non_finite_value",
        feature,
        denominator,
        f"Feature '{feature}' contains NaN or infinite values",
    )
    comparable = present & numeric.notna() & finite
    out_of_range = comparable & ((numeric < rule.minimum) | (numeric > rule.maximum))
    _append_mask_issue(
        issues,
        out_of_range,
        "numeric_out_of_range",
        feature,
        denominator,
        f"Feature '{feature}' must be between {rule.minimum:g} and {rule.maximum:g}",
    )
    if rule.integer:
        fractional = comparable & ~np.isclose(numeric % 1, 0)
        _append_mask_issue(
            issues,
            fractional,
            "integer_required",
            feature,
            denominator,
            f"Feature '{feature}' must contain integer values",
        )


def _validate_categorical_feature(
    series: pd.Series,
    missing_mask: pd.Series,
    model_id: str,
    feature: str,
    allowed_values: list[str],
    unseen_policy: ValidationPolicy,
    denominator: int,
    issues: list[ValidationIssue],
) -> None:
    normalized = series.map(lambda value: _canonical_category(value) if not pd.isna(value) else None)
    unseen = ~missing_mask & ~normalized.isin(set(allowed_values))
    unseen_count = int(unseen.sum())
    if not unseen_count:
        return
    controlled = feature in CONTROLLED_CATEGORIES[model_id]
    severity: Literal["warning", "error"] = (
        "error" if controlled or unseen_policy == "reject" else "warning"
    )
    issues.append(
        ValidationIssue(
            "unseen_category",
            severity,
            feature,
            unseen_count,
            unseen_count / denominator,
            f"Feature '{feature}' contains categories absent from the fitted encoder",
            distinct_count=int(normalized[unseen].nunique()),
        )
    )


def _validate_business_rules(
    frame: pd.DataFrame,
    model_id: str,
    denominator: int,
    issues: list[ValidationIssue],
) -> None:
    def numeric(feature: str) -> pd.Series:
        return pd.to_numeric(frame[feature], errors="coerce")

    calendar_prefixes = ["visit"] if model_id == RISK_MODEL_ID else ["billing", "visit"]
    for prefix in calendar_prefixes:
        month_name = f"{prefix}_month"
        quarter_name = f"{prefix}_quarter"
        day_name = f"{prefix}_day_of_week"
        weekend_name = f"{prefix}_is_weekend"
        if month_name in frame and quarter_name in frame:
            month = numeric(month_name)
            quarter = numeric(quarter_name)
            valid = month.notna() & quarter.notna()
            mismatch = valid & (quarter != (((month - 1) // 3) + 1))
            _append_mask_issue(
                issues,
                mismatch,
                "inconsistent_calendar_field",
                quarter_name,
                denominator,
                f"Feature '{quarter_name}' is inconsistent with '{month_name}'",
            )
        if day_name in frame and weekend_name in frame:
            day = numeric(day_name)
            weekend = numeric(weekend_name)
            valid = day.notna() & weekend.notna()
            expected = day.isin([5, 6]).astype(int)
            mismatch = valid & (weekend != expected)
            _append_mask_issue(
                issues,
                mismatch,
                "inconsistent_calendar_field",
                weekend_name,
                denominator,
                f"Feature '{weekend_name}' is inconsistent with '{day_name}'",
            )

    if model_id == RISK_MODEL_ID and {"approved_amount", "billed_amount"}.issubset(frame.columns):
        approved = numeric("approved_amount")
        billed = numeric("billed_amount")
        violation = approved.notna() & billed.notna() & (approved > billed)
        _append_mask_issue(
            issues,
            violation,
            "business_rule_violation",
            "approved_amount",
            denominator,
            "Feature 'approved_amount' cannot exceed 'billed_amount'",
        )


def _append_mask_issue(
    issues: list[ValidationIssue],
    mask: pd.Series,
    code: str,
    feature: str,
    denominator: int,
    message: str,
) -> None:
    count = int(mask.sum())
    if count:
        issues.append(ValidationIssue(code, "error", feature, count, count / denominator, message))
