"""Validated API request and response contracts."""

# Summary:
# 1. Purpose: Defines all FastAPI request, response, health, and error data structures.
# 2. What it does: Converts JSON into typed Claim V3 and Risk V5 model inputs.
# 3. Invoked by: app/main.py for endpoint validation and OpenAPI documentation.
# 4. Main functions/classes: Prediction input, batch, result, health, and error models.
# 5. Validations/controls: Rejects extra fields, blank text, bad ranges, and inconsistent business fields.

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


NonEmptyText = Annotated[str, Field(min_length=1, max_length=100)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class StrictInputModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    @field_validator("*", mode="before")
    @classmethod
    def reject_blank_strings(cls, value):
        if isinstance(value, str) and not value.strip():
            raise ValueError("blank strings are not allowed")
        return value


class ClaimPredictionInput(StrictInputModel):
    age: int | None = Field(..., ge=0, le=120)
    chronic_flag: Literal[0, 1] | None = Field(...)
    length_of_stay_hours: float | None = Field(..., ge=0, le=100_000, allow_inf_nan=False)
    billed_amount: float | None = Field(..., ge=0, le=1_000_000_000, allow_inf_nan=False)
    days_from_visit_to_billing: int | None = Field(..., ge=-36_500, le=36_500)
    days_since_registration_at_visit: int | None = Field(..., ge=0, le=36_500)
    billing_month: int | None = Field(..., ge=1, le=12)
    billing_quarter: int | None = Field(..., ge=1, le=4)
    billing_day_of_week: int | None = Field(..., ge=0, le=6)
    billing_is_weekend: Literal[0, 1] | None = Field(...)
    visit_month: int | None = Field(..., ge=1, le=12)
    visit_quarter: int | None = Field(..., ge=1, le=4)
    visit_day_of_week: int | None = Field(..., ge=0, le=6)
    visit_is_weekend: Literal[0, 1] | None = Field(...)
    gender: Literal["F", "M"] | None = Field(...)
    city: NonEmptyText | None = Field(...)
    insurance_provider: NonEmptyText | None = Field(...)
    department: NonEmptyText | None = Field(...)
    visit_type: NonEmptyText | None = Field(...)
    risk_score: Literal["High", "Low", "Medium"] | None = Field(...)
    doctor_id: int | None = Field(..., ge=1, le=10_000_000)

    @model_validator(mode="after")
    def validate_calendar_consistency(self):
        _validate_quarter("billing", self.billing_month, self.billing_quarter)
        _validate_quarter("visit", self.visit_month, self.visit_quarter)
        _validate_weekend("billing", self.billing_day_of_week, self.billing_is_weekend)
        _validate_weekend("visit", self.visit_day_of_week, self.visit_is_weekend)
        return self


class RiskPredictionInput(StrictInputModel):
    age: int | None = Field(..., ge=0, le=120)
    chronic_flag: Literal[0, 1] | None = Field(...)
    length_of_stay_hours: float | None = Field(..., ge=0, le=100_000, allow_inf_nan=False)
    billed_amount: float | None = Field(..., ge=0, le=1_000_000_000, allow_inf_nan=False)
    approved_amount: float | None = Field(..., ge=0, le=1_000_000_000, allow_inf_nan=False)
    payment_days: float | None = Field(..., ge=0, le=36_500, allow_inf_nan=False)
    days_since_registration_at_visit: int | None = Field(..., ge=0, le=36_500)
    visit_year: int | None = Field(..., ge=2000, le=2100)
    visit_month: int | None = Field(..., ge=1, le=12)
    visit_quarter: int | None = Field(..., ge=1, le=4)
    visit_day_of_week: int | None = Field(..., ge=0, le=6)
    visit_is_weekend: Literal[0, 1] | None = Field(...)
    gender: Literal["F", "M"] | None = Field(...)
    city: NonEmptyText | None = Field(...)
    insurance_provider: NonEmptyText | None = Field(...)
    department: NonEmptyText | None = Field(...)
    visit_type: NonEmptyText | None = Field(...)
    claim_status: Literal["Paid", "Pending", "Rejected"] | None = Field(...)
    doctor_id: int | None = Field(..., ge=1, le=10_000_000)

    @model_validator(mode="after")
    def validate_business_rules(self):
        _validate_quarter("visit", self.visit_month, self.visit_quarter)
        _validate_weekend("visit", self.visit_day_of_week, self.visit_is_weekend)
        if (
            self.billed_amount is not None
            and self.approved_amount is not None
            and self.approved_amount > self.billed_amount
        ):
            raise ValueError("approved_amount cannot exceed billed_amount")
        return self


def _validate_quarter(prefix: str, month: int | None, quarter: int | None) -> None:
    if month is not None and quarter is not None:
        expected = ((month - 1) // 3) + 1
        if quarter != expected:
            raise ValueError(f"{prefix}_quarter must be {expected} for {prefix}_month={month}")


def _validate_weekend(prefix: str, day_of_week: int | None, is_weekend: int | None) -> None:
    if day_of_week is not None and is_weekend is not None:
        expected = int(day_of_week in (5, 6))
        if is_weekend != expected:
            raise ValueError(
                f"{prefix}_is_weekend must be {expected} for {prefix}_day_of_week={day_of_week}"
            )


class ClaimBatchRequest(StrictInputModel):
    instances: list[ClaimPredictionInput] = Field(..., min_length=1, max_length=10_000)


class RiskBatchRequest(StrictInputModel):
    instances: list[RiskPredictionInput] = Field(..., min_length=1, max_length=10_000)


class PredictionResult(BaseModel):
    prediction: str
    probabilities: dict[str, Probability]


class PredictionResponse(BaseModel):
    request_id: str
    model_id: str
    model_name: str
    predicted_at: datetime
    result: PredictionResult


class BatchPredictionResponse(BaseModel):
    request_id: str
    model_id: str
    model_name: str
    predicted_at: datetime
    count: int
    results: list[PredictionResult]


class LivenessResponse(BaseModel):
    status: Literal["alive"]
    service: str
    version: str


class ModelHealth(BaseModel):
    model_id: str
    ready: bool
    model_name: str
    artifact_path: str
    classes: list[str]
    artifact_sha256: str | None = None
    category_contract_sha256: str | None = None
    categorical_feature_count: int = 0
    error: str | None = None


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    models: dict[str, ModelHealth]


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str
    details: object | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
