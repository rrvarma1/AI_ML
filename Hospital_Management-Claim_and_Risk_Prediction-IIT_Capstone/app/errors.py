"""Service-specific exceptions with safe client-facing messages."""

# Summary:
# 1. Purpose: Defines safe application errors returned by the prediction API.
# 2. What it does: Stores HTTP status, error code, message, and optional safe details.
# 3. Invoked by: app/main.py and app/model_registry.py when requests or inference fail.
# 4. Main functions/classes: APIError, ModelUnavailableError, and PredictionError.
# 5. Validations/controls: Keeps internal exception details out of client-facing responses.

from __future__ import annotations


class APIError(Exception):
    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        details: object | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


class ModelUnavailableError(APIError):
    def __init__(self, model_id: str) -> None:
        super().__init__(
            status_code=503,
            code="model_unavailable",
            message=f"Model '{model_id}' is not ready",
        )


class PredictionError(APIError):
    def __init__(self, model_id: str) -> None:
        super().__init__(
            status_code=500,
            code="prediction_failed",
            message=f"Prediction failed for model '{model_id}'",
        )

