"""FastAPI application exposing health and model prediction endpoints."""

# Summary:
# 1. Purpose: Defines the FastAPI service, health checks, and prediction endpoints.
# 2. What it does: Validates requests, runs models, formats responses, and writes audit logs.
# 3. Invoked by: Uvicorn through app.main:app and by tests/test_api.py.
# 4. Main functions/classes: FastAPI app, lifespan, middleware, handlers, and endpoint functions.
# 5. Validations/controls: Applies schema and data-quality checks before every inference call.

# Drift monitoring change:
# 1. Purpose: Saves every validated feature record and prediction in the encrypted Oracle monitoring store.
# 2. Functions/validations: Initializes Oracle safely and applies fail-closed or warning behavior on write failure.

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .audit import input_feature_hash, log_prediction
from .config import settings
from .data_validation import log_validation_report
from .errors import APIError
from .model_registry import CLAIM_MODEL_ID, RISK_MODEL_ID, ModelRegistry
from .oracle_monitoring_store import (
    MonitoringStoreError,
    OracleMonitoringStore,
    OracleStoreConfig,
)
from .schemas import (
    BatchPredictionResponse,
    ClaimBatchRequest,
    ClaimPredictionInput,
    ErrorResponse,
    LivenessResponse,
    ModelHealth,
    PredictionResponse,
    PredictionResult,
    ReadinessResponse,
    RiskBatchRequest,
    RiskPredictionInput,
)


SERVICE_NAME = "hospital-analytics-inference"
SERVICE_VERSION = "1.0.0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger(__name__)
registry = ModelRegistry(settings)
monitoring_store: OracleMonitoringStore | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global monitoring_store
    registry.load_all()
    if settings.monitoring_enabled:
        try:
            monitoring_store = OracleMonitoringStore.from_config(
                OracleStoreConfig.from_environment()
            )
            monitoring_store.health_check()
            LOGGER.info("Oracle model-monitoring store is ready")
        except Exception:
            LOGGER.exception("Oracle model-monitoring store initialization failed")
            monitoring_store = None
            if settings.monitoring_failure_policy == "fail_closed":
                raise RuntimeError("Required Oracle monitoring store is unavailable")
    try:
        yield
    finally:
        if monitoring_store is not None:
            monitoring_store.close()
            monitoring_store = None


app = FastAPI(
    title="Hospital Analytics Prediction API",
    summary="Predictions from the selected Claim V3 and Risk V5 classifiers.",
    description=(
        "Provides validated inference for patient visit risk and insurance claim outcomes. "
        "Predictions support human review and must not be used as autonomous clinical or claim decisions."
    ),
    version=SERVICE_VERSION,
    lifespan=lifespan,
)


def request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


@app.middleware("http")
async def request_context(request: Request, call_next):
    supplied_request_id = request.headers.get("X-Request-ID", "").strip()
    request.state.request_id = supplied_request_id[:128] if supplied_request_id else str(uuid4())
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
    LOGGER.info(
        "request_id=%s method=%s path=%s status=%s",
        request.state.request_id,
        request.method,
        request.url.path,
        response.status_code,
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    details = [
        {
            "type": error.get("type", "value_error"),
            "loc": list(error.get("loc", ())),
            "msg": error.get("msg", "Invalid value"),
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": "Request validation failed",
                "request_id": request_id(request),
                "details": details,
            }
        },
    )


@app.exception_handler(APIError)
async def api_exception_handler(request: Request, exc: APIError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "request_id": request_id(request),
                "details": exc.details,
            }
        },
    )


@app.exception_handler(Exception)
async def unexpected_exception_handler(request: Request, exc: Exception):
    LOGGER.exception("Unhandled request failure request_id=%s", request_id(request))
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "internal_error",
                "message": "An unexpected internal error occurred",
                "request_id": request_id(request),
                "details": None,
            }
        },
    )


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "docs": "/docs",
        "health": "/health/ready",
    }


@app.get("/health/live", response_model=LivenessResponse, tags=["health"])
async def liveness() -> LivenessResponse:
    return LivenessResponse(status="alive", service=SERVICE_NAME, version=SERVICE_VERSION)


@app.get(
    "/health/ready",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse}},
    tags=["health"],
)
async def readiness():
    payload = ReadinessResponse(
        status="ready" if registry.ready() else "not_ready",
        models={key: ModelHealth(**value) for key, value in registry.health().items()},
    )
    if not registry.ready():
        return JSONResponse(status_code=503, content=payload.model_dump(mode="json"))
    return payload


@app.get("/health/models", response_model=dict[str, ModelHealth], tags=["health"])
async def model_health() -> dict[str, ModelHealth]:
    return {key: ModelHealth(**value) for key, value in registry.health().items()}


def _single_response(request: Request, model_id: str, record: dict) -> PredictionResponse:
    validation = registry.validate_inputs(model_id, [record])
    log_validation_report(validation, request_id(request))
    if not validation.valid:
        raise APIError(
            status_code=422,
            code="input_data_quality_error",
            message="Input failed model data-quality validation",
            details=validation.client_errors(),
        )
    started = time.perf_counter()
    result = registry.predict(model_id, [record])[0]
    duration_ms = (time.perf_counter() - started) * 1000
    model = registry.models[model_id]
    predicted_at = datetime.now(timezone.utc)
    if model.artifact_sha256 is None:
        raise APIError(status_code=503, code="model_unavailable", message=f"Model '{model_id}' is not ready")
    feature_hash = input_feature_hash(record, settings.prediction_hash_key)
    _store_monitoring_telemetry(
        request=request,
        model_id=model_id,
        artifact_sha256=model.artifact_sha256,
        predicted_at=predicted_at,
        records=[record],
        results=[result],
        feature_hashes=[feature_hash],
    )
    log_prediction(
        timestamp=predicted_at,
        request_id=request_id(request),
        model_id=model_id,
        model_name=model.model_name,
        artifact_sha256=model.artifact_sha256,
        feature_hash=feature_hash,
        prediction=result["prediction"],
        probabilities=result["probabilities"],
        duration_ms=duration_ms,
        record_index=0,
        batch_size=1,
    )
    return PredictionResponse(
        request_id=request_id(request),
        model_id=model_id,
        model_name=model.model_name,
        predicted_at=predicted_at,
        result=PredictionResult(**result),
    )


def _batch_response(request: Request, model_id: str, records: list[dict]) -> BatchPredictionResponse:
    if len(records) > settings.max_batch_size:
        raise APIError(
            status_code=422,
            code="batch_too_large",
            message=f"Batch contains {len(records)} instances; maximum is {settings.max_batch_size}",
        )
    validation = registry.validate_inputs(model_id, records)
    log_validation_report(validation, request_id(request))
    if not validation.valid:
        raise APIError(
            status_code=422,
            code="input_data_quality_error",
            message="Input failed model data-quality validation",
            details=validation.client_errors(),
        )
    started = time.perf_counter()
    results = registry.predict(model_id, records)
    duration_ms = (time.perf_counter() - started) * 1000
    model = registry.models[model_id]
    predicted_at = datetime.now(timezone.utc)
    if model.artifact_sha256 is None:
        raise APIError(status_code=503, code="model_unavailable", message=f"Model '{model_id}' is not ready")
    feature_hashes = [
        input_feature_hash(record, settings.prediction_hash_key) for record in records
    ]
    _store_monitoring_telemetry(
        request=request,
        model_id=model_id,
        artifact_sha256=model.artifact_sha256,
        predicted_at=predicted_at,
        records=records,
        results=results,
        feature_hashes=feature_hashes,
    )
    for index, (record, result, feature_hash) in enumerate(
        zip(records, results, feature_hashes)
    ):
        log_prediction(
            timestamp=predicted_at,
            request_id=request_id(request),
            model_id=model_id,
            model_name=model.model_name,
            artifact_sha256=model.artifact_sha256,
            feature_hash=feature_hash,
            prediction=result["prediction"],
            probabilities=result["probabilities"],
            duration_ms=duration_ms,
            record_index=index,
            batch_size=len(records),
        )
    return BatchPredictionResponse(
        request_id=request_id(request),
        model_id=model_id,
        model_name=model.model_name,
        predicted_at=predicted_at,
        count=len(results),
        results=[PredictionResult(**result) for result in results],
    )


def _store_monitoring_telemetry(
    *,
    request: Request,
    model_id: str,
    artifact_sha256: str,
    predicted_at: datetime,
    records: list[dict],
    results: list[dict],
    feature_hashes: list[str],
) -> None:
    if monitoring_store is None:
        return
    try:
        monitoring_store.store_predictions(
            model_id=model_id,
            artifact_sha256=artifact_sha256,
            request_id=request_id(request),
            recorded_at=predicted_at,
            records=records,
            results=results,
            input_hashes=feature_hashes,
        )
    except (MonitoringStoreError, ValueError):
        LOGGER.exception(
            "Oracle monitoring telemetry write failed request_id=%s model_id=%s",
            request_id(request),
            model_id,
        )
        if settings.monitoring_failure_policy == "fail_closed":
            raise APIError(
                status_code=503,
                code="monitoring_store_unavailable",
                message="Required prediction monitoring could not be recorded",
            )


@app.post(
    "/v1/predictions/claim-outcome",
    response_model=PredictionResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["predictions"],
)
def predict_claim(payload: ClaimPredictionInput, request: Request) -> PredictionResponse:
    return _single_response(request, CLAIM_MODEL_ID, payload.model_dump())


@app.post(
    "/v1/predictions/claim-outcome/batch",
    response_model=BatchPredictionResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["predictions"],
)
def predict_claim_batch(payload: ClaimBatchRequest, request: Request) -> BatchPredictionResponse:
    return _batch_response(
        request,
        CLAIM_MODEL_ID,
        [instance.model_dump() for instance in payload.instances],
    )


@app.post(
    "/v1/predictions/visit-risk",
    response_model=PredictionResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["predictions"],
)
def predict_risk(payload: RiskPredictionInput, request: Request) -> PredictionResponse:
    return _single_response(request, RISK_MODEL_ID, payload.model_dump())


@app.post(
    "/v1/predictions/visit-risk/batch",
    response_model=BatchPredictionResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["predictions"],
)
def predict_risk_batch(payload: RiskBatchRequest, request: Request) -> BatchPredictionResponse:
    return _batch_response(
        request,
        RISK_MODEL_ID,
        [instance.model_dump() for instance in payload.instances],
    )
