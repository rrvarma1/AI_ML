# Hospital Analytics Prediction API

FastAPI inference service for:

- Claim outcome classification using `claim_random_forest_hypertuned_v3.pkl`, trained by `03_claim_model_hypertuned_v3.ipynb`.
- Patient visit risk classification using `risk_random_forest_smote_undersampling_v5.pkl`, trained by `02_risk_model_SMOTE_Undersampling_v5.ipynb`.

The service loads persisted training pipelines, including preprocessing and risk interaction-feature generation. It does not execute notebooks at request time.

## Install and run

From the Capstone project root:

```bash
python -m pip install -r requirements-api.txt
export PREDICTION_HASH_KEY="$(openssl rand -hex 32)"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` for the generated OpenAPI interface.

The default model paths are resolved under the project root. Override them with the variables documented in `.env.example` when artifacts are stored elsewhere.

`PREDICTION_HASH_KEY` is required and must contain at least 32 bytes. Store it in a secret manager or deployment secret, not in source control. Keep the same key across replicas and restarts when hashes need to remain comparable.

## Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health/live` | Process liveness; does not require loaded models |
| `GET` | `/health/ready` | Service readiness; returns `503` unless both models are loaded and validated |
| `GET` | `/health/models` | Per-model status, artifact path, and class contract |
| `POST` | `/v1/predictions/claim-outcome` | Single claim prediction |
| `POST` | `/v1/predictions/claim-outcome/batch` | Batch claim predictions |
| `POST` | `/v1/predictions/visit-risk` | Single patient visit risk prediction |
| `POST` | `/v1/predictions/visit-risk/batch` | Batch patient visit risk predictions |

Every response includes `X-Request-ID` and `X-Process-Time-Ms`. Supply `X-Request-ID` to propagate a caller trace identifier.

## Prediction audit logs

Every successful prediction writes one compact JSON audit event to the `prediction_audit` logger. Batch requests produce one event per instance. Events include the UTC timestamp, request ID, logical model version, exact model artifact SHA-256, HMAC-SHA256 input feature hash, prediction, probabilities, record index, batch size, and inference duration.

Raw input features and the HMAC secret are never logged. The artifact hash is also returned by `/health/ready` and `/health/models` so operations can identify the exact deployed model. Route JSON stdout to a durable log platform and apply appropriate access controls and retention policies in production.

## Input data-quality monitoring

The API validates required columns, missing values, numeric types/ranges, fitted-model categories, calendar consistency, and business rules immediately before inference. `MISSING_VALUE_POLICY` and `UNSEEN_CATEGORY_POLICY` accept `warn` or `reject`; both default to `warn` because trained imputers and legitimate new operational categories may otherwise interrupt service. Controlled enum violations remain errors.

Warnings and errors are emitted as privacy-safe JSON on the `data_quality_validation` logger. Offline batch validation and category-contract export are documented in `README_MONITORING.md`.

## Oracle production telemetry and drift

When `MONITORING_ENABLED=true`, every validated feature record, prediction, and probability set is encrypted before being written to the restricted Oracle monitoring schema. `MONITORING_FAILURE_POLICY=fail_closed` is recommended so a prediction is not returned when required telemetry cannot be persisted.

Training baseline creation, Oracle schema installation, encryption-key rotation, retention, and scheduled feature/prediction drift calculations are documented in `README_MONITORING.md`.

## Example claim request

```bash
curl -X POST http://localhost:8000/v1/predictions/claim-outcome \
  -H 'Content-Type: application/json' \
  -d '{
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
    "doctor_id": 169
  }'
```

## Example risk request

```bash
curl -X POST http://localhost:8000/v1/predictions/visit-risk \
  -H 'Content-Type: application/json' \
  -d '{
    "age": 53,
    "chronic_flag": 0,
    "length_of_stay_hours": 3.48,
    "billed_amount": 23577.37,
    "approved_amount": 0,
    "payment_days": 16,
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
    "doctor_id": 169
  }'
```

## Validation and errors

All model inputs are required. Nullable values are accepted for features handled by trained imputers. Unknown fields, blank strings, invalid categories, non-finite numbers, out-of-range values, inconsistent date-derived fields, and `approved_amount > billed_amount` are rejected with a structured `422` response.

Model loading failures produce `503 model_unavailable`; sanitized inference failures produce `500 prediction_failed`. The detailed load cause is available from readiness/model health endpoints and server logs.

## Tests

The tests load both real artifacts and exercise health, prediction, batch, and validation behavior:

```bash
CAPSTONE_PROJECT_ROOT="$PWD" pytest -q tests
```

These predictions are decision-support outputs. They require human review and must not autonomously determine clinical treatment or insurance disposition.
