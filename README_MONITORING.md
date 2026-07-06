# Model Input Data-Quality Monitoring

The monitoring commands use the same feature contracts as the FastAPI service and extract categorical vocabularies from the fitted model pipelines. They check:

- Required and unexpected columns.
- Missing and blank values.
- Numeric types, finite values, integer requirements, and allowed ranges.
- Categories absent from each fitted `OneHotEncoder`.
- Month/quarter and weekday/weekend consistency.
- The Risk V5 rule that `approved_amount` cannot exceed `billed_amount`.

Reports never include raw field values or unseen-category values.

## Export fitted category contracts

Run after every approved model release:

```bash
python monitoring/export_category_contracts.py
```

This creates:

```text
models/monitoring/claim-v3-category-contract.json
models/monitoring/risk-v5-category-contract.json
```

The service exposes each contract SHA-256 through `/health/ready` and `/health/models`.

## Validate an input batch

Input files must contain raw model features, not one-hot encoded or interaction features.

```bash
python monitoring/validate_model_inputs.py data/incoming/claims.csv \
  --model claim-v3 \
  --output metrics/monitoring/claim-input-quality.json
```

Strict production-gate example:

```bash
python monitoring/validate_model_inputs.py data/incoming/risk.jsonl \
  --model risk-v5 \
  --missing-policy reject \
  --unseen-policy reject \
  --fail-on-warnings \
  --output metrics/monitoring/risk-input-quality.json
```

Supported formats are CSV, JSON, JSONL/NDJSON, and Parquet.

Exit codes:

| Code | Meaning |
|---:|---|
| `0` | No validation errors; warnings permitted unless `--fail-on-warnings` is set |
| `1` | Script, file, artifact, or format failure |
| `2` | Validation failed or warning was promoted to failure |

## Live API policy

The API applies the same checks immediately before inference. Controlled enum violations are rejected. New operational categories and nullable inputs produce privacy-safe JSON events by default so normal hospital changes do not automatically cause an outage.

Configure behavior with:

```text
MISSING_VALUE_POLICY=warn|reject
UNSEEN_CATEGORY_POLICY=warn|reject
```

`warn` is the default. In strict environments, set both to `reject`. Data-quality events are written to the `data_quality_validation` logger and contain counts, rates, feature names, model/artifact identity, and contract identity without raw values.

## Oracle drift-monitoring store

The production monitoring store uses Oracle, not local files. Install the schema as the monitoring schema owner:

```bash
sqlplus monitoring_schema_owner@database @monitoring/oracle_monitoring_schema.sql
```

Ask the Oracle DBA to replace `MONITORING_SCHEMA` in `monitoring/oracle_monitoring_access.sql`, run it, and grant separate least-privilege roles to the API writer, drift reader, and monitoring administrator. The schema owner also needs `CREATE JOB` for the daily retention job. Oracle TDE must protect the tablespace and backups.

Validated feature and prediction payloads are encrypted with Fernet before insertion into the `ML_PREDICTION_TELEMETRY` BLOB. Only model/version, timestamps, key ID, expiry, and ciphertext integrity metadata remain queryable. Baseline aggregates are stored in `ML_MONITOR_BASELINE`, and drift results are stored in `ML_DRIFT_REPORT`.

Generate the primary encryption key in a secure administrative session and store it in the hospital secret manager:

```bash
python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'
```

Set `MONITORING_ENCRYPTION_KEY` to the active key. During rotation, move the previous key to the comma-separated `MONITORING_DECRYPTION_KEYS` setting until all retained records encrypted with it have expired.

Enable API telemetry only after schema, grants, TDE, secrets, and connectivity are verified:

```text
MONITORING_ENABLED=true
MONITORING_FAILURE_POLICY=fail_closed
MONITORING_RETENTION_DAYS=90
```

`fail_closed` returns HTTP 503 if a validated prediction cannot be stored. Use `warn` only when governance explicitly permits predictions without durable telemetry.

## Build the training baselines

The baseline command reads the cleaned Oracle `PATIENTS`, `VISITS`, and `BILLING` tables used by the notebooks, recreates the earliest 80% chronological training split, scores it with the deployed artifact, writes aggregate JSON, and registers it as the active Oracle baseline.

```bash
python monitoring/build_training_baseline.py --model all
```

Configure `TRAINING_ORACLE_USER`, `TRAINING_ORACLE_PASSWORD`, `TRAINING_ORACLE_DSN`, and `TRAINING_ORACLE_SCHEMA`. The training account needs read-only access to the three source tables. Local CSV mode is available only for development because local exports may not match the Oracle training snapshot:

```bash
python monitoring/build_training_baseline.py --model claim-v3 --source csv --no-store
```

Each baseline includes source-table fingerprints, artifact and category-contract hashes, training period, missing rates, numeric decile bins, categorical distributions, prediction distribution, confidence distribution, and low-confidence rate. No raw training records are stored in the baseline table.

## Calculate drift

Run for each model on an approved schedule, normally daily:

```bash
python monitoring/calculate_drift.py \
  --model claim-v3 \
  --lookback-hours 24 \
  --minimum-records 100 \
  --output metrics/monitoring/claim-drift.json
```

The job loads the active Oracle baseline, decrypts only telemetry matching its exact artifact hash, calculates drift, and stores the report in `ML_DRIFT_REPORT`.

Metrics include:

- Numeric PSI, missing-rate delta, and standardized mean shift.
- Categorical Jensen-Shannon divergence, missing-rate delta, and unseen-category rate.
- Prediction-class PSI and Jensen-Shannon divergence.
- Confidence PSI, mean-confidence change, and low-confidence rate.

Default warning/critical thresholds are included in every report. Exit code `2` means critical drift, `3` means insufficient records, and `1` means execution or Oracle failure.

## Retention and health

The Oracle scheduler deletes rows when `EXPIRES_AT <= SYSTIMESTAMP`. Authorized administrators can verify or run retention manually:

```bash
python monitoring/manage_monitoring_store.py health
python monitoring/manage_monitoring_store.py purge-expired
```
