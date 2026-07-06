"""Reproduce model training frames and create baseline profiles."""

# Summary:
# 1. Purpose: Rebuilds the chronological training population used by Claim V3 and Risk V5.
# 2. Functions/validations: Joins source tables, derives model fields, validates contracts, and profiles predictions.

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import oracledb
import pandas as pd

from app.data_validation import (
    CLAIM_MODEL_ID,
    EXPECTED_COLUMNS,
    RISK_MODEL_ID,
    category_contract_sha256,
    extract_category_contract,
    file_sha256,
    validate_frame,
)
from monitoring.drift_metrics import create_training_baseline


ORACLE_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_$#]{0,29}$")


@dataclass(frozen=True)
class OracleTrainingSourceConfig:
    user: str
    password: str = field(repr=False)
    dsn: str
    schema: str

    @classmethod
    def from_environment(cls) -> "OracleTrainingSourceConfig":
        names = ["TRAINING_ORACLE_USER", "TRAINING_ORACLE_PASSWORD", "TRAINING_ORACLE_DSN"]
        missing = [name for name in names if not os.getenv(name)]
        if missing:
            raise ValueError(f"Missing Oracle training source settings: {', '.join(missing)}")
        user = os.environ["TRAINING_ORACLE_USER"]
        schema = os.getenv("TRAINING_ORACLE_SCHEMA", user)
        if not ORACLE_IDENTIFIER.fullmatch(schema):
            raise ValueError("TRAINING_ORACLE_SCHEMA is not a valid Oracle identifier")
        return cls(
            user=user,
            password=os.environ["TRAINING_ORACLE_PASSWORD"],
            dsn=os.environ["TRAINING_ORACLE_DSN"],
            schema=schema.upper(),
        )


def load_oracle_training_tables(
    config: OracleTrainingSourceConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    connection = oracledb.connect(user=config.user, password=config.password, dsn=config.dsn)
    try:
        patients = _read_oracle_table(connection, config.schema, "PATIENTS")
        visits = _read_oracle_table(connection, config.schema, "VISITS")
        billing = _read_oracle_table(connection, config.schema, "BILLING")
    finally:
        connection.close()
    source = {
        "type": "oracle",
        "schema": config.schema,
        "extracted_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "tables": {
            "PATIENTS": {"rows": len(patients), "sha256": _dataframe_sha256(patients, "patient_id")},
            "VISITS": {"rows": len(visits), "sha256": _dataframe_sha256(visits, "visit_id")},
            "BILLING": {"rows": len(billing), "sha256": _dataframe_sha256(billing, "bill_id")},
        },
    }
    return patients, visits, billing, source


def load_csv_training_tables(
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw_dir = project_root / "data" / "raw"
    source_paths = {
        "patients.csv": raw_dir / "patients.csv",
        "visits.csv": raw_dir / "visits.csv",
        "billing.csv": raw_dir / "billing.csv",
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Training source file not found: {path}")
    patients = pd.read_csv(source_paths["patients.csv"])
    visits = pd.read_csv(source_paths["visits.csv"])
    billing = pd.read_csv(source_paths["billing.csv"])
    source = {
        "type": "csv-development-fallback",
        "files": {name: file_sha256(path) for name, path in source_paths.items()},
    }
    return patients, visits, billing, source


def build_training_frame_from_tables(
    patients: pd.DataFrame,
    visits: pd.DataFrame,
    billing: pd.DataFrame,
    *,
    model_id: str,
    source: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    patients = _normalize_columns(patients)
    visits = _normalize_columns(visits)
    billing = _normalize_columns(billing)
    patients["registration_date"] = pd.to_datetime(patients["registration_date"], errors="raise")
    visits["visit_date"] = pd.to_datetime(visits["visit_date"], errors="raise")
    billing["billing_date"] = pd.to_datetime(billing["billing_date"], errors="raise")

    if model_id == CLAIM_MODEL_ID:
        data = billing.merge(visits, on="visit_id", how="left").merge(
            patients, on="patient_id", how="left"
        )
        data["billing_month"] = data["billing_date"].dt.month
        data["billing_quarter"] = data["billing_date"].dt.quarter
        data["billing_day_of_week"] = data["billing_date"].dt.dayofweek
        data["billing_is_weekend"] = data["billing_day_of_week"].isin([5, 6]).astype(int)
        data["visit_month"] = data["visit_date"].dt.month
        data["visit_quarter"] = data["visit_date"].dt.quarter
        data["visit_day_of_week"] = data["visit_date"].dt.dayofweek
        data["visit_is_weekend"] = data["visit_day_of_week"].isin([5, 6]).astype(int)
        data["days_from_visit_to_billing"] = (
            data["billing_date"] - data["visit_date"]
        ).dt.days
        data["days_since_registration_at_visit"] = (
            data["visit_date"] - data["registration_date"]
        ).dt.days
        date_column = "billing_date"
        target_column = "claim_status"
    elif model_id == RISK_MODEL_ID:
        data = visits.merge(patients, on="patient_id", how="left").merge(
            billing, on="visit_id", how="left"
        )
        data["visit_year"] = data["visit_date"].dt.year
        data["visit_month"] = data["visit_date"].dt.month
        data["visit_quarter"] = data["visit_date"].dt.quarter
        data["visit_day_of_week"] = data["visit_date"].dt.dayofweek
        data["visit_is_weekend"] = data["visit_day_of_week"].isin([5, 6]).astype(int)
        data["days_since_registration_at_visit"] = (
            data["visit_date"] - data["registration_date"]
        ).dt.days
        date_column = "visit_date"
        target_column = "risk_score"
    else:
        raise ValueError(f"Unsupported model_id: {model_id}")

    model_data = data[EXPECTED_COLUMNS[model_id] + [target_column, date_column]].copy()
    model_data = model_data.dropna(subset=[target_column, date_column])
    model_data = model_data.sort_values(date_column).reset_index(drop=True)
    split_index = int(len(model_data) * 0.80)
    if split_index < 1:
        raise ValueError("Training split contains no records")
    train = model_data.iloc[:split_index].copy()
    metadata = {
        "target_column": target_column,
        "date_column": date_column,
        "period": {
            "start": train[date_column].min().isoformat(),
            "end": train[date_column].max().isoformat(),
        },
        "source": source,
        "target_distribution": {
            str(label): float(rate)
            for label, rate in train[target_column].value_counts(normalize=True).items()
        },
    }
    return train[EXPECTED_COLUMNS[model_id]], metadata


def build_baseline_from_artifact(
    *,
    model_id: str,
    artifact: dict[str, Any],
    artifact_sha256: str,
    training_frame: pd.DataFrame,
    training_metadata: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError("Artifact must be a dictionary containing 'model'")
    pipeline = artifact["model"]
    frame = training_frame
    metadata = training_metadata
    contract = extract_category_contract(pipeline)
    validation = validate_frame(
        frame,
        model_id=model_id,
        category_contract=contract,
        missing_policy="warn",
        unseen_policy="reject",
        artifact_sha256=artifact_sha256,
    )
    if not validation.valid:
        raise ValueError(f"Training baseline failed data validation: {validation.client_errors()}")
    predictions = [str(value) for value in pipeline.predict(frame)]
    probabilities = np.asarray(pipeline.predict_proba(frame), dtype=float)
    classes = [str(value) for value in pipeline.classes_]
    baseline = create_training_baseline(
        model_id=model_id,
        artifact_sha256=artifact_sha256,
        category_contract_sha256=category_contract_sha256(contract),
        training_frame=frame,
        predictions=predictions,
        probabilities=probabilities,
        classes=classes,
        training_period=metadata["period"],
        source_files=_source_fingerprints(metadata["source"]),
    )
    baseline["target_distribution"] = metadata["target_distribution"]
    baseline["source"] = metadata["source"]
    baseline["validation_summary"] = validation.to_dict()["summary"]
    return baseline


def _read_oracle_table(connection: Any, schema: str, table: str) -> pd.DataFrame:
    if not ORACLE_IDENTIFIER.fullmatch(schema) or not ORACLE_IDENTIFIER.fullmatch(table):
        raise ValueError("Unsafe Oracle schema or table identifier")
    with connection.cursor() as cursor:
        cursor.execute(f"SELECT * FROM {schema}.{table}")
        columns = [description[0].lower() for description in cursor.description]
        return pd.DataFrame(cursor.fetchall(), columns=columns)


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [str(column).lower() for column in normalized.columns]
    return normalized


def _dataframe_sha256(frame: pd.DataFrame, key_column: str) -> str:
    normalized = _normalize_columns(frame)
    if key_column in normalized.columns:
        normalized = normalized.sort_values(key_column).reset_index(drop=True)
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    row_hashes = pd.util.hash_pandas_object(normalized, index=False).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256()
    digest.update("|".join(normalized.columns).encode("utf-8"))
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def _source_fingerprints(source: dict[str, Any]) -> dict[str, str]:
    if source["type"] == "oracle":
        return {
            table: details["sha256"] for table, details in source["tables"].items()
        }
    return dict(source["files"])
