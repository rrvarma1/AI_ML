"""Encrypted Oracle persistence for model monitoring records and metrics."""

# Summary:
# 1. Purpose: Stores validated features, baselines, and drift reports in the Oracle monitoring schema.
# 2. Functions/validations: Encrypts records, validates keys/schema names, enforces expiry, and supports key rotation.

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from uuid import uuid4

import oracledb
from cryptography.fernet import Fernet, InvalidToken


ORACLE_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_$#]{0,29}$")


class MonitoringStoreError(RuntimeError):
    """Raised when encrypted monitoring persistence cannot be completed safely."""


@dataclass(frozen=True)
class OracleStoreConfig:
    user: str
    password: str = field(repr=False)
    dsn: str
    schema: str | None
    encryption_key: bytes = field(repr=False)
    decryption_keys: tuple[bytes, ...] = field(default=(), repr=False)
    retention_days: int = 90
    pool_min: int = 1
    pool_max: int = 5
    pool_increment: int = 1

    @classmethod
    def from_environment(cls) -> "OracleStoreConfig":
        required_names = [
            "MONITORING_ORACLE_USER",
            "MONITORING_ORACLE_PASSWORD",
            "MONITORING_ORACLE_DSN",
            "MONITORING_ENCRYPTION_KEY",
        ]
        missing = [name for name in required_names if not os.getenv(name)]
        if missing:
            raise ValueError(f"Missing monitoring settings: {', '.join(missing)}")
        schema = os.getenv("MONITORING_ORACLE_SCHEMA") or None
        if schema and not ORACLE_IDENTIFIER.fullmatch(schema):
            raise ValueError("MONITORING_ORACLE_SCHEMA is not a valid Oracle identifier")
        primary_key = os.environ["MONITORING_ENCRYPTION_KEY"].encode("ascii")
        _validate_fernet_key(primary_key, "MONITORING_ENCRYPTION_KEY")
        old_keys = tuple(
            value.strip().encode("ascii")
            for value in os.getenv("MONITORING_DECRYPTION_KEYS", "").split(",")
            if value.strip()
        )
        for key in old_keys:
            _validate_fernet_key(key, "MONITORING_DECRYPTION_KEYS")
        retention_days = int(os.getenv("MONITORING_RETENTION_DAYS", "90"))
        if retention_days < 1 or retention_days > 3650:
            raise ValueError("MONITORING_RETENTION_DAYS must be between 1 and 3650")
        pool_min = int(os.getenv("MONITORING_POOL_MIN", "1"))
        pool_max = int(os.getenv("MONITORING_POOL_MAX", "5"))
        pool_increment = int(os.getenv("MONITORING_POOL_INCREMENT", "1"))
        if pool_min < 0 or pool_max < 1 or pool_min > pool_max or pool_increment < 1:
            raise ValueError("Monitoring Oracle pool settings are invalid")
        return cls(
            user=os.environ["MONITORING_ORACLE_USER"],
            password=os.environ["MONITORING_ORACLE_PASSWORD"],
            dsn=os.environ["MONITORING_ORACLE_DSN"],
            schema=schema,
            encryption_key=primary_key,
            decryption_keys=old_keys,
            retention_days=retention_days,
            pool_min=pool_min,
            pool_max=pool_max,
            pool_increment=pool_increment,
        )


class OracleMonitoringStore:
    def __init__(
        self,
        *,
        connection_factory: Callable[[], Any],
        encryption_key: bytes,
        decryption_keys: tuple[bytes, ...] = (),
        retention_days: int = 90,
        schema: str | None = None,
        pool: Any | None = None,
    ) -> None:
        if retention_days < 1 or retention_days > 3650:
            raise ValueError("retention_days must be between 1 and 3650")
        if schema and not ORACLE_IDENTIFIER.fullmatch(schema):
            raise ValueError("schema is not a valid Oracle identifier")
        self.connection_factory = connection_factory
        self.retention_days = retention_days
        self.schema = schema.upper() if schema else None
        self.pool = pool
        keys = (encryption_key, *decryption_keys)
        self._fernets = {_key_id(key): Fernet(key) for key in keys}
        self._primary_key_id = _key_id(encryption_key)

    @classmethod
    def from_config(cls, config: OracleStoreConfig) -> "OracleMonitoringStore":
        pool = oracledb.create_pool(
            user=config.user,
            password=config.password,
            dsn=config.dsn,
            min=config.pool_min,
            max=config.pool_max,
            increment=config.pool_increment,
        )
        return cls(
            connection_factory=pool.acquire,
            encryption_key=config.encryption_key,
            decryption_keys=config.decryption_keys,
            retention_days=config.retention_days,
            schema=config.schema,
            pool=pool,
        )

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close(force=True)

    def health_check(self) -> None:
        try:
            with self.connection_factory() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(f"SELECT 1 FROM {self._table('ML_PREDICTION_TELEMETRY')} WHERE 1 = 0")
        except Exception as exc:
            raise MonitoringStoreError("Oracle monitoring store health check failed") from exc

    def store_predictions(
        self,
        *,
        model_id: str,
        artifact_sha256: str,
        request_id: str,
        recorded_at: datetime,
        records: list[dict[str, Any]],
        results: list[dict[str, Any]],
        input_hashes: list[str],
    ) -> int:
        if not (len(records) == len(results) == len(input_hashes)):
            raise ValueError("records, results, and input_hashes must have the same length")
        timestamp = _as_utc(recorded_at)
        expires_at = timestamp + timedelta(days=self.retention_days)
        rows = []
        for index, (features, result, feature_hash) in enumerate(
            zip(records, results, input_hashes)
        ):
            payload = {
                "schema_version": 1,
                "request_id": request_id,
                "record_index": index,
                "input_feature_hash": feature_hash,
                "features": features,
                "prediction": result["prediction"],
                "probabilities": result["probabilities"],
                "max_probability": max(result["probabilities"].values()),
            }
            ciphertext = self._encrypt(payload)
            rows.append(
                {
                    "telemetry_id": str(uuid4()),
                    "model_id": model_id,
                    "artifact_sha256": artifact_sha256,
                    "recorded_at": timestamp,
                    "expires_at": expires_at,
                    "key_id": self._primary_key_id,
                    "encrypted_payload": ciphertext,
                    "payload_sha256": hashlib.sha256(ciphertext).hexdigest(),
                }
            )
        sql = f"""
            INSERT INTO {self._table('ML_PREDICTION_TELEMETRY')} (
                telemetry_id, model_id, artifact_sha256, recorded_at, expires_at,
                key_id, encrypted_payload, payload_sha256
            ) VALUES (
                :telemetry_id, :model_id, :artifact_sha256, :recorded_at, :expires_at,
                :key_id, :encrypted_payload, :payload_sha256
            )
        """
        self._execute_many(sql, rows)
        return len(rows)

    def fetch_predictions(
        self,
        *,
        model_id: str,
        artifact_sha256: str,
        start_at: datetime,
        end_at: datetime,
        limit: int = 1_000_000,
    ) -> list[dict[str, Any]]:
        if limit < 1 or limit > 1_000_000:
            raise ValueError("limit must be between 1 and 1000000")
        sql = f"""
            SELECT telemetry_id, recorded_at, key_id, encrypted_payload, payload_sha256
            FROM (
                SELECT telemetry_id, recorded_at, key_id, encrypted_payload, payload_sha256
                FROM {self._table('ML_PREDICTION_TELEMETRY')}
                WHERE model_id = :model_id
                  AND artifact_sha256 = :artifact_sha256
                  AND recorded_at >= :start_at
                  AND recorded_at < :end_at
                  AND expires_at > SYSTIMESTAMP
                ORDER BY recorded_at, telemetry_id
            )
            WHERE ROWNUM <= :row_limit
        """
        binds = {
            "model_id": model_id,
            "artifact_sha256": artifact_sha256,
            "start_at": _as_utc(start_at),
            "end_at": _as_utc(end_at),
            "row_limit": limit,
        }
        try:
            with self.connection_factory() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(sql, binds)
                    rows = cursor.fetchall()
        except Exception as exc:
            raise MonitoringStoreError("Failed to read encrypted Oracle telemetry") from exc
        output = []
        for telemetry_id, recorded_at, key_id, encrypted_payload, payload_sha256 in rows:
            ciphertext = _lob_bytes(encrypted_payload)
            if hashlib.sha256(ciphertext).hexdigest() != payload_sha256:
                raise MonitoringStoreError(f"Telemetry integrity check failed: {telemetry_id}")
            payload = self._decrypt(key_id, ciphertext)
            payload["telemetry_id"] = telemetry_id
            #payload["recorded_at"] = _as_utc(recorded_at).isoformat().replace("+00:00", "Z")
            payload["recorded_at"] = (
    _oracle_timestamp_as_utc(recorded_at)
    .isoformat()
    .replace("+00:00", "Z")
)
            output.append(payload)
        return output

    def save_baseline(self, baseline: dict[str, Any]) -> str:
        baseline_id = str(uuid4())
        model_id = str(baseline["model_id"])
        artifact_sha256 = str(baseline["artifact_sha256"])
        baseline_json = json.dumps(baseline, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        try:
            with self.connection_factory() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        f"UPDATE {self._table('ML_MONITOR_BASELINE')} "
                        "SET is_active = 'N' WHERE model_id = :model_id AND is_active = 'Y'",
                        {"model_id": model_id},
                    )
                    cursor.execute(
                        f"""
                        INSERT INTO {self._table('ML_MONITOR_BASELINE')} (
                            baseline_id, model_id, artifact_sha256, baseline_json, is_active
                        ) VALUES (:baseline_id, :model_id, :artifact_sha256, :baseline_json, 'Y')
                        """,
                        {
                            "baseline_id": baseline_id,
                            "model_id": model_id,
                            "artifact_sha256": artifact_sha256,
                            "baseline_json": baseline_json,
                        },
                    )
                connection.commit()
        except Exception as exc:
            raise MonitoringStoreError("Failed to save Oracle training baseline") from exc
        return baseline_id

    def load_active_baseline(self, model_id: str) -> dict[str, Any]:
        sql = f"""
            SELECT baseline_json
            FROM {self._table('ML_MONITOR_BASELINE')}
            WHERE model_id = :model_id AND is_active = 'Y'
            ORDER BY created_at DESC
            FETCH FIRST 1 ROWS ONLY
        """
        try:
            with self.connection_factory() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(sql, {"model_id": model_id})
                    row = cursor.fetchone()
        except Exception as exc:
            raise MonitoringStoreError("Failed to load Oracle training baseline") from exc
        if row is None:
            raise MonitoringStoreError(f"No active baseline found for model '{model_id}'")
        return json.loads(_lob_text(row[0]))

    def save_drift_report(self, report: dict[str, Any]) -> str:
        report_id = str(uuid4())
        report_json = json.dumps(report, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        try:
            with self.connection_factory() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        f"""
                        INSERT INTO {self._table('ML_DRIFT_REPORT')} (
                            report_id, model_id, artifact_sha256, window_start,
                            window_end, record_count, overall_status, report_json
                        ) VALUES (
                            :report_id, :model_id, :artifact_sha256, :window_start,
                            :window_end, :record_count, :overall_status, :report_json
                        )
                        """,
                        {
                            "report_id": report_id,
                            "model_id": report["model_id"],
                            "artifact_sha256": report["artifact_sha256"],
                            "window_start": _parse_timestamp(report["window"]["start"]),
                            "window_end": _parse_timestamp(report["window"]["end"]),
                            "record_count": report["record_count"],
                            "overall_status": report["overall_status"],
                            "report_json": report_json,
                        },
                    )
                connection.commit()
        except Exception as exc:
            raise MonitoringStoreError("Failed to save Oracle drift report") from exc
        return report_id

    def purge_expired(self) -> int:
        try:
            with self.connection_factory() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        f"DELETE FROM {self._table('ML_PREDICTION_TELEMETRY')} "
                        "WHERE expires_at <= SYSTIMESTAMP"
                    )
                    deleted = int(cursor.rowcount or 0)
                connection.commit()
        except Exception as exc:
            raise MonitoringStoreError("Failed to purge expired Oracle telemetry") from exc
        return deleted

    def _execute_many(self, sql: str, rows: list[dict[str, Any]]) -> None:
        try:
            with self.connection_factory() as connection:
                with connection.cursor() as cursor:
                    cursor.setinputsizes(encrypted_payload=oracledb.DB_TYPE_BLOB)
                    cursor.executemany(sql, rows)
                connection.commit()
        except Exception as exc:
            raise MonitoringStoreError("Failed to write encrypted Oracle telemetry") from exc

    def _encrypt(self, payload: dict[str, Any]) -> bytes:
        plaintext = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("utf-8")
        return self._fernets[self._primary_key_id].encrypt(plaintext)

    def _decrypt(self, key_id: str, ciphertext: bytes) -> dict[str, Any]:
        if key_id not in self._fernets:
            raise MonitoringStoreError(f"No decryption key is configured for key_id '{key_id}'")
        try:
            plaintext = self._fernets[key_id].decrypt(ciphertext)
        except InvalidToken as exc:
            raise MonitoringStoreError("Encrypted telemetry authentication failed") from exc
        return json.loads(plaintext)

    def _table(self, table_name: str) -> str:
        return f"{self.schema}.{table_name}" if self.schema else table_name


def _validate_fernet_key(key: bytes, setting_name: str) -> None:
    try:
        Fernet(key)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{setting_name} must be a valid Fernet key") from exc


def _key_id(key: bytes) -> str:
    _validate_fernet_key(key, "encryption key")
    return hashlib.sha256(key).hexdigest()[:16]

def _oracle_timestamp_as_utc(value: datetime) -> datetime:
    """Normalize timestamps read from Oracle to timezone-aware UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)

def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("Monitoring timestamps must include a timezone")
    return value.astimezone(timezone.utc)


def _parse_timestamp(value: str) -> datetime:
    return _as_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _lob_bytes(value: Any) -> bytes:
    raw = value.read() if hasattr(value, "read") else value
    return bytes(raw)


def _lob_text(value: Any) -> str:
    raw = value.read() if hasattr(value, "read") else value
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
