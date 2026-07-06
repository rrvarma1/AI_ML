"""Thread-safe loading, validation, and inference for persisted pipelines."""

# Summary:
# 1. Purpose: Loads, verifies, stores, and runs the two serialized model pipelines.
# 2. What it does: Checks artifact contracts, extracts categories, and performs thread-safe inference.
# 3. Invoked by: app/main.py during startup, health checks, validation, and predictions.
# 4. Main functions/classes: LoadedModel, ModelRegistry, predict, validate_inputs, and _file_sha256.
# 5. Validations/controls: Verifies features, classes, artifact hashes, and fitted category contracts.

from __future__ import annotations

import hashlib
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import Settings
from .data_validation import (
    CLAIM_MODEL_ID,
    EXPECTED_CLAIM_COLUMNS,
    EXPECTED_RISK_COLUMNS,
    RISK_MODEL_ID,
    ValidationReport,
    category_contract_sha256,
    extract_category_contract,
    validate_records,
)
from .errors import ModelUnavailableError, PredictionError


LOGGER = logging.getLogger(__name__)

@dataclass
class LoadedModel:
    model_id: str
    model_name: str
    artifact_path: Path
    expected_columns: list[str]
    expected_classes: list[str]
    artifact: dict[str, Any] | None = None
    pipeline: Any | None = None
    classes: list[str] = field(default_factory=list)
    category_contract: dict[str, list[str]] = field(default_factory=dict)
    category_contract_sha256: str | None = None
    artifact_sha256: str | None = None
    error: str | None = None
    lock: RLock = field(default_factory=RLock)

    @property
    def ready(self) -> bool:
        return (
            self.pipeline is not None
            and self.artifact_sha256 is not None
            and self.category_contract_sha256 is not None
            and self.error is None
        )


class ModelRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.models = {
            CLAIM_MODEL_ID: LoadedModel(
                model_id=CLAIM_MODEL_ID,
                model_name="Model B V3 - Hypertuned Claim Random Forest",
                artifact_path=settings.claim_model_path,
                expected_columns=EXPECTED_CLAIM_COLUMNS,
                expected_classes=["Paid", "Pending", "Rejected"],
            ),
            RISK_MODEL_ID: LoadedModel(
                model_id=RISK_MODEL_ID,
                model_name="Model A V5 - Visit Risk with Random Undersampling",
                artifact_path=settings.risk_model_path,
                expected_columns=EXPECTED_RISK_COLUMNS,
                expected_classes=["High", "Low", "Medium"],
            ),
        }

    def load_all(self) -> None:
        project_root = str(self.settings.project_root)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        for loaded_model in self.models.values():
            self._load(loaded_model)

    def _load(self, loaded_model: LoadedModel) -> None:
        try:
            if not loaded_model.artifact_path.is_file():
                raise FileNotFoundError(f"Artifact not found: {loaded_model.artifact_path}")
            artifact_sha256 = _file_sha256(loaded_model.artifact_path)
            artifact = joblib.load(loaded_model.artifact_path)
            if not isinstance(artifact, dict) or "model" not in artifact:
                raise ValueError("Artifact must be a dictionary containing a 'model' pipeline")
            pipeline = artifact["model"]
            if not hasattr(pipeline, "predict") or not hasattr(pipeline, "predict_proba"):
                raise TypeError("Persisted model must provide predict and predict_proba")

            artifact_columns = artifact.get("feature_columns") or artifact.get("raw_feature_columns")
            if artifact_columns != loaded_model.expected_columns:
                raise ValueError(
                    f"Feature contract mismatch: expected {loaded_model.expected_columns}, got {artifact_columns}"
                )
            classes = [str(value) for value in pipeline.classes_]
            if classes != loaded_model.expected_classes:
                raise ValueError(
                    f"Class contract mismatch: expected {loaded_model.expected_classes}, got {classes}"
                )
            category_contract = extract_category_contract(pipeline)

            loaded_model.artifact = artifact
            loaded_model.pipeline = pipeline
            loaded_model.classes = classes
            loaded_model.category_contract = category_contract
            loaded_model.category_contract_sha256 = category_contract_sha256(category_contract)
            loaded_model.artifact_sha256 = artifact_sha256
            loaded_model.error = None
            LOGGER.info(
                "Loaded model %s artifact_sha256=%s from %s",
                loaded_model.model_id,
                artifact_sha256,
                loaded_model.artifact_path,
            )
        except Exception as exc:  # Readiness reports the sanitized failure.
            loaded_model.artifact = None
            loaded_model.pipeline = None
            loaded_model.classes = []
            loaded_model.category_contract = {}
            loaded_model.category_contract_sha256 = None
            loaded_model.artifact_sha256 = None
            loaded_model.error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("Failed to load model %s", loaded_model.model_id)

    def ready(self) -> bool:
        return all(model.ready for model in self.models.values())

    def health(self) -> dict[str, dict[str, Any]]:
        return {
            model_id: {
                "model_id": model.model_id,
                "ready": model.ready,
                "model_name": model.model_name,
                "artifact_path": str(model.artifact_path),
                "classes": model.classes,
                "artifact_sha256": model.artifact_sha256,
                "category_contract_sha256": model.category_contract_sha256,
                "categorical_feature_count": len(model.category_contract),
                "error": model.error,
            }
            for model_id, model in self.models.items()
        }

    def predict(self, model_id: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        loaded_model = self.models[model_id]
        if not loaded_model.ready:
            raise ModelUnavailableError(model_id)
        frame = pd.DataFrame.from_records(records, columns=loaded_model.expected_columns)
        try:
            with loaded_model.lock:
                predictions = loaded_model.pipeline.predict(frame)
                probabilities = loaded_model.pipeline.predict_proba(frame)
            return [
                {
                    "prediction": str(prediction),
                    "probabilities": {
                        label: float(np.clip(probability, 0.0, 1.0))
                        for label, probability in zip(loaded_model.classes, row)
                    },
                }
                for prediction, row in zip(predictions, probabilities)
            ]
        except ModelUnavailableError:
            raise
        except Exception as exc:
            LOGGER.exception("Inference failed for model %s", model_id)
            raise PredictionError(model_id) from exc

    def validate_inputs(self, model_id: str, records: list[dict[str, Any]]) -> ValidationReport:
        loaded_model = self.models[model_id]
        if not loaded_model.ready:
            raise ModelUnavailableError(model_id)
        return validate_records(
            records,
            model_id=model_id,
            category_contract=loaded_model.category_contract,
            missing_policy=self.settings.missing_value_policy,
            unseen_policy=self.settings.unseen_category_policy,
            artifact_sha256=loaded_model.artifact_sha256,
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
