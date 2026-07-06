#!/usr/bin/env python3
"""Validate raw model-input batches and write a machine-readable report."""

# Summary:
# 1. Purpose: Validates offline model-input files before prediction or ingestion.
# 2. What it does: Reads CSV, JSON, JSONL, or Parquet and writes a JSON quality report.
# 3. Invoked by: Operators, scheduled jobs, CI pipelines, and tests/test_data_validation.py.
# 4. Main functions/classes: read_input, validate_input_file, parse_args, and main.
# 5. Validations/controls: Supports warning or reject policies and returns automation-friendly exit codes.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data_validation import (  # noqa: E402
    CLAIM_MODEL_ID,
    RISK_MODEL_ID,
    extract_category_contract,
    file_sha256,
    validate_frame,
)


DEFAULT_ARTIFACTS = {
    CLAIM_MODEL_ID: Path("models/claim/claim_random_forest_hypertuned_v3.pkl"),
    RISK_MODEL_ID: Path("models/risk/risk_random_forest_smote_undersampling_v5.pkl"),
}


def read_input(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("Input format must be CSV, JSON, JSONL/NDJSON, or Parquet")


def validate_input_file(
    input_path: Path,
    *,
    model_id: str,
    artifact_path: Path,
    missing_policy: str = "warn",
    unseen_policy: str = "warn",
):
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError("Model artifact must be a dictionary containing 'model'")
    category_contract = extract_category_contract(artifact["model"])
    frame = read_input(input_path)
    report = validate_frame(
        frame,
        model_id=model_id,
        category_contract=category_contract,
        missing_policy=missing_policy,  # type: ignore[arg-type]
        unseen_policy=unseen_policy,  # type: ignore[arg-type]
        artifact_sha256=file_sha256(artifact_path),
    )
    payload = report.to_dict()
    payload["source"] = {
        "path": str(input_path.resolve()),
        "sha256": file_sha256(input_path),
        "format": input_path.suffix.lower().lstrip("."),
    }
    return report, payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate missing values, numeric ranges, and fitted-model categories."
    )
    parser.add_argument("input", type=Path, help="Raw model-input CSV, JSON(L), or Parquet file")
    parser.add_argument("--model", choices=[CLAIM_MODEL_ID, RISK_MODEL_ID], required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--artifact", type=Path, help="Override serialized model artifact path")
    parser.add_argument("--missing-policy", choices=["warn", "reject"], default="warn")
    parser.add_argument("--unseen-policy", choices=["warn", "reject"], default="warn")
    parser.add_argument("--fail-on-warnings", action="store_true")
    parser.add_argument("--output", type=Path, help="Write full JSON report to this path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    artifact_path = (
        args.artifact.expanduser().resolve()
        if args.artifact
        else (project_root / DEFAULT_ARTIFACTS[args.model]).resolve()
    )
    try:
        report, payload = validate_input_file(
            args.input.expanduser().resolve(),
            model_id=args.model,
            artifact_path=artifact_path,
            missing_policy=args.missing_policy,
            unseen_policy=args.unseen_policy,
        )
    except Exception as exc:
        print(f"validation execution failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    rendered = json.dumps(payload, indent=2, ensure_ascii=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"report={args.output} rows={report.row_count} errors={report.error_count} warnings={report.warning_count}")
    else:
        print(rendered)
    if not report.valid or (args.fail_on_warnings and report.warning_count):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
