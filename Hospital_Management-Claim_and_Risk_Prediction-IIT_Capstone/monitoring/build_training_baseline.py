#!/usr/bin/env python3
"""Build and register chronological training baselines for deployed models."""

# Summary:
# 1. Purpose: Creates versioned Claim V3 and Risk V5 training baselines and stores them in Oracle.
# 2. Functions/validations: Rebuilds the 80% training split, validates it, predicts it, and saves aggregate JSON.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data_validation import CLAIM_MODEL_ID, RISK_MODEL_ID, file_sha256  # noqa: E402
from app.oracle_monitoring_store import (  # noqa: E402
    OracleMonitoringStore,
    OracleStoreConfig,
)
from monitoring.training_baseline import (  # noqa: E402
    OracleTrainingSourceConfig,
    build_baseline_from_artifact,
    build_training_frame_from_tables,
    load_csv_training_tables,
    load_oracle_training_tables,
)
from monitoring.validate_model_inputs import DEFAULT_ARTIFACTS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build training drift baselines")
    parser.add_argument(
        "--model", choices=[CLAIM_MODEL_ID, RISK_MODEL_ID, "all"], default="all"
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output-dir", type=Path, help="Default: <project>/models/monitoring"
    )
    parser.add_argument(
        "--no-store", action="store_true", help="Build local JSON without writing Oracle"
    )
    parser.add_argument(
        "--source",
        choices=["oracle", "csv"],
        default="oracle",
        help="Oracle is required for release baselines; CSV is a development fallback",
    )
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    output_dir = args.output_dir or (root / "models" / "monitoring")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_ids = [CLAIM_MODEL_ID, RISK_MODEL_ID] if args.model == "all" else [args.model]
    store = None
    try:
        if args.source == "oracle":
            patients, visits, billing, source = load_oracle_training_tables(
                OracleTrainingSourceConfig.from_environment()
            )
        else:
            patients, visits, billing, source = load_csv_training_tables(root)
        if not args.no_store:
            store = OracleMonitoringStore.from_config(OracleStoreConfig.from_environment())
            store.health_check()
        for model_id in model_ids:
            training_frame, training_metadata = build_training_frame_from_tables(
                patients,
                visits,
                billing,
                model_id=model_id,
                source=source,
            )
            artifact_path = root / DEFAULT_ARTIFACTS[model_id]
            artifact = joblib.load(artifact_path)
            baseline = build_baseline_from_artifact(
                model_id=model_id,
                artifact=artifact,
                artifact_sha256=file_sha256(artifact_path),
                training_frame=training_frame,
                training_metadata=training_metadata,
            )
            output_path = output_dir / f"{model_id}-training-baseline.json"
            output_path.write_text(
                json.dumps(baseline, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            baseline_id = store.save_baseline(baseline) if store else "not-stored"
            print(
                f"model={model_id} rows={baseline['record_count']} "
                f"baseline_id={baseline_id} output={output_path}"
            )
    except Exception as exc:
        print(f"baseline build failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if store:
            store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
