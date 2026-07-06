#!/usr/bin/env python3
"""Calculate feature and prediction drift from encrypted Oracle telemetry."""

# Summary:
# 1. Purpose: Compares a production time window with the active Oracle training baseline.
# 2. Functions/validations: Decrypts authorized records, checks sample size, calculates drift, and stores the report.

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data_validation import CLAIM_MODEL_ID, RISK_MODEL_ID  # noqa: E402
from app.oracle_monitoring_store import OracleMonitoringStore, OracleStoreConfig  # noqa: E402
from monitoring.drift_metrics import calculate_drift_report  # noqa: E402


def parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("Timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calculate and store model drift metrics")
    parser.add_argument("--model", choices=[CLAIM_MODEL_ID, RISK_MODEL_ID], required=True)
    parser.add_argument("--start", type=parse_timestamp)
    parser.add_argument("--end", type=parse_timestamp)
    parser.add_argument("--lookback-hours", type=int, default=24)
    parser.add_argument("--minimum-records", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.lookback_hours < 1 or args.minimum_records < 1:
        parser.error("lookback-hours and minimum-records must be positive")
    end_at = args.end or datetime.now(timezone.utc)
    start_at = args.start or (end_at - timedelta(hours=args.lookback_hours))
    if start_at >= end_at:
        parser.error("start must be earlier than end")

    store = None
    try:
        store = OracleMonitoringStore.from_config(OracleStoreConfig.from_environment())
        baseline = store.load_active_baseline(args.model)
        records = store.fetch_predictions(
            model_id=args.model,
            artifact_sha256=baseline["artifact_sha256"],
            start_at=start_at,
            end_at=end_at,
        )
        if len(records) < args.minimum_records:
            report = {
                "schema_version": 1,
                "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "model_id": args.model,
                "artifact_sha256": baseline["artifact_sha256"],
                "window": {
                    "start": start_at.isoformat().replace("+00:00", "Z"),
                    "end": end_at.isoformat().replace("+00:00", "Z"),
                },
                "record_count": len(records),
                "minimum_records": args.minimum_records,
                "overall_status": "insufficient_data",
                "summary": {"message": "Not enough telemetry records for reliable drift metrics"},
            }
            exit_code = 3
        else:
            report = calculate_drift_report(
                baseline=baseline,
                telemetry_records=records,
                window_start=start_at,
                window_end=end_at,
            )
            exit_code = 2 if report["overall_status"] == "critical" else 0
        report_id = store.save_drift_report(report)
        rendered = json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(
            f"report_id={report_id} model={args.model} records={len(records)} "
            f"status={report['overall_status']}"
        )
        if not args.output:
            print(rendered)
        return exit_code
    except Exception as exc:
        print(f"drift calculation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if store:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
