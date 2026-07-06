#!/usr/bin/env python3
"""Generate deterministic synthetic API batch requests for drift testing."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CITIES = ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Mumbai", "Pune"]
INSURERS = ["CareOne", "HealthPlus", "MediCareX", "SecureLife"]
DEPARTMENTS = ["Cardiology", "ER", "General", "ICU", "Neurology", "Orthopedics"]
VISIT_TYPES = ["ER", "ICU", "OPD"]


def _calendar_fields(rng: random.Random, prefix: str) -> dict[str, int]:
    month = rng.randint(1, 12)
    day_of_week = rng.randint(0, 6)
    return {
        f"{prefix}_month": month,
        f"{prefix}_quarter": ((month - 1) // 3) + 1,
        f"{prefix}_day_of_week": day_of_week,
        f"{prefix}_is_weekend": int(day_of_week in (5, 6)),
    }


def _common_fields(rng: random.Random) -> dict[str, Any]:
    return {
        "age": round(rng.triangular(18, 90, 45)),
        "chronic_flag": rng.choices([0, 1], weights=[0.58, 0.42], k=1)[0],
        "length_of_stay_hours": round(min(rng.gammavariate(2.3, 8.0), 120.0), 2),
        "billed_amount": round(min(rng.lognormvariate(9.7, 0.55), 100_000.0), 2),
        "days_since_registration_at_visit": rng.randint(5, 730),
        "gender": rng.choice(["F", "M"]),
        "city": rng.choice(CITIES),
        "insurance_provider": rng.choice(INSURERS),
        "department": rng.choice(DEPARTMENTS),
        "visit_type": rng.choice(VISIT_TYPES),
        "doctor_id": rng.randint(100, 200),
    }


def generate_claim_records(count: int, rng: random.Random) -> list[dict[str, Any]]:
    records = []
    for _ in range(count):
        record = _common_fields(rng)
        record.update(
            {
                "days_from_visit_to_billing": rng.randint(0, 120),
                "risk_score": rng.choices(
                    ["Low", "Medium", "High"], weights=[0.45, 0.35, 0.20], k=1
                )[0],
            }
        )
        record.update(_calendar_fields(rng, "billing"))
        record.update(_calendar_fields(rng, "visit"))
        records.append(record)
    return records


def generate_risk_records(count: int, rng: random.Random) -> list[dict[str, Any]]:
    records = []
    for _ in range(count):
        record = _common_fields(rng)
        claim_status = rng.choices(
            ["Paid", "Pending", "Rejected"], weights=[0.45, 0.25, 0.30], k=1
        )[0]
        billed_amount = float(record["billed_amount"])
        if claim_status == "Paid":
            approved_amount = billed_amount * rng.uniform(0.70, 1.0)
        elif claim_status == "Pending":
            approved_amount = billed_amount * rng.uniform(0.0, 0.60)
        else:
            approved_amount = 0.0
        record.update(
            {
                "approved_amount": round(approved_amount, 2),
                "payment_days": round(rng.uniform(0, 90), 1),
                "visit_year": rng.choice([2025, 2026]),
                "claim_status": claim_status,
            }
        )
        record.update(_calendar_fields(rng, "visit"))
        records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "data" / "synthetic"
    )
    args = parser.parse_args()
    if args.count < 1 or args.count > 10_000:
        parser.error("count must be between 1 and 10000")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    claim_payload = {"instances": generate_claim_records(args.count, random.Random(args.seed))}
    risk_payload = {"instances": generate_risk_records(args.count, random.Random(args.seed + 1))}
    outputs = {
        output_dir / f"claim-v3-batch-{args.count}.json": claim_payload,
        output_dir / f"risk-v5-batch-{args.count}.json": risk_payload,
    }
    for path, payload in outputs.items():
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"created={path} records={len(payload['instances'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
