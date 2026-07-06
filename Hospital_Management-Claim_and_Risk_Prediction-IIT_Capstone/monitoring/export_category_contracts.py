#!/usr/bin/env python3
"""Export fitted encoder categories as versioned monitoring contracts."""

# Summary:
# 1. Purpose: Exports known categories from each fitted model encoder.
# 2. What it does: Writes versioned Claim V3 and Risk V5 category-contract JSON files.
# 3. Invoked by: Operators or release automation after an approved model release.
# 4. Main functions/classes: export_contract and main.
# 5. Validations/controls: Verifies artifact structure and records artifact and contract SHA-256 values.

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data_validation import (  # noqa: E402
    CLAIM_MODEL_ID,
    RISK_MODEL_ID,
    category_contract_sha256,
    extract_category_contract,
    file_sha256,
)
from monitoring.validate_model_inputs import DEFAULT_ARTIFACTS  # noqa: E402


def export_contract(model_id: str, artifact_path: Path, output_path: Path) -> dict:
    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError(f"Invalid model artifact: {artifact_path}")
    categories = extract_category_contract(artifact["model"])
    payload = {
        "model_id": model_id,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": file_sha256(artifact_path),
        "category_contract_sha256": category_contract_sha256(categories),
        "categories": categories,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Export categorical contracts from fitted models")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, help="Default: <project>/models/monitoring")
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    output_dir = args.output_dir or (root / "models" / "monitoring")
    try:
        for model_id in (CLAIM_MODEL_ID, RISK_MODEL_ID):
            artifact_path = root / DEFAULT_ARTIFACTS[model_id]
            output_path = output_dir / f"{model_id}-category-contract.json"
            payload = export_contract(model_id, artifact_path, output_path)
            print(
                f"model={model_id} features={len(payload['categories'])} "
                f"contract_sha256={payload['category_contract_sha256']} output={output_path}"
            )
    except Exception as exc:
        print(f"category export failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
