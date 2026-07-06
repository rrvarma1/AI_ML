#!/usr/bin/env python3
"""Administrative health and retention commands for the Oracle monitoring store."""

# Summary:
# 1. Purpose: Lets authorized operators verify Oracle monitoring access and purge expired telemetry.
# 2. Functions/validations: Uses encrypted-store settings, health checks the schema, and deletes only expired rows.

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.oracle_monitoring_store import OracleMonitoringStore, OracleStoreConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage Oracle model-monitoring storage")
    parser.add_argument("command", choices=["health", "purge-expired"])
    args = parser.parse_args()
    store = None
    try:
        store = OracleMonitoringStore.from_config(OracleStoreConfig.from_environment())
        if args.command == "health":
            store.health_check()
            print("monitoring_store=ready")
        else:
            print(f"expired_rows_deleted={store.purge_expired()}")
        return 0
    except Exception as exc:
        print(f"monitoring command failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if store:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
