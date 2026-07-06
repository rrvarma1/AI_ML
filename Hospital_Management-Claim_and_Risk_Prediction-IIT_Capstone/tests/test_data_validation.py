"""Tests for shared online and offline input quality monitoring."""

# Summary:
# 1. Purpose: Verifies shared live and offline model-input quality checks.
# 2. What it does: Tests missing data, ranges, categories, logs, columns, and JSONL reports.
# 3. Invoked by: Pytest during automated data-quality verification.
# 4. Main functions/classes: claim_context and data-validation test functions.
# 5. Validations/controls: Confirms strict modes fail and raw unseen values are never logged.

from __future__ import annotations

import json
import logging

import pandas as pd

from app.data_validation import DATA_QUALITY_LOGGER, log_validation_report, validate_records
from monitoring.validate_model_inputs import validate_input_file


def claim_context(client):
    from app.main import registry

    model = registry.models["claim-v3"]
    return registry, model


def test_valid_record_has_no_quality_issues(client, claim_payload) -> None:
    registry, _ = claim_context(client)
    report = registry.validate_inputs("claim-v3", [claim_payload])
    assert report.valid
    assert report.issues == []


def test_missing_value_is_reported_as_warning(client, claim_payload) -> None:
    _, model = claim_context(client)
    claim_payload["city"] = None
    report = validate_records(
        [claim_payload],
        model_id="claim-v3",
        category_contract=model.category_contract,
        missing_policy="warn",
    )
    issue = next(issue for issue in report.issues if issue.code == "missing_value")
    assert issue.feature == "city"
    assert issue.severity == "warning"
    assert report.valid


def test_missing_value_can_be_promoted_to_error(client, claim_payload) -> None:
    _, model = claim_context(client)
    claim_payload["city"] = None
    report = validate_records(
        [claim_payload],
        model_id="claim-v3",
        category_contract=model.category_contract,
        missing_policy="reject",
    )
    assert not report.valid
    assert any(issue.code == "missing_value" and issue.severity == "error" for issue in report.issues)


def test_numeric_range_and_type_violations_are_errors(client, claim_payload) -> None:
    _, model = claim_context(client)
    invalid_range = dict(claim_payload, age=121)
    invalid_type = dict(claim_payload, billed_amount="not-a-number")
    report = validate_records(
        [invalid_range, invalid_type],
        model_id="claim-v3",
        category_contract=model.category_contract,
    )
    assert not report.valid
    assert any(issue.code == "numeric_out_of_range" and issue.feature == "age" for issue in report.issues)
    assert any(
        issue.code == "invalid_numeric_type" and issue.feature == "billed_amount"
        for issue in report.issues
    )


def test_missing_and_unexpected_columns_are_errors(client, claim_payload) -> None:
    _, model = claim_context(client)
    del claim_payload["doctor_id"]
    claim_payload["unexpected"] = "value"
    report = validate_records(
        [claim_payload], model_id="claim-v3", category_contract=model.category_contract
    )
    assert not report.valid
    assert any(issue.code == "missing_column" and issue.feature == "doctor_id" for issue in report.issues)
    assert any(
        issue.code == "unexpected_column" and issue.feature == "unexpected"
        for issue in report.issues
    )


def test_non_finite_numeric_value_is_error(client, claim_payload) -> None:
    _, model = claim_context(client)
    claim_payload["billed_amount"] = float("inf")
    report = validate_records(
        [claim_payload], model_id="claim-v3", category_contract=model.category_contract
    )
    assert not report.valid
    assert any(
        issue.code == "non_finite_value" and issue.feature == "billed_amount"
        for issue in report.issues
    )


def test_unseen_operational_category_warns_without_value_disclosure(client, claim_payload) -> None:
    _, model = claim_context(client)
    unseen_value = "Confidential New City"
    claim_payload["city"] = unseen_value
    report = validate_records(
        [claim_payload],
        model_id="claim-v3",
        category_contract=model.category_contract,
        unseen_policy="warn",
    )
    issue = next(issue for issue in report.issues if issue.code == "unseen_category")
    assert issue.feature == "city"
    assert issue.severity == "warning"
    assert issue.distinct_count == 1
    assert unseen_value not in json.dumps(report.to_dict())


def test_controlled_or_strict_unseen_category_is_error(client, claim_payload) -> None:
    _, model = claim_context(client)
    controlled = dict(claim_payload, gender="X")
    open_category = dict(claim_payload, city="New City")
    controlled_report = validate_records(
        [controlled], model_id="claim-v3", category_contract=model.category_contract
    )
    strict_report = validate_records(
        [open_category],
        model_id="claim-v3",
        category_contract=model.category_contract,
        unseen_policy="reject",
    )
    assert not controlled_report.valid
    assert not strict_report.valid


def test_quality_log_is_json_and_excludes_unseen_value(client, claim_payload) -> None:
    _, model = claim_context(client)
    unseen_value = "Do Not Log This City"
    claim_payload["city"] = unseen_value
    report = validate_records(
        [claim_payload], model_id="claim-v3", category_contract=model.category_contract
    )
    records: list[logging.LogRecord] = []

    class CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = CaptureHandler()
    DATA_QUALITY_LOGGER.addHandler(handler)
    try:
        log_validation_report(report, "quality-test")
    finally:
        DATA_QUALITY_LOGGER.removeHandler(handler)
    event = json.loads(records[0].getMessage())
    assert event["event"] == "data_quality_validation"
    assert event["request_id"] == "quality-test"
    assert unseen_value not in records[0].getMessage()


def test_offline_jsonl_validator_uses_same_contract(client, claim_payload, tmp_path) -> None:
    _, model = claim_context(client)
    input_path = tmp_path / "claim-input.jsonl"
    output = pd.DataFrame([claim_payload])
    output.to_json(input_path, orient="records", lines=True)
    report, payload = validate_input_file(
        input_path,
        model_id="claim-v3",
        artifact_path=model.artifact_path,
    )
    assert report.valid
    assert payload["source"]["sha256"]
    assert payload["category_contract_sha256"] == model.category_contract_sha256
