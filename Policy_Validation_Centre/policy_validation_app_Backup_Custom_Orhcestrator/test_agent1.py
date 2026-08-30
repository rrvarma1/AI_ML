import json
from pathlib import Path

from agent1 import ComplianceValidationAgent


def _handoff(tmp_path: Path, terms: dict) -> Path:
    path = tmp_path / "extractor_output.json"
    path.write_text(json.dumps({"extracted_terms": terms}), encoding="utf-8")
    return path


def test_agent1_evaluates_greenleaf_thresholds(tmp_path):
    terms = {
        "deal_type": "New secured term loan", "currency": "INR",
        "purpose": "Purchase of a biodegradable packaging production line",
        "industry": "Sustainable packaging manufacturing",
        "collateral_type": "First charge over manufacturing plant and machinery",
        "loan_amount": 50_000_000, "term_years": 5, "term_months": 60,
        "interest_rate_pct": 10.25, "estimated_dscr": 1.42, "current_ratio": 1.35,
        "debt_to_equity": 1.80,
        "revenue_growth_pct": 12.5, "collateral_market_value": 72_000_000,
        "collateral_realisable_value": 57_500_000, "stated_collateral_coverage": 1.15,
        "total_facility_amount": 50_000_000
    }
    result = ComplianceValidationAgent(Path("rules.json")).validate_handoff(_handoff(tmp_path, terms))
    assert result.passed == 17
    assert result.failed == 0
    assert result.overall_status == "COMPLIANT"
    assert next(f for f in result.findings if f.rule_id == "FIN-008").status == "PASS"
    assert next(f for f in result.findings if f.rule_id == "DEAL-001").status == "PASS"


def test_deal_type_still_rejects_unsecured_term_loan(tmp_path):
    result = ComplianceValidationAgent(Path("rules.json")).validate_handoff(
        _handoff(tmp_path, {"deal_type": "New unsecured term loan"})
    )
    finding = next(f for f in result.findings if f.rule_id == "DEAL-001")
    assert finding.status == "FAIL"


def test_missing_value_is_insufficient_data(tmp_path):
    result = ComplianceValidationAgent(Path("rules.json")).validate_handoff(
        _handoff(tmp_path, {"currency": "INR"})
    )
    assert result.insufficient_data == 16
    assert result.overall_status == "INSUFFICIENT_DATA"


def test_threshold_is_strict(tmp_path):
    result = ComplianceValidationAgent(Path("rules.json")).validate_handoff(
        _handoff(tmp_path, {"interest_rate_pct": 10.75})
    )
    finding = next(f for f in result.findings if f.rule_id == "FIN-004")
    assert finding.status == "FAIL"


def test_agent1_rejects_leverage_and_liquidity_breaches(tmp_path):
    result = ComplianceValidationAgent(Path("rules.json")).validate_handoff(
        _handoff(tmp_path, {"debt_to_equity": 2.80, "current_ratio": 1.05})
    )
    debt_to_equity = next(f for f in result.findings if f.rule_id == "FIN-009")
    current_ratio = next(f for f in result.findings if f.rule_id == "FIN-006")
    assert debt_to_equity.status == "FAIL"
    assert debt_to_equity.threshold == 2.0
    assert current_ratio.status == "FAIL"
    assert current_ratio.threshold == 1.25
    assert result.overall_status == "NON_COMPLIANT"
