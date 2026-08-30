from pathlib import Path

import agent2
from agent2 import RiskSummaryAgent
from extractor import DealTerms


def _greenleaf_terms(**updates) -> DealTerms:
    values = {
        "estimated_dscr": 1.42,
        "debt_to_equity": 1.80,
        "current_ratio": 1.35,
        "revenue_growth_pct": 12.5,
        "loan_amount": 50_000_000,
        "collateral_market_value": 72_000_000,
        "collateral_realisable_value": 57_500_000,
    }
    values.update(updates)
    return DealTerms.model_validate(values)


def test_agent2_calculates_metrics_and_low_risk_result():
    result = RiskSummaryAgent(Path("risk_rules.json")).assess(
        _greenleaf_terms(), use_llm_summary=False
    )
    assert result.overall_risk == "LOW"
    assert result.recommendation == "PROCEED"
    assert result.risk_flags == []
    assert result.calculated_metrics["calculated_ltv_pct"] == 69.44
    assert result.calculated_metrics["realisable_collateral_coverage"] == 1.15
    assert result.calculated_metrics["collateral_valuation_haircut_pct"] == 20.14


def test_multiple_high_risks_are_critical():
    result = RiskSummaryAgent(Path("risk_rules.json")).assess(
        _greenleaf_terms(estimated_dscr=1.10, current_ratio=1.0),
        use_llm_summary=False,
    )
    assert result.overall_risk == "CRITICAL"
    assert result.recommendation == "DECLINE"
    assert {flag.risk_id for flag in result.risk_flags} >= {"RSK-001", "RSK-003"}


def test_severe_collateral_shortfall_is_critical():
    result = RiskSummaryAgent(Path("risk_rules.json")).assess(
        _greenleaf_terms(collateral_realisable_value=40_000_000),
        use_llm_summary=False,
    )
    coverage = next(flag for flag in result.risk_flags if flag.risk_id == "RSK-006")
    assert coverage.severity == "CRITICAL"
    assert result.overall_risk == "CRITICAL"


def test_missing_critical_inputs_returns_unable_to_assess():
    result = RiskSummaryAgent(Path("risk_rules.json")).assess(
        DealTerms(revenue_growth_pct=12.5), use_llm_summary=False
    )
    assert result.overall_risk == "UNABLE_TO_ASSESS"
    assert result.recommendation == "INSUFFICIENT_INFORMATION"
    assert "estimated_dscr" in result.missing_information


def test_gemma_is_used_only_for_the_grounded_narrative(monkeypatch):
    captured = {}

    def grounded_summary(evidence, model_name):
        captured["evidence"] = evidence
        captured["model_name"] = model_name
        return "The configured evidence indicates low risk. PROCEED is a review recommendation, not a lending decision."

    monkeypatch.setattr(agent2, "_gemma_summary", grounded_summary)
    result = RiskSummaryAgent(Path("risk_rules.json")).assess(
        _greenleaf_terms(), use_llm_summary=True
    )
    assert captured["model_name"] == "gemma3:4b"
    assert captured["evidence"]["overall_risk"] == "LOW"
    assert result.summary_model == "gemma3:4b"
    assert result.overall_risk == "LOW"
