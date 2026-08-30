"""Agent 2: deterministic risk flags with an evidence-bounded Gemma summary."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from extractor import DealTerms


Severity = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL", "UNABLE_TO_ASSESS"]
Recommendation = Literal[
    "PROCEED", "PROCEED_WITH_CONDITIONS", "MANUAL_REVIEW", "DECLINE", "INSUFFICIENT_INFORMATION"
]


class RiskRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_id: str = Field(pattern=r"^RSK-\d{3}$")
    category: str
    title: str
    metric: str
    operator: Literal["lt", "lte", "gt", "gte"]
    threshold: float
    severity: Severity
    critical_threshold: float | None = None
    required: bool = False
    message: str
    clear_message: str
    recommended_action: str


class RiskRuleCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    rules: list[RiskRule]


class RiskFlag(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_id: str
    category: str
    title: str
    severity: Severity
    status: Literal["OPEN"] = "OPEN"
    original_values: dict[str, Any]
    calculated_value: float | int
    threshold_value: float
    explanation: str
    recommended_action: str


class Agent2Result(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_name: Literal["agent_2_risk_and_summary"] = "agent_2_risk_and_summary"
    risk_rule_catalog_version: str
    summary_model: str
    overall_risk: RiskLevel
    recommendation: Recommendation
    risk_flags: list[RiskFlag]
    positive_factors: list[str]
    missing_information: list[str]
    calculated_metrics: dict[str, float | int | None]
    summary: str = Field(min_length=20, max_length=1600)
    confidence: float = Field(ge=0, le=1)


def load_risk_rules(path: Path) -> RiskRuleCatalog:
    if not path.is_file():
        raise FileNotFoundError(f"Risk rule catalog not found: {path}")
    catalog = RiskRuleCatalog.model_validate_json(path.read_text(encoding="utf-8"))
    ids = [rule.risk_id for rule in catalog.rules]
    if len(ids) != len(set(ids)):
        raise ValueError("risk_rules.json contains duplicate risk_id values")
    return catalog


def _evaluate(actual: float | int, operator: str, threshold: float) -> bool:
    operations = {
        "lt": lambda a, t: a < t,
        "lte": lambda a, t: a <= t,
        "gt": lambda a, t: a > t,
        "gte": lambda a, t: a >= t,
    }
    try:
        return bool(operations[operator](float(actual), float(threshold)))
    except KeyError as exc:
        raise ValueError(f"Unsupported risk operator: {operator}") from exc


def _round(value: float) -> float:
    return round(value, 2)


def _metrics(terms: DealTerms) -> tuple[dict[str, float | int | None], dict[str, dict[str, Any]]]:
    values: dict[str, float | int | None] = {
        "estimated_dscr": terms.estimated_dscr,
        "debt_to_equity": terms.debt_to_equity,
        "current_ratio": terms.current_ratio,
        "revenue_growth_pct": terms.revenue_growth_pct,
        "calculated_ltv_pct": None,
        "realisable_collateral_coverage": None,
        "collateral_valuation_haircut_pct": None,
    }
    inputs: dict[str, dict[str, Any]] = {
        "estimated_dscr": {"estimated_dscr": terms.estimated_dscr},
        "debt_to_equity": {"debt_to_equity": terms.debt_to_equity},
        "current_ratio": {"current_ratio": terms.current_ratio},
        "revenue_growth_pct": {"revenue_growth_pct": terms.revenue_growth_pct},
        "calculated_ltv_pct": {
            "loan_amount": terms.loan_amount,
            "collateral_market_value": terms.collateral_market_value,
        },
        "realisable_collateral_coverage": {
            "collateral_realisable_value": terms.collateral_realisable_value,
            "loan_amount": terms.loan_amount,
        },
        "collateral_valuation_haircut_pct": {
            "collateral_market_value": terms.collateral_market_value,
            "collateral_realisable_value": terms.collateral_realisable_value,
        },
    }
    if terms.loan_amount is not None and terms.collateral_market_value not in (None, 0):
        values["calculated_ltv_pct"] = _round(100 * terms.loan_amount / terms.collateral_market_value)
    if terms.collateral_realisable_value is not None and terms.loan_amount not in (None, 0):
        values["realisable_collateral_coverage"] = _round(terms.collateral_realisable_value / terms.loan_amount)
    if terms.collateral_market_value not in (None, 0) and terms.collateral_realisable_value is not None:
        values["collateral_valuation_haircut_pct"] = _round(
            100 * (terms.collateral_market_value - terms.collateral_realisable_value) / terms.collateral_market_value
        )
    return values, inputs


def _aggregate(flags: list[RiskFlag], required_missing: list[str]) -> tuple[RiskLevel, Recommendation]:
    critical = sum(flag.severity == "CRITICAL" for flag in flags)
    high = sum(flag.severity == "HIGH" for flag in flags)
    medium = sum(flag.severity == "MEDIUM" for flag in flags)
    if critical or high >= 2:
        return "CRITICAL", "DECLINE"
    if required_missing:
        return "UNABLE_TO_ASSESS", "INSUFFICIENT_INFORMATION"
    if high:
        return "HIGH", "MANUAL_REVIEW"
    if medium:
        return "MEDIUM", "PROCEED_WITH_CONDITIONS"
    return "LOW", "PROCEED"


def _deterministic_summary(
    overall_risk: RiskLevel,
    recommendation: Recommendation,
    flags: list[RiskFlag],
    positives: list[str],
    missing: list[str],
) -> str:
    if missing:
        return (
            f"The deal is unable to be fully assessed because required information is missing: {', '.join(missing)}. "
            f"The system recommendation is {recommendation}."
        )
    if flags:
        titles = "; ".join(flag.title for flag in flags)
        return (
            f"The deal has an overall {overall_risk.lower()} risk assessment. Flagged risks are: {titles}. "
            f"The system recommendation is {recommendation}. This is a review recommendation, not a lending decision."
        )
    return (
        f"The deal has an overall low risk assessment across the configured risk rules. "
        f"The system recommendation is {recommendation}. This is a review recommendation, not a lending decision."
    )


def _numeric_tokens(value: object) -> set[str]:
    tokens = set()
    for number in re.findall(r"-?\d+(?:\.\d+)?", json.dumps(value, default=str)):
        tokens.add(str(float(number)).rstrip("0").rstrip("."))
    return tokens


def _gemma_summary(evidence: dict[str, Any], model_name: str) -> str:
    from langchain_ollama import ChatOllama

    prompt = f"""You write only the final narrative for a deterministic financial risk assessment.
Use exclusively the evidence JSON below. Do not add risks, values, calculations, missing fields,
conditions, or recommendations. Do not change the overall risk or recommendation. Write 2-4 concise
sentences and state that the recommendation is not a lending decision. Return narrative text only.

<risk_evidence>
{json.dumps(evidence, indent=2)}
</risk_evidence>"""
    response = ChatOllama(model=model_name, temperature=0).invoke(prompt)
    summary = str(response.content).strip()
    if not summary:
        raise ValueError("Gemma returned an empty risk summary")
    allowed_numbers = _numeric_tokens(evidence)
    generated_numbers = _numeric_tokens(summary)
    if not generated_numbers.issubset(allowed_numbers):
        raise ValueError("Gemma summary introduced a number not present in deterministic evidence")
    return summary


class RiskSummaryAgent:
    def __init__(self, risk_rules_path: Path, summary_model: str = "gemma3:4b"):
        self.catalog = load_risk_rules(risk_rules_path)
        self.summary_model = summary_model

    def assess(self, terms: DealTerms, use_llm_summary: bool = True) -> Agent2Result:
        metrics, original_inputs = _metrics(terms)
        flags: list[RiskFlag] = []
        positives: list[str] = []
        missing: list[str] = []

        for rule in self.catalog.rules:
            actual = metrics.get(rule.metric)
            if actual is None:
                if rule.required:
                    missing.append(rule.metric)
                continue
            if _evaluate(actual, rule.operator, rule.threshold):
                severity: Severity = rule.severity
                if rule.critical_threshold is not None and float(actual) < rule.critical_threshold:
                    severity = "CRITICAL"
                flags.append(RiskFlag(
                    risk_id=rule.risk_id,
                    category=rule.category,
                    title=rule.title,
                    severity=severity,
                    original_values=original_inputs[rule.metric],
                    calculated_value=actual,
                    threshold_value=rule.threshold,
                    explanation=rule.message,
                    recommended_action=rule.recommended_action,
                ))
            else:
                positives.append(f"{rule.clear_message} Observed value: {actual}.")

        missing = sorted(set(missing))
        overall_risk, recommendation = _aggregate(flags, missing)
        confidence = round((len(self.catalog.rules) - len(missing)) / len(self.catalog.rules), 2)
        evidence = {
            "overall_risk": overall_risk,
            "recommendation": recommendation,
            "risk_flags": [flag.model_dump(mode="json") for flag in flags],
            "positive_factors": positives,
            "missing_information": missing,
            "calculated_metrics": metrics,
        }
        summary = _deterministic_summary(overall_risk, recommendation, flags, positives, missing)
        summary_model = "deterministic_template"
        if use_llm_summary:
            try:
                summary = _gemma_summary(evidence, self.summary_model)
                summary_model = self.summary_model
            except Exception:
                # A grounded deterministic summary is safer than returning an
                # unverified or unavailable model response.
                summary_model = f"{self.summary_model}:fallback"

        return Agent2Result(
            risk_rule_catalog_version=self.catalog.version,
            summary_model=summary_model,
            overall_risk=overall_risk,
            recommendation=recommendation,
            risk_flags=flags,
            positive_factors=positives,
            missing_information=missing,
            calculated_metrics=metrics,
            summary=summary,
            confidence=confidence,
        )
