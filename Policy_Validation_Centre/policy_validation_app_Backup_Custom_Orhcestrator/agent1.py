"""Agent 1: deterministic compliance validation over extractor JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from extractor import DealTerms
from validation_rules import RuleDefinition, evaluate_value, load_rule_catalog


class RuleFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rule_id: str
    field: str
    status: Literal["PASS", "FAIL", "INSUFFICIENT_DATA"]
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    actual_value: object | None
    operator: str
    threshold: object
    message: str


class Agent1Result(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent_name: Literal["agent_1_compliance_validator"] = "agent_1_compliance_validator"
    rule_catalog_version: str
    overall_status: Literal["COMPLIANT", "NON_COMPLIANT", "INSUFFICIENT_DATA"]
    score: int
    passed: int
    failed: int
    insufficient_data: int
    findings: list[RuleFinding]


class ComplianceValidationAgent:
    def __init__(self, rules_path: Path):
        self.rules_path = rules_path

    @staticmethod
    def _evaluate_rule(rule: RuleDefinition, terms: DealTerms) -> RuleFinding:
        actual = getattr(terms, rule.field, None)
        status = "INSUFFICIENT_DATA" if actual in (None, "") else (
            "PASS" if evaluate_value(actual, rule.operator, rule.threshold) else "FAIL"
        )
        return RuleFinding(
            rule_id=rule.rule_id, field=rule.field, status=status, severity=rule.severity,
            actual_value=actual, operator=rule.operator, threshold=rule.threshold, message=rule.message,
        )

    def validate_handoff(self, extractor_json_path: Path) -> Agent1Result:
        if not extractor_json_path.is_file():
            raise FileNotFoundError(f"Extractor handoff not found: {extractor_json_path}")
        handoff = json.loads(extractor_json_path.read_text(encoding="utf-8"))
        terms = DealTerms.model_validate(handoff.get("extracted_terms"))
        catalog = load_rule_catalog(self.rules_path)
        unknown = sorted({rule.field for rule in catalog.rules if rule.field not in DealTerms.model_fields})
        if unknown:
            raise ValueError("Rules reference unknown extractor fields: " + ", ".join(unknown))

        findings = [self._evaluate_rule(rule, terms) for rule in catalog.rules]
        passed = sum(item.status == "PASS" for item in findings)
        failed = sum(item.status == "FAIL" for item in findings)
        insufficient = sum(item.status == "INSUFFICIENT_DATA" for item in findings)
        score = round(100 * passed / len(findings)) if findings else 0
        overall = "NON_COMPLIANT" if failed else "INSUFFICIENT_DATA" if insufficient else "COMPLIANT"
        return Agent1Result(
            rule_catalog_version=catalog.version, overall_status=overall, score=score,
            passed=passed, failed=failed, insufficient_data=insufficient, findings=findings,
        )

