"""Orchestrator coordinating extraction handoff and Agent 1 validation."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict

from agent1 import Agent1Result, ComplianceValidationAgent
from agent2 import Agent2Result, RiskSummaryAgent
from extractor import ExtractionOutput, run_extractor, run_extractor_fallback


ProgressCallback = Callable[
    [Literal["orchestrator", "extractor", "parallel_agents", "compliance_validator", "risk_summary"],
     Literal["in_progress", "completed", "failed", "compliant", "non_compliant", "insufficient_data", "acceptable_risk", "adverse_risk"]],
    None,
]


class OrchestrationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    extractor_output: ExtractionOutput
    extractor_handoff_path: str
    agent1_result: Agent1Result
    agent2_result: Agent2Result
    agent2_output_path: str


class DealReviewOrchestrator:
    def __init__(self, data_dir: Path, rules_path: Path, risk_rules_path: Path | None = None):
        self.data_dir = data_dir
        self.agent1 = ComplianceValidationAgent(rules_path)
        self.agent2 = RiskSummaryAgent(risk_rules_path or rules_path.with_name("risk_rules.json"))

    def run(
        self,
        document_text: str,
        document_bytes: bytes,
        mode: str,
        model_name: str,
        progress_callback: ProgressCallback | None = None,
    ) -> OrchestrationResult:
        notify = progress_callback or (lambda _stage, _status: None)
        notify("orchestrator", "in_progress")
        digest = hashlib.sha256(document_bytes).hexdigest()
        stamp = datetime.now(timezone.utc)
        case_id = f"PV-{stamp.strftime('%Y%m%d')}-{digest[:6].upper()}"
        notify("extractor", "in_progress")
        try:
            if mode == "Deterministic fallback":
                extraction = run_extractor_fallback(document_text)
            else:
                provider = "openai" if mode == "OpenAI LLM" else "ollama"
                extraction = run_extractor(document_text, provider=provider, model_name=model_name)
        except Exception:
            notify("extractor", "failed")
            notify("orchestrator", "failed")
            raise
        notify("extractor", "completed")

        case_dir = self.data_dir / "cases" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        handoff_path = case_dir / "extractor_output.json"
        handoff_path.write_text(json.dumps({
            "case_id": case_id,
            "created_at": stamp.isoformat(),
            "extraction_method": extraction.extraction_method,
            "model_name": extraction.model_name,
            "warnings": extraction.warnings,
            "extracted_terms": extraction.terms.model_dump(mode="json"),
        }, indent=2), encoding="utf-8")
        # Both agents consume the same extractor output and have no dependency
        # on each other, so the Orchestrator dispatches them concurrently.
        # A single event lets the UI mark both independent branches as running
        # atomically, avoiding a misleading one-agent-at-a-time transition.
        notify("parallel_agents", "in_progress")
        validation: Agent1Result | None = None
        risk_summary: Agent2Result | None = None
        failures: list[Exception] = []
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="deal-review") as executor:
            futures = {
                executor.submit(self.agent1.validate_handoff, handoff_path): "compliance_validator",
                executor.submit(
                    self.agent2.assess,
                    extraction.terms,
                    mode != "Deterministic fallback",
                ): "risk_summary",
            }
            for future in as_completed(futures):
                stage = futures[future]
                try:
                    stage_result = future.result()
                    if stage == "compliance_validator":
                        validation = stage_result
                        outcome_status = {
                            "COMPLIANT": "compliant",
                            "NON_COMPLIANT": "non_compliant",
                            "INSUFFICIENT_DATA": "insufficient_data",
                        }[validation.overall_status]
                    else:
                        risk_summary = stage_result
                        outcome_status = (
                            "acceptable_risk"
                            if risk_summary.overall_risk == "LOW" and risk_summary.recommendation == "PROCEED"
                            else "adverse_risk"
                        )
                    # Publish each business outcome as soon as that agent finishes;
                    # do not wait for the other parallel branch.
                    notify(stage, outcome_status)
                except Exception as exc:
                    failures.append(exc)
                    notify(stage, "failed")

        if failures:
            notify("orchestrator", "failed")
            raise failures[0]
        if validation is None or risk_summary is None:
            notify("orchestrator", "failed")
            raise RuntimeError("Parallel agent execution completed without both results")

        agent2_output_path = case_dir / "agent2_output.json"
        agent2_output_path.write_text(risk_summary.model_dump_json(indent=2), encoding="utf-8")
        notify("orchestrator", "completed")
        return OrchestrationResult(
            case_id=case_id, extractor_output=extraction,
            extractor_handoff_path=str(handoff_path), agent1_result=validation,
            agent2_result=risk_summary, agent2_output_path=str(agent2_output_path),
        )
