"""LangGraph orchestration for the deal-review workflow."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, ConfigDict

from agent1 import Agent1Result, ComplianceValidationAgent
from agent2 import Agent2Result, RiskSummaryAgent
from extractor import ExtractionOutput, run_extractor, run_extractor_fallback


Stage = Literal["orchestrator", "extractor", "parallel_agents", "compliance_validator", "risk_summary"]
StageStatus = Literal[
    "in_progress", "completed", "failed", "interrupted", "compliant", "non_compliant",
    "insufficient_data", "acceptable_risk", "adverse_risk",
]
ProgressCallback = Callable[[Stage, StageStatus], None]
Decision = Literal["Approve", "Reject", "On Hold"]


class DealReviewState(TypedDict, total=False):
    document_text: str
    document_bytes: bytes
    mode: str
    model_name: str
    case_id: str
    created_at: str
    extractor_output: dict
    extractor_handoff_path: str
    agent1_result: dict
    agent2_result: dict
    agent2_output_path: str
    requires_human_review: bool
    interrupt_payload: dict
    final_decision: Decision | None
    workflow_status: Literal["RUNNING", "COMPLETED"]


class OrchestrationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    thread_id: str
    workflow_status: Literal["INTERRUPTED", "COMPLETED"]
    extractor_output: ExtractionOutput
    extractor_handoff_path: str
    agent1_result: Agent1Result
    agent2_result: Agent2Result
    agent2_output_path: str
    interrupt_payload: dict | None = None
    final_decision: Decision | None = None


# Shared across Streamlit reruns. The validation ID is the checkpoint thread ID.
MEMORY_SAVER = MemorySaver()


class DealReviewOrchestrator:
    def __init__(self, data_dir: Path, rules_path: Path, risk_rules_path: Path | None = None):
        self.data_dir = data_dir
        self.agent1 = ComplianceValidationAgent(rules_path)
        self.agent2 = RiskSummaryAgent(risk_rules_path or rules_path.with_name("risk_rules.json"))
        self._notify: ProgressCallback = lambda _stage, _status: None
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(DealReviewState)
        builder.add_node("initialize", self._initialize_node)
        builder.add_node("extract", self._extract_node)
        builder.add_node("dispatch_specialists", self._dispatch_node)
        builder.add_node("compliance_validator", self._compliance_node)
        builder.add_node("risk_summary", self._risk_node)
        builder.add_node("route_outcome", self._route_node)
        builder.add_node("deal_officer_review", self._deal_officer_node)
        builder.add_node("finalize", self._finalize_node)
        builder.add_edge(START, "initialize")
        builder.add_edge("initialize", "extract")
        builder.add_edge("extract", "dispatch_specialists")
        builder.add_edge("dispatch_specialists", "compliance_validator")
        builder.add_edge("dispatch_specialists", "risk_summary")
        builder.add_edge(["compliance_validator", "risk_summary"], "route_outcome")
        builder.add_conditional_edges(
            "route_outcome",
            lambda state: "deal_officer_review" if state["requires_human_review"] else "finalize",
            {"deal_officer_review": "deal_officer_review", "finalize": "finalize"},
        )
        builder.add_edge("deal_officer_review", "finalize")
        builder.add_edge("finalize", END)
        return builder.compile(checkpointer=MEMORY_SAVER)

    def _initialize_node(self, state: DealReviewState) -> dict:
        self._notify("orchestrator", "in_progress")
        digest = hashlib.sha256(state["document_bytes"]).hexdigest()
        stamp = datetime.now(timezone.utc)
        return {
            "case_id": f"PV-{stamp.strftime('%Y%m%d')}-{digest[:6].upper()}",
            "created_at": stamp.isoformat(), "workflow_status": "RUNNING", "final_decision": None,
        }

    def _extract_node(self, state: DealReviewState) -> dict:
        self._notify("extractor", "in_progress")
        try:
            if state["mode"] == "Deterministic fallback":
                extraction = run_extractor_fallback(state["document_text"])
            else:
                provider = "openai" if state["mode"] == "OpenAI LLM" else "ollama"
                extraction = run_extractor(state["document_text"], provider=provider, model_name=state["model_name"])
        except Exception:
            self._notify("extractor", "failed")
            self._notify("orchestrator", "failed")
            raise
        case_dir = self.data_dir / "cases" / state["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        handoff_path = case_dir / "extractor_output.json"
        payload = {
            "case_id": state["case_id"], "created_at": state["created_at"],
            "extraction_method": extraction.extraction_method, "model_name": extraction.model_name,
            "warnings": extraction.warnings, "extracted_terms": extraction.terms.model_dump(mode="json"),
        }
        handoff_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._notify("extractor", "completed")
        return {"extractor_output": extraction.model_dump(mode="json"), "extractor_handoff_path": str(handoff_path)}

    def _dispatch_node(self, _state: DealReviewState) -> dict:
        self._notify("parallel_agents", "in_progress")
        return {}

    def _compliance_node(self, state: DealReviewState) -> dict:
        try:
            result = self.agent1.validate_handoff(Path(state["extractor_handoff_path"]))
        except Exception:
            self._notify("compliance_validator", "failed")
            raise
        outcome = {"COMPLIANT": "compliant", "NON_COMPLIANT": "non_compliant", "INSUFFICIENT_DATA": "insufficient_data"}[result.overall_status]
        self._notify("compliance_validator", outcome)
        return {"agent1_result": result.model_dump(mode="json")}

    def _risk_node(self, state: DealReviewState) -> dict:
        extraction = ExtractionOutput.model_validate(state["extractor_output"])
        try:
            result = self.agent2.assess(extraction.terms, state["mode"] != "Deterministic fallback")
        except Exception:
            self._notify("risk_summary", "failed")
            raise
        outcome = "acceptable_risk" if result.overall_risk == "LOW" and result.recommendation == "PROCEED" else "adverse_risk"
        self._notify("risk_summary", outcome)
        output_path = self.data_dir / "cases" / state["case_id"] / "agent2_output.json"
        output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return {"agent2_result": result.model_dump(mode="json"), "agent2_output_path": str(output_path)}

    def _route_node(self, state: DealReviewState) -> dict:
        compliance = Agent1Result.model_validate(state["agent1_result"])
        risk = Agent2Result.model_validate(state["agent2_result"])
        adverse = compliance.overall_status != "COMPLIANT" or not (
            risk.overall_risk == "LOW" and risk.recommendation == "PROCEED"
        )
        payload = {
            "case_id": state["case_id"],
            "instruction": "Review all extracted terms, compliance findings, and risk findings, then choose Approve, Reject, or On Hold.",
            "extractor_output": state["extractor_output"], "compliance_result": state["agent1_result"],
            "risk_result": state["agent2_result"], "allowed_decisions": ["Approve", "Reject", "On Hold"],
        }
        return {"requires_human_review": adverse, "interrupt_payload": payload if adverse else {}}

    def _deal_officer_node(self, state: DealReviewState) -> dict:
        self._notify("orchestrator", "interrupted")
        decision = interrupt(state["interrupt_payload"])
        if decision not in {"Approve", "Reject", "On Hold"}:
            raise ValueError("Deal Officer decision must be Approve, Reject, or On Hold")
        return {"final_decision": decision, "workflow_status": "RUNNING"}

    def _finalize_node(self, _state: DealReviewState) -> dict:
        self._notify("orchestrator", "completed")
        return {"workflow_status": "COMPLETED"}

    @staticmethod
    def _config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def _to_result(self, state: dict, thread_id: str) -> OrchestrationResult:
        interrupted = bool(state.get("__interrupt__"))
        interrupt_payload = state["__interrupt__"][0].value if interrupted else None
        return OrchestrationResult(
            case_id=state["case_id"], thread_id=thread_id,
            workflow_status="INTERRUPTED" if interrupted else "COMPLETED",
            extractor_output=ExtractionOutput.model_validate(state["extractor_output"]),
            extractor_handoff_path=state["extractor_handoff_path"],
            agent1_result=Agent1Result.model_validate(state["agent1_result"]),
            agent2_result=Agent2Result.model_validate(state["agent2_result"]),
            agent2_output_path=state["agent2_output_path"], interrupt_payload=interrupt_payload,
            final_decision=state.get("final_decision"),
        )

    def run(self, document_text: str, document_bytes: bytes, mode: str, model_name: str,
            progress_callback: ProgressCallback | None = None) -> OrchestrationResult:
        self._notify = progress_callback or (lambda _stage, _status: None)
        digest = hashlib.sha256(document_bytes).hexdigest()
        thread_id = f"PV-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{digest[:6].upper()}"
        initial: DealReviewState = {
            "document_text": document_text, "document_bytes": document_bytes, "mode": mode, "model_name": model_name,
        }
        state = self.graph.invoke(initial, config=self._config(thread_id))
        return self._to_result(state, thread_id)

    def resume(self, thread_id: str, decision: Decision,
               progress_callback: ProgressCallback | None = None) -> OrchestrationResult:
        if decision not in {"Approve", "Reject", "On Hold"}:
            raise ValueError("Deal Officer decision must be Approve, Reject, or On Hold")
        self._notify = progress_callback or (lambda _stage, _status: None)
        state = self.graph.invoke(Command(resume=decision), config=self._config(thread_id))
        return self._to_result(state, thread_id)

    def has_pending_interrupt(self, thread_id: str) -> bool:
        """Return True only when LangGraph has a resumable human-review checkpoint."""
        if not thread_id:
            return False
        try:
            snapshot = self.graph.get_state(self._config(thread_id))
        except Exception:
            return False
        return bool(snapshot and snapshot.next and "deal_officer_review" in snapshot.next)
