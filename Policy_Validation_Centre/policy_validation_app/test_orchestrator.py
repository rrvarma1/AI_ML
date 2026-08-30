import json
import threading
from pathlib import Path

import pytest

import orchestrator
from orchestrator import DealReviewOrchestrator


def test_orchestrator_persists_json_handoff_and_invokes_agent1(tmp_path):
    text = Path("sample_deal.txt").read_text(encoding="utf-8")
    events = []
    result = DealReviewOrchestrator(tmp_path, Path("rules.json")).run(
        document_text=text, document_bytes=text.encode(), mode="Deterministic fallback", model_name="",
        progress_callback=lambda stage, status: events.append((stage, status)),
    )
    handoff = Path(result.extractor_handoff_path)
    assert handoff.is_file()
    payload = json.loads(handoff.read_text(encoding="utf-8"))
    assert payload["case_id"] == result.case_id
    assert payload["extracted_terms"]["loan_amount"] == 50_000_000
    assert result.agent1_result.rule_catalog_version == "greenleaf-demo-1.1"
    assert Path(result.agent2_output_path).is_file()
    assert result.agent2_result.overall_risk == "LOW"
    assert result.agent2_result.recommendation == "PROCEED"
    assert events[:4] == [
        ("orchestrator", "in_progress"),
        ("extractor", "in_progress"),
        ("extractor", "completed"),
        ("parallel_agents", "in_progress"),
    ]
    assert set(events[4:6]) == {
        ("compliance_validator", "compliant"),
        ("risk_summary", "acceptable_risk"),
    }
    assert events[-1] == ("orchestrator", "completed")


def test_orchestrator_runs_agent1_and_agent2_in_parallel(tmp_path, monkeypatch):
    review = DealReviewOrchestrator(tmp_path, Path("rules.json"))
    barrier = threading.Barrier(2)
    original_agent1 = review.agent1.validate_handoff
    original_agent2 = review.agent2.assess

    def agent1_with_barrier(path):
        barrier.wait(timeout=2)
        return original_agent1(path)

    def agent2_with_barrier(terms, use_llm_summary=True):
        barrier.wait(timeout=2)
        return original_agent2(terms, use_llm_summary)

    monkeypatch.setattr(review.agent1, "validate_handoff", agent1_with_barrier)
    monkeypatch.setattr(review.agent2, "assess", agent2_with_barrier)
    text = Path("sample_deal.txt").read_text(encoding="utf-8")
    result = review.run(
        document_text=text, document_bytes=text.encode(),
        mode="Deterministic fallback", model_name="",
    )
    assert result.agent1_result.overall_status == "COMPLIANT"
    assert result.agent2_result.overall_risk == "LOW"


def test_orchestrator_publishes_adverse_business_outcomes(tmp_path):
    text = Path("sample_deal.txt").read_text(encoding="utf-8")
    text = text.replace("Debt-to-Equity: 1.35x", "Debt-to-Equity: 2.80x")
    text = text.replace("Current Ratio: 1.35x", "Current Ratio: 1.05x")
    events = []
    review = DealReviewOrchestrator(tmp_path, Path("rules.json"))
    result = review.run(
        document_text=text, document_bytes=text.encode(), mode="Deterministic fallback", model_name="",
        progress_callback=lambda stage, status: events.append((stage, status)),
    )
    assert result.agent1_result.overall_status == "NON_COMPLIANT"
    assert result.agent2_result.overall_risk in {"HIGH", "CRITICAL"}
    assert ("parallel_agents", "in_progress") in events
    assert ("compliance_validator", "non_compliant") in events
    assert ("risk_summary", "adverse_risk") in events
    assert result.workflow_status == "INTERRUPTED"
    assert result.interrupt_payload["case_id"] == result.case_id
    assert result.interrupt_payload["extractor_output"]["terms"]["debt_to_equity"] == 2.8
    assert events[-1] == ("orchestrator", "interrupted")
    checkpoint = review.graph.get_state({"configurable": {"thread_id": result.thread_id}})
    assert checkpoint.values["agent1_result"]["overall_status"] == "NON_COMPLIANT"
    assert checkpoint.values["agent2_result"]["overall_risk"] in {"HIGH", "CRITICAL"}
    assert checkpoint.next == ("deal_officer_review",)
    assert review.has_pending_interrupt(result.thread_id) is True

    resumed = DealReviewOrchestrator(tmp_path, Path("rules.json")).resume(result.thread_id, "On Hold")
    assert resumed.workflow_status == "COMPLETED"
    assert resumed.final_decision == "On Hold"
    assert review.has_pending_interrupt(result.thread_id) is False


def test_orchestrator_does_not_report_missing_checkpoint_as_resumable(tmp_path):
    review = DealReviewOrchestrator(tmp_path, Path("rules.json"))
    assert review.has_pending_interrupt("") is False
    assert review.has_pending_interrupt("PV-NOT-FOUND") is False


def test_orchestrator_reports_extractor_failure(tmp_path, monkeypatch):
    events = []

    def fail_extraction(_document_text):
        raise RuntimeError("extraction failed")

    monkeypatch.setattr(orchestrator, "run_extractor_fallback", fail_extraction)
    with pytest.raises(RuntimeError, match="extraction failed"):
        DealReviewOrchestrator(tmp_path, Path("rules.json")).run(
            document_text="deal", document_bytes=b"deal", mode="Deterministic fallback", model_name="",
            progress_callback=lambda stage, status: events.append((stage, status)),
        )

    assert events[-2:] == [("extractor", "failed"), ("orchestrator", "failed")]
