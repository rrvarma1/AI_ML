from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langsmith import Client, traceable

from orchestrator import DealReviewOrchestrator


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATASET_NAME = "Policy_Validation_Center_Golden_Dataset_v1"
MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma3:4b")
PROMPT_VERSION = os.getenv("PVC_PROMPT_VERSION", "extractor-v1")
RULES_VERSION = os.getenv("PVC_RULES_VERSION", "rules-v1")
RISK_RULES_VERSION = os.getenv(
    "PVC_RISK_RULES_VERSION",
    "risk-rules-v1",
)

orchestrator = DealReviewOrchestrator(
    data_dir=BASE_DIR / "data",
    rules_path=BASE_DIR / "rules.json",
    risk_rules_path=BASE_DIR / "risk_rules.json",
)


@traceable(
    name="policy-validation-center-case",
    run_type="chain",
)
def execute_case(inputs: dict[str, Any]) -> dict[str, Any]:
    test_id = str(inputs.get("Test_Id") or "UNKNOWN-CASE")
    document_text = str(inputs.get("User_Input") or "")
    officer_decision = inputs.get("Officer_Decision")

    if not document_text.strip():
        return {
            "test_id": test_id,
            "workflow_status": "FAILED",
            "error_stage": "input",
            "error_class": "MissingDocumentText",
            "error": "User_Input is empty.",
        }

    current_stage = {"name": "orchestrator"}

    def track_progress(stage: str, status: str) -> None:
        current_stage["name"] = stage

    try:
        # Including Test_Id in document_bytes gives every golden case a
        # separate LangGraph checkpoint ID without altering extracted text.
        checkpoint_bytes = (
            f"EVALUATION_CASE={test_id}\n{document_text}"
        ).encode("utf-8")

        result = orchestrator.run(
            document_text=document_text,
            document_bytes=checkpoint_bytes,
            mode="Ollama LLM",
            model_name=MODEL_NAME,
            progress_callback=track_progress,
        )

        if (
            result.workflow_status == "INTERRUPTED"
            and officer_decision not in (None, "")
        ):
            normalized_decision = str(officer_decision).strip()

            if normalized_decision not in {
                "Approve",
                "Reject",
                "On Hold",
            }:
                raise ValueError(
                    "Officer_Decision must be Approve, Reject, or On Hold"
                )

            result = orchestrator.resume(
                thread_id=result.thread_id,
                decision=normalized_decision,
                progress_callback=track_progress,
            )

        output = result.model_dump(mode="json")
        output["test_id"] = test_id
        return output

    except Exception as exc:
        return {
            "test_id": test_id,
            "workflow_status": "FAILED",
            "error_stage": current_stage["name"],
            "error_class": type(exc).__name__,
            "error": str(exc),
        }


def evaluation_target(inputs: dict[str, Any]) -> dict[str, Any]:
    test_id = str(inputs.get("Test_Id") or "UNKNOWN-CASE")

    return execute_case(
        inputs,
        langsmith_extra={
            "name": f"pvc-eval-{test_id}",
            "metadata": {
                "case_id": test_id,
                "run_name": f"pvc-eval-{test_id}",
                "prompt_version": PROMPT_VERSION,
                "model": MODEL_NAME,
                "rules_version": RULES_VERSION,
                "risk_rules_version": RISK_RULES_VERSION,
            },
        },
    )


def main() -> None:
    client = Client()

    results = client.evaluate(
        evaluation_target,
        data=DATASET_NAME,
        experiment_prefix="pvc-corrected-v2-gemma3-4b",
        description=(
            "Updaed Policy Validation Center evaluation using "
            "Gemma 3 4B and golden dataset v1."
        ),
        metadata={
            "evaluation_version": "corrected-v2",
            "model": "gemma3:4b",
            "prompt_version": "extractor-v2",
            "rules_version": "rules-v1",
            "risk_rules_version": "risk-rules-v1",
        },
        max_concurrency=1,
    )

    print(results)


if __name__ == "__main__":
    main()