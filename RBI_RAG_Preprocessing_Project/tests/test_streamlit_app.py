from pathlib import Path
import unittest

from rbi_rag_preprocessing.streamlit_app import (
    build_download_markdown,
    build_score_lookup,
    build_stage_command,
    clear_evidence_states,
    heading_label,
    page_label,
    history_title,
    pipeline_progress_html,
)


class StreamlitAppTests(unittest.TestCase):
    def test_stage_command_uses_package_cli_and_configured_values(self):
        command = build_stage_command(
            stage="hybrid", query="What is CRR?", corpus_dir=Path("/data/pdfs"),
            inventory_file=Path("/data/inventory.xlsx"), project_root=Path("/data"),
            namespace="RBI_RAG", rerank_model="cohere-rerank-3.5", llm_model="gemma3:4b",
        )
        self.assertEqual(command[1:3], ["-m", "rbi_rag_preprocessing"])
        self.assertEqual(command[command.index("--step") + 1], "hybrid")
        self.assertEqual(command[command.index("--query") + 1], "What is CRR?")
        self.assertEqual(command[command.index("--namespace") + 1], "RBI_RAG")

    def test_score_lookup_combines_hybrid_and_reranker_results(self):
        hybrid = {"candidates": [{
            "chunk_id": "c1", "rank": 2, "rrf_score": 0.03,
            "matched_by": ["semantic", "lexical"],
            "semantic": {"rank": 1, "raw_score": 0.8, "normalized_score": 1.0},
            "lexical": {"rank": 4, "raw_score": 5.1, "normalized_score": 0.7},
        }]}
        rerank = {"candidates": [{
            "chunk_id": "c1", "reranker_rank": 1, "reranker_score": 0.94,
        }]}
        result = build_score_lookup(hybrid, rerank)["c1"]
        self.assertEqual(result["hybrid_rank"], 2)
        self.assertEqual(result["semantic"]["rank"], 1)
        self.assertEqual(result["lexical"]["raw_score"], 5.1)
        self.assertEqual(result["reranker_score"], 0.94)

    def test_heading_and_page_labels_preserve_regulatory_path(self):
        metadata = {
            "heading_path": ["Chapter VI", "Section 33", "Clause 33(a)"],
            "page_start": 42, "page_end": 43,
        }
        self.assertEqual(
            heading_label(metadata), "Chapter VI → Section 33 → Clause 33(a)"
        )
        self.assertEqual(page_label(metadata), "Pages 42–43")
        self.assertEqual(page_label({"page_start": 7, "page_end": 7}), "Page 7")

    def test_history_title_is_compact(self):
        question = "For how long must regulated entities maintain transaction records?"
        self.assertLessEqual(len(history_title(question, 32)), 32)
        self.assertTrue(history_title(question, 32).endswith("…"))

    def test_pipeline_progress_keeps_all_user_facing_stages_visible(self):
        markup = pipeline_progress_html("rerank")
        self.assertIn("Running the grounded Q&amp;A pipeline", markup)
        self.assertIn("Retrieve semantic and lexical candidates", markup)
        self.assertIn("Rerank the fused candidate pool", markup)
        self.assertIn("Expand regulatory parent context", markup)
        self.assertIn("Generate a grounded answer with Gemma", markup)
        self.assertIn('pipeline-step active', markup)

    def test_evidence_state_resets_only_for_new_chat_or_question(self):
        state = {
            "evidence_details_open_query_1": True,
            "evidence_details_open_query_2": True,
            "rbi_active_chat": 1,
        }
        clear_evidence_states(state)
        self.assertNotIn("evidence_details_open_query_1", state)
        self.assertNotIn("evidence_details_open_query_2", state)
        self.assertEqual(state["rbi_active_chat"], 1)

    def test_download_contains_question_answer_and_deterministic_source(self):
        results = {
            "answer": {"query": "How long?", "answer": {
                "answer": "Five years. [E1]", "conditions_and_exceptions": [],
                "conflicts": [],
            }},
            "citations": {"unique_citations_used": [{
                "document_title": "KYC Direction", "full_heading_path_text": "Record Management",
                "section_or_clause": "Section 46(a)", "page_range": "page 49",
                "chunk_id": "rbi_doc7__chunk_0210", "source_url": "https://rbi.org.in/source",
            }]},
        }
        markdown = build_download_markdown(results)
        self.assertIn("How long?", markdown)
        self.assertIn("Five years. [E1]", markdown)
        self.assertIn("rbi_doc7__chunk_0210", markdown)
        self.assertIn("https://rbi.org.in/source", markdown)
