import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path


class EvaluationDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        os.environ["RBI_CHUNK_OUTPUT_DIR"] = str(root / "chunks")
        os.environ["RBI_EVALUATION_OUTPUT_DIR"] = str(root / "evaluation")
        cls.module = importlib.import_module(
            "rbi_rag_preprocessing.steps.build_evaluation_dataset"
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_selector_is_restricted_to_requested_document(self):
        chunks = [
            {"chunk_id": "wrong", "content": "five years date of transaction", "metadata": {"document_id": "rbi_doc1"}},
            {"chunk_id": "right", "content": "maintain records for at least five years", "metadata": {"document_id": "rbi_doc7"}},
        ]
        chosen = self.module.resolve_selector(
            chunks, {"document_id": "rbi_doc7", "terms": ["five years"]}
        )
        self.assertEqual(chosen[0]["chunk_id"], "right")

    def test_validation_rejects_unknown_supporting_chunk(self):
        item = {
            "question_id": "Q1", "category": "direct_lookup", "question": "q",
            "expected_answer": "a", "supporting_chunk_ids": ["missing"],
            "required_citations": [], "should_abstain": False,
        }
        report = self.module.validate([item] * 15, {"known"})
        self.assertEqual(report["status"], "failed")
        self.assertTrue(any("unknown chunks" in error for error in report["errors"]))

    def test_chunk_loader_preserves_real_deterministic_id(self):
        path = Path(self.temp.name) / "sample.jsonl"
        path.write_text(json.dumps({
            "content": "rule", "metadata": {
                "chunk_id": "rbi_doc7__chunk_0047", "document_id": "rbi_doc7"
            }
        }) + "\n", encoding="utf-8")
        loaded = self.module.load_chunks(path)
        self.assertEqual(loaded[0]["chunk_id"], "rbi_doc7__chunk_0047")
