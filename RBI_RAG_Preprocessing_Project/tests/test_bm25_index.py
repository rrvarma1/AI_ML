import importlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


class BM25IndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory()
        root = Path(cls.temp.name)
        os.environ["RBI_CHUNK_OUTPUT_DIR"] = str(root / "RBI_Chunked_Documents")
        os.environ["RBI_BM25_OUTPUT_DIR"] = str(root / "RBI_BM25_Index")
        cls.module = importlib.import_module("rbi_rag_preprocessing.steps.build_bm25_index")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_regulatory_tokenizer_preserves_searchable_values(self):
        tokens = self.module.tokenize("KYC clause 33(a): INR 10,000 at 7.5% on 2025-08-14")
        self.assertIn("kyc", tokens)
        self.assertIn("33", tokens)
        self.assertIn("inr", tokens)
        self.assertIn("7.5%", tokens)
        self.assertIn("2025-08-14", tokens)

    def test_loader_preserves_deterministic_chunk_ids(self):
        chunks_file = Path(self.temp.name) / "chunks.jsonl"
        chunks_file.write_text(json.dumps({
            "content": "Know Your Customer requirements",
            "metadata": {"chunk_id": "rbi_doc1_chunk_0001", "document_id": "rbi_doc1"},
        }) + "\n", encoding="utf-8")
        records = self.module.load_chunk_records(chunks_file)
        self.assertEqual(records[0]["chunk_id"], "rbi_doc1_chunk_0001")
        self.assertEqual(records[0]["metadata"]["chunk_id"], "rbi_doc1_chunk_0001")
        self.assertTrue(records[0]["tokenized_text"])

    def test_duplicate_chunk_ids_are_rejected(self):
        chunks_file = Path(self.temp.name) / "duplicate.jsonl"
        record = {"content": "RBI requirement", "metadata": {"chunk_id": "same"}}
        chunks_file.write_text((json.dumps(record) + "\n") * 2, encoding="utf-8")
        with self.assertRaises(ValueError):
            self.module.load_chunk_records(chunks_file)


if __name__ == "__main__":
    unittest.main()
