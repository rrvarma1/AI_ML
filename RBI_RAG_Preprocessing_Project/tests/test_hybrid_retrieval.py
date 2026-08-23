import importlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


class HybridRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory()
        root = Path(cls.temp.name)
        os.environ["RBI_BM25_OUTPUT_DIR"] = str(root / "RBI_BM25_Index")
        os.environ["RBI_HYBRID_OUTPUT_DIR"] = str(root / "RBI_Hybrid_Retrieval")
        cls.module = importlib.import_module("rbi_rag_preprocessing.steps.hybrid_retrieval")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def candidate(chunk_id, score):
        return {
            "chunk_id": chunk_id,
            "raw_score": score,
            "normalized_score": score,
            "content": f"Content {chunk_id}",
            "metadata": {"chunk_id": chunk_id, "title": f"Title {chunk_id}"},
        }

    def test_min_max_normalization(self):
        candidates = [
            {"chunk_id": "a", "raw_score": 10.0},
            {"chunk_id": "b", "raw_score": 20.0},
            {"chunk_id": "c", "raw_score": 15.0},
        ]
        normalized = self.module.min_max_normalize(candidates)
        self.assertEqual(normalized[0]["normalized_score"], 0.0)
        self.assertEqual(normalized[1]["normalized_score"], 1.0)
        self.assertEqual(normalized[2]["normalized_score"], 0.5)

    def test_rrf_merges_by_chunk_id(self):
        semantic = [self.candidate("shared", 1.0), self.candidate("semantic-only", 0.5)]
        lexical = [self.candidate("shared", 1.0), self.candidate("lexical-only", 0.5)]
        results = self.module.reciprocal_rank_fusion(semantic, lexical, 60, 20)
        self.assertEqual(results[0]["chunk_id"], "shared")
        self.assertEqual(results[0]["matched_by"], ["semantic", "lexical"])
        self.assertEqual(len(results), 3)
        self.assertAlmostEqual(results[0]["rrf_score"], 2 / 61)

    def test_rrf_respects_final_pool_limit(self):
        semantic = [self.candidate(str(i), 1.0) for i in range(25)]
        results = self.module.reciprocal_rank_fusion(semantic, [], 60, 20)
        self.assertEqual(len(results), 20)


if __name__ == "__main__":
    unittest.main()
