import importlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


class RerankCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory()
        root = Path(cls.temp.name)
        os.environ["RBI_HYBRID_OUTPUT_DIR"] = str(root / "RBI_Hybrid_Retrieval")
        os.environ["RBI_RERANK_OUTPUT_DIR"] = str(root / "RBI_Reranked_Retrieval")
        cls.module = importlib.import_module("rbi_rag_preprocessing.steps.rerank_candidates")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def candidates():
        return [
            {
                "rank": index + 1,
                "chunk_id": f"chunk-{index}",
                "content": f"Regulatory content {index}",
                "metadata": {"title": "Master Direction", "heading_path": ["Chapter I"]},
                "rrf_score": 0.01,
            }
            for index in range(3)
        ]

    def test_rerank_text_includes_retrieval_context(self):
        text = self.module.rerank_text(self.candidates()[0])
        self.assertIn("Document title: Master Direction", text)
        self.assertIn("Heading path: Chapter I", text)
        self.assertIn("Regulatory content 0", text)

    def test_pinecone_response_is_mapped_to_original_chunk(self):
        class Inference:
            def rerank(self, **kwargs):
                return {
                    "data": [
                        {"index": 2, "score": 0.95},
                        {"index": 0, "score": 0.80},
                        {"index": 1, "score": 0.25},
                    ],
                    "usage": {"rerank_units": 1},
                }

        class Pinecone:
            inference = Inference()

        reranked, usage = self.module.rerank(Pinecone(), "What is required?", self.candidates())
        self.assertEqual(reranked[0]["chunk_id"], "chunk-2")
        self.assertEqual(reranked[0]["hybrid_rank"], 3)
        self.assertEqual(reranked[0]["reranker_rank"], 1)
        self.assertEqual(reranked[0]["reranker_score"], 0.95)
        self.assertEqual(usage["rerank_units"], 1)

    def test_mismatched_query_is_detectable_from_loaded_payload(self):
        path = Path(self.temp.name) / "hybrid.json"
        import json
        path.write_text(json.dumps({"query": "Original", "candidates": self.candidates()}))
        query, candidates, _ = self.module.load_hybrid_candidates(path)
        self.assertEqual(query, "Original")
        self.assertEqual(len(candidates), 3)

    def test_usage_with_non_callable_model_dump_is_serialized(self):
        class PineconeUsage:
            model_dump = None

            def __init__(self):
                self.rerank_units = 1

        self.assertEqual(
            self.module.jsonable(PineconeUsage()),
            {"rerank_units": 1},
        )


if __name__ == "__main__":
    unittest.main()
