import importlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


class PineconeIndexingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory()
        root = Path(cls.temp.name)
        os.environ["RBI_CHUNK_OUTPUT_DIR"] = str(root / "RBI_Chunked_Documents")
        os.environ["RBI_VECTOR_OUTPUT_DIR"] = str(root / "RBI_Vector_Indexing")
        cls.module = importlib.import_module("rbi_rag_preprocessing.steps.index_pinecone")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_metadata_is_flattened_for_pinecone(self):
        cleaned = self.module.serializable_metadata({
            "document_id": "rbi_doc1",
            "heading_path": ["Chapter I", "Section 3"],
            "nested": {"clause": "3(a)"},
            "empty": None,
        })
        self.assertEqual(cleaned["heading_path"], ["Chapter I", "Section 3"])
        self.assertEqual(cleaned["nested"], '{"clause":"3(a)"}')
        self.assertNotIn("empty", cleaned)

    def test_index_configuration_validation(self):
        class FakePinecone:
            def describe_index(self, name):
                return {"name": name, "dimension": 512, "metric": "cosine"}

        result = self.module.verify_existing_index(FakePinecone(), "rbi-index")
        self.assertEqual(result["dimension"], 512)
        self.assertEqual(result["metric"], "cosine")

    def test_wrong_dimension_is_rejected_before_upload(self):
        class FakePinecone:
            def describe_index(self, name):
                return {"name": name, "dimension": 1536, "metric": "cosine"}

        with self.assertRaises(ValueError):
            self.module.verify_existing_index(FakePinecone(), "wrong-index")


if __name__ == "__main__":
    unittest.main()
