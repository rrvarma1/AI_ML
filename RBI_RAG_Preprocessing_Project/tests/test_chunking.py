import importlib
import io
import os
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


class DoclingLlamaIndexChunkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory()
        root = Path(cls.temp.name)
        os.environ["RBI_CORPUS_DIR"] = str(root / "corpus")
        os.environ["RBI_HIERARCHY_OUTPUT_DIR"] = str(root / "hierarchy")
        os.environ["RBI_CHUNK_OUTPUT_DIR"] = str(root / "chunks")
        cls.module = importlib.import_module(
            "rbi_rag_preprocessing.steps.chunk_regulatory_documents"
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_page_metadata_from_docling_markers(self):
        text = "<!-- RBI_PAGE:1 -->\nfirst\n<!-- RBI_PAGE:2 -->\nsecond"
        self.assertEqual(self.module.page_for_offset(text, text.index("second")), 2)

    def test_full_markdown_heading_path(self):
        text = "# Chapter I\n## 4. KYC\n### 4.1 Records\nRule text"
        path = self.module.heading_path_for_offset(text, len(text), "Direction")
        self.assertEqual(path, ["Direction", "Chapter I", "4. KYC", "4.1 Records"])

    def test_legal_dependency_warning_detection(self):
        self.assertTrue(self.module.LEGAL_DEPENDENCY.match("Provided that the bank may..."))

    def test_summary_prints_totals_and_longest_split(self):
        manifest = [
            {"document_id": "short", "title": "Short", "source_character_count": 10},
            {"document_id": "long", "title": "Long", "source_character_count": 100},
        ]
        chunks = [{
            "content": "A final chunk",
            "metadata": {
                "document_id": "long", "start_index": 20, "end_index": 33,
            },
        }]
        output = io.StringIO()
        with redirect_stdout(output):
            self.module.print_chunking_summary(manifest, chunks)
        rendered = output.getvalue()
        self.assertIn("Documents in:  2", rendered)
        self.assertIn("Chunks out:    1", rendered)
        self.assertIn('Longest doc: "Long" (100 chars)', rendered)
        self.assertIn("start_index=20", rendered)

    def test_pdf_resolution_uses_hash_when_inventory_name_contains_slashes(self):
        with TemporaryDirectory() as directory:
            corpus = Path(directory)
            actual = corpus / "Penal Interest - Currency Chest Transactions.pdf"
            actual.write_bytes(b"test-pdf-content")
            metadata = {
                "document_id": "rbi_doc5",
                "file_name": "Penal Interest / Wrong Reporting / Non-Reporting.pdf",
                "pdf_sha256": self.module.sha256_file(actual),
            }
            resolved, method = self.module.resolve_source_pdf(metadata, corpus)
            self.assertEqual(resolved, actual)
            self.assertEqual(method, "sha256")


if __name__ == "__main__":
    unittest.main()
