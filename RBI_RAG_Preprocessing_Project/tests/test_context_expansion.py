import importlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


class ContextExpansionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory()
        root = Path(cls.temp.name)
        os.environ["RBI_CHUNK_OUTPUT_DIR"] = str(root / "chunks")
        os.environ["RBI_RERANK_OUTPUT_DIR"] = str(root / "reranked")
        os.environ["RBI_CONTEXT_OUTPUT_DIR"] = str(root / "expanded")
        cls.module = importlib.import_module(
            "rbi_rag_preprocessing.steps.expand_parent_context"
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def nodes():
        return {
            "parent": {
                "node_id": "parent", "document_id": "doc1",
                "text": "Main rule.\n[^1] This footnote changes interpretation.",
                "metadata": {"document_id": "doc1"},
                "relationships": {"children": ["previous", "leaf"]},
            },
            "previous": {
                "node_id": "previous", "document_id": "doc1",
                "text": "The regulated entity shall complete due diligence.",
                "metadata": {"document_id": "doc1"},
                "relationships": {"parent": "parent", "next": "leaf"},
            },
            "leaf": {
                "node_id": "leaf", "document_id": "doc1",
                "text": "Provided that the exception applies.[^1]",
                "metadata": {"document_id": "doc1"},
                "relationships": {"parent": "parent", "previous": "previous"},
            },
        }

    @staticmethod
    def candidate(content="Provided that the exception applies.[^1]"):
        return {
            "rank": 1, "chunk_id": "doc1__chunk_0002", "content": content,
            "metadata": {
                "document_id": "doc1", "llamaindex_node_id": "leaf",
                "parent_node_id": "parent", "heading_path": ["Direction", "Section 3"],
                "heading_path_text": "Direction > Section 3",
                "starts_with_legal_dependency": True,
            },
        }

    def test_legal_exception_gets_parent_and_governing_rule(self):
        expanded, contexts = self.module.expand_candidate(self.candidate(), self.nodes())
        self.assertIn("parent", expanded["expansion"]["context_node_ids"])
        self.assertIn("previous", expanded["expansion"]["context_node_ids"])
        self.assertIn("governing_rule_for_legal_dependency", expanded["expansion"]["reasons"])
        self.assertIn("The regulated entity shall", expanded["expanded_content"])
        self.assertEqual(len(contexts), 2)

    def test_relevant_footnote_is_resolved_from_parent(self):
        expanded, _ = self.module.expand_candidate(self.candidate(), self.nodes())
        self.assertEqual(
            expanded["expansion"]["relevant_footnotes"],
            ["[^1] This footnote changes interpretation."],
        )

    def test_table_headers_are_repeated(self):
        text = "| Requirement | Timeline |\n|---|---|\n| KYC | 10 days |"
        headers = self.module.table_headers(text)
        self.assertEqual(headers, ["| Requirement | Timeline |\n|---|---|"])

    def test_context_catalog_deduplicates_shared_parent(self):
        first = self.candidate()
        second = self.candidate("Provided that another exception applies.")
        second["chunk_id"] = "doc1__chunk_0003"
        expanded, catalog = self.module.expand_all([first, second], self.nodes())
        self.assertEqual(len(expanded), 2)
        self.assertIn("parent", catalog)
        self.assertEqual(list(catalog).count("parent"), 1)


if __name__ == "__main__":
    unittest.main()
