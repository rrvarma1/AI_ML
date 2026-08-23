import importlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


class CitationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory()
        root = Path(cls.temp.name)
        os.environ["RBI_ANSWER_OUTPUT_DIR"] = str(root / "answers")
        os.environ["RBI_CITATION_OUTPUT_DIR"] = str(root / "citations")
        cls.module = importlib.import_module(
            "rbi_rag_preprocessing.steps.generate_citations"
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def evidence():
        return {
            "evidence_id": "E1", "chunk_id": "rbi_doc1__chunk_0047",
            "metadata": {
                "document_id": "rbi_doc1",
                "title": "Master Direction - Know Your Customer",
                "heading_path": [
                    "Master Direction - Know Your Customer", "Chapter VI",
                    "Section 33", "Clause 33(a)",
                ],
                "page_start": 42, "page_end": 43,
                "source_url": "https://rbi.org.in/example",
                "regulatory_status": "binding_direction",
            },
        }

    def test_complete_citation_comes_only_from_metadata(self):
        citation = self.module.make_citation(self.evidence())
        self.assertEqual(citation["document_title"], "Master Direction - Know Your Customer")
        self.assertEqual(
            citation["full_heading_path"], ["Chapter VI", "Section 33", "Clause 33(a)"]
        )
        self.assertIn("Section 33", citation["section_or_clause"])
        self.assertIn("Clause 33(a)", citation["section_or_clause"])
        self.assertEqual(citation["page_range"], "pages 42–43")
        self.assertEqual(citation["source_url"], "https://rbi.org.in/example")
        self.assertEqual(citation["chunk_id"], "rbi_doc1__chunk_0047")
        self.assertTrue(citation["metadata_complete"])

    def test_missing_source_url_is_flagged_not_invented(self):
        evidence = self.evidence()
        del evidence["metadata"]["source_url"]
        citation = self.module.make_citation(evidence)
        self.assertIsNone(citation["source_url"])
        self.assertIn("missing_or_invalid_source_url", citation["warnings"])

    def test_claim_maps_to_supporting_chunk(self):
        citation = self.module.make_citation(self.evidence())
        answer = {
            "answer_status": "answered",
            "answer": "CDD is required. [E1]",
            "claims": [{"text": "CDD is required.", "evidence_ids": ["E1"]}],
            "conditions_and_exceptions": [], "conflicts": [],
            "regulatory_status_note": "The direction is binding.",
            "evidence_ids_used": ["E1"],
        }
        mappings, errors = self.module.build_claim_mappings(answer, {"E1": citation})
        self.assertFalse(errors)
        claim = next(item for item in mappings if item["claim_id"] == "CLM-001")
        self.assertEqual(claim["supporting_chunk_ids"], ["rbi_doc1__chunk_0047"])
        self.assertTrue(claim["mapping_complete"])

    def test_unknown_evidence_id_causes_mapping_error(self):
        answer = {
            "answer_status": "answered", "answer": "Claim. [E9]",
            "claims": [{"text": "Claim", "evidence_ids": ["E9"]}],
            "conditions_and_exceptions": [], "conflicts": [],
            "regulatory_status_note": "", "evidence_ids_used": ["E9"],
        }
        _, errors = self.module.build_claim_mappings(answer, {})
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
