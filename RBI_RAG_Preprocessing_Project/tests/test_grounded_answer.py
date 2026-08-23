import importlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


class GroundedAnswerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory()
        root = Path(cls.temp.name)
        os.environ["RBI_CONTEXT_OUTPUT_DIR"] = str(root / "expanded")
        os.environ["RBI_ANSWER_OUTPUT_DIR"] = str(root / "answers")
        cls.module = importlib.import_module(
            "rbi_rag_preprocessing.steps.generate_grounded_answer"
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def evidence():
        return [{
            "rank": 1, "chunk_id": "doc1__chunk_0001", "reranker_score": 0.9,
            "expanded_content": "The regulated entity shall identify its customers.",
            "metadata": {
                "title": "Master Direction - KYC", "document_id": "doc1",
                "heading_path": ["Master Direction - KYC", "Section 33"],
                "page_start": 42, "page_end": 43,
                "regulatory_status": "binding_direction",
            },
            "expansion": {"legal_dependency_detected": False},
        }]

    def test_evidence_package_assigns_deterministic_ids(self):
        selected, omitted, text = self.module.build_evidence_package(self.evidence(), 1000)
        self.assertEqual(selected[0]["evidence_id"], "E1")
        self.assertFalse(omitted)
        self.assertIn("[E1]", text)
        self.assertIn("Section 33", text)

    def test_valid_grounded_answer_passes(self):
        answer = {
            "answer_status": "answered",
            "answer": "The entity shall identify customers. [E1]",
            "conditions_and_exceptions": [],
            "claims": [{"text": "Identification is mandatory.", "evidence_ids": ["E1"]}],
            "conflicts": [],
            "regulatory_status_note": "The evidence marks this as binding.",
            "evidence_ids_used": ["E1"],
        }
        self.assertEqual(self.module.validate_answer(answer, {"E1"}), [])

    def test_invented_citation_is_rejected(self):
        answer = {
            "answer_status": "answered", "answer": "Requirement. [E9]",
            "conditions_and_exceptions": [],
            "claims": [{"text": "Requirement", "evidence_ids": ["E9"]}],
            "conflicts": [], "regulatory_status_note": "Binding",
            "evidence_ids_used": ["E9"],
        }
        errors = self.module.validate_answer(answer, {"E1"})
        self.assertTrue(any("unknown" in error.lower() or "invented" in error.lower() for error in errors))

    def test_citation_is_rendered_from_metadata(self):
        selected, _, _ = self.module.build_evidence_package(self.evidence(), 1000)
        citation = self.module.citation_for(selected[0])
        self.assertIn("Master Direction - KYC", citation)
        self.assertIn("Section 33", citation)
        self.assertIn("pages 42–43", citation)

    def test_generation_uses_structured_schema(self):
        valid = {
            "answer_status": "answered",
            "answer": "The entity shall identify customers. [E1]",
            "conditions_and_exceptions": [],
            "claims": [{"text": "Identification is mandatory.", "evidence_ids": ["E1"]}],
            "conflicts": [], "regulatory_status_note": "Binding",
            "evidence_ids_used": ["E1"],
        }

        class Client:
            def chat(self, **kwargs):
                self.kwargs = kwargs
                return {"message": {"content": json.dumps(valid)}, "eval_count": 20}

        client = Client()
        answer, attempts, errors = self.module.generate_validated_answer(
            client, "Question?", "[E1] Evidence", {"E1"}
        )
        self.assertFalse(errors)
        self.assertEqual(answer["answer_status"], "answered")
        self.assertEqual(len(attempts), 1)
        self.assertEqual(client.kwargs["format"], self.module.ANSWER_SCHEMA)

    def test_gemma_answer_missing_schema_fields_is_grounded_and_repaired(self):
        evidence = """[E1]
Chunk ID: rbi_doc7__chunk_0210

Regulated entities shall maintain all necessary records of domestic and international
transactions for at least five years from the date of transaction."""
        raw = {
            "answer": (
                "Regulated entities must maintain necessary records of domestic and "
                "international transactions for at least five years from the transaction date."
            )
        }

        class Client:
            def chat(self, **kwargs):
                return {"message": {"content": "```json\n" + json.dumps(raw) + "\n```"}}

        answer, attempts, errors = self.module.generate_validated_answer(
            Client(), "How long?", evidence, {"E1"}
        )
        self.assertFalse(errors)
        self.assertEqual(answer["answer_status"], "answered")
        self.assertIn("[E1]", answer["answer"])
        self.assertEqual(answer["evidence_ids_used"], ["E1"])
        self.assertTrue(answer["claims"])
        self.assertIn(
            "replaced_model_citations_with_verified_citations",
            attempts[0]["repair_actions"],
        )

    def test_unsupported_numeric_claim_is_not_auto_cited(self):
        blocks = {"E1": "Records shall be maintained for at least five years."}
        ids = self.module.supporting_evidence_ids(
            "Records must be maintained for ten years.", blocks
        )
        self.assertEqual(ids, [])

    def test_parser_accepts_prose_around_json(self):
        parsed = self.module.parse_json_object(
            'Here is the answer: {"answer": "Grounded response"} Thanks.'
        )
        self.assertEqual(parsed["answer"], "Grounded response")

    def test_loose_inline_citation_is_canonicalized(self):
        self.assertEqual(
            self.module.canonicalize_inline_citations(
                "Records are retained for five years (Evidence E 1).", {"E1"}
            ),
            "Records are retained for five years [E1].",
        )

    def test_answer_inherits_verified_citation_from_overlapping_claim(self):
        claims = [{
            "text": "Transaction records must be retained for at least five years.",
            "evidence_ids": ["E1"],
        }]
        ids = self.module.answer_supported_by_grounded_claims(
            "The transaction-record retention period is five years.", claims
        )
        self.assertEqual(ids, ["E1"])

    def test_answer_does_not_inherit_citation_for_unrelated_summary(self):
        claims = [{
            "text": "Transaction records must be retained for at least five years.",
            "evidence_ids": ["E1"],
        }]
        ids = self.module.answer_supported_by_grounded_claims(
            "Banks may disregard customer identification requirements.", claims
        )
        self.assertEqual(ids, [])

    def test_uncitable_summary_is_rebuilt_from_grounded_claims(self):
        answer = {
            "answer_status": "answered",
            "answer": "A vague summary that cannot be matched.",
            "conditions_and_exceptions": [],
            "claims": [{
                "text": "Records must be maintained for at least five years.",
                "evidence_ids": ["E1"],
            }],
            "conflicts": [],
            "regulatory_status_note": "",
            "evidence_ids_used": ["E1"],
        }
        rebuilt = self.module.rebuild_answer_from_grounded_claims(answer)
        self.assertEqual(
            rebuilt["answer"],
            "Records must be maintained for at least five years. [E1]",
        )
        self.assertEqual(self.module.validate_answer(rebuilt, {"E1"}), [])

    def test_generation_rebuilds_summary_when_only_inline_citation_is_missing(self):
        evidence = """[E1]
Chunk ID: rbi_doc7__chunk_0210

Records must be maintained for at least five years from the transaction date."""
        raw = {
            "answer_status": "answered",
            "answer": "The applicable retention requirement is established.",
            "conditions_and_exceptions": [],
            "claims": [{
                "text": "Records must be maintained for at least five years from the transaction date.",
                "evidence_ids": ["E1"],
            }],
            "conflicts": [],
            "regulatory_status_note": "",
            "evidence_ids_used": ["E1"],
        }

        class Client:
            def chat(self, **kwargs):
                return {"message": {"content": json.dumps(raw)}}

        answer, attempts, errors = self.module.generate_validated_answer(
            Client(), "How long?", evidence, {"E1"}
        )
        self.assertFalse(errors)
        self.assertIn("five years", answer["answer"])
        self.assertIn("[E1]", answer["answer"])
        self.assertIn(
            "rebuilt_answer_from_verified_grounded_claims",
            attempts[0]["repair_actions"],
        )

    def test_model_inline_id_is_replaced_by_independently_verified_id(self):
        evidence = """[E1]
Chunk ID: irrelevant

This paragraph concerns a different requirement.

[E2]
Chunk ID: beneficial-owner

For an unincorporated association, the beneficial owner is the natural person
with ownership of more than fifteen percent of property, capital, or profits."""
        raw = {
            "answer_status": "answered",
            "answer": (
                "The beneficial owner is the natural person with ownership of more than "
                "fifteen percent of property, capital, or profits. [E1]"
            ),
            "conditions_and_exceptions": [],
            "claims": [{
                "text": "The threshold is more than fifteen percent.",
                "evidence_ids": ["E2"],
            }],
            "conflicts": [], "regulatory_status_note": "",
            "evidence_ids_used": ["E1", "E2"],
        }
        normalized, actions = self.module.normalize_and_ground_answer(
            raw, evidence, {"E1", "E2"}
        )
        self.assertIn("[E2]", normalized["answer"])
        self.assertNotIn("[E1]", normalized["answer"])
        self.assertEqual(normalized["evidence_ids_used"], ["E2"])
        self.assertIn("replaced_model_citations_with_verified_citations", actions)

    def test_partial_status_is_reconciled_when_answer_is_fully_grounded(self):
        evidence = """[E1]
Records must be maintained for five years from the transaction date."""
        raw = {
            "answer_status": "partially_answered",
            "answer": "Records must be maintained for five years from the transaction date.",
            "claims": [{
                "text": "Records must be maintained for five years from the transaction date.",
                "evidence_ids": ["E1"],
            }],
            "conditions_and_exceptions": [], "conflicts": [],
            "regulatory_status_note": "",
        }
        normalized, actions = self.module.normalize_and_ground_answer(raw, evidence, {"E1"})
        self.assertEqual(normalized["answer_status"], "answered")
        self.assertIn("reconciled_fully_grounded_status_to_answered", actions)


if __name__ == "__main__":
    unittest.main()
