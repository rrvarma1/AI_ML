from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rbi_rag_preprocessing.config import PipelinePaths


class PipelinePathTests(unittest.TestCase):
    def test_shared_parent_is_project_root(self):
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            corpus = tmp_path / "RBI_Document_Corpus"
            corpus.mkdir()
            inventory = tmp_path / "RBI_Document_Inventory.xlsx"
            inventory.touch()
            paths = PipelinePaths.resolve(corpus, inventory)
            self.assertEqual(paths.project_root, tmp_path)
            self.assertEqual(paths.raw_output_dir, tmp_path / "RBI_Raw_Extraction")
            self.assertEqual(paths.cleaned_output_dir, tmp_path / "RBI_Cleaned_Extraction")
            self.assertEqual(paths.chunk_output_dir, tmp_path / "RBI_Chunked_Documents")
            self.assertEqual(paths.vector_output_dir, tmp_path / "RBI_Vector_Indexing")
            self.assertEqual(paths.bm25_output_dir, tmp_path / "RBI_BM25_Index")
            self.assertEqual(paths.hybrid_output_dir, tmp_path / "RBI_Hybrid_Retrieval")
            self.assertEqual(paths.rerank_output_dir, tmp_path / "RBI_Reranked_Retrieval")
            self.assertEqual(paths.context_output_dir, tmp_path / "RBI_Expanded_Context")
            self.assertEqual(paths.answer_output_dir, tmp_path / "RBI_Grounded_Answers")
            self.assertEqual(paths.citation_output_dir, tmp_path / "RBI_Citations")

    def test_different_parents_require_explicit_root(self):
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            corpus = tmp_path / "a" / "RBI_Document_Corpus"
            corpus.mkdir(parents=True)
            inventory = tmp_path / "b" / "RBI_Document_Inventory.xlsx"
            inventory.parent.mkdir()
            inventory.touch()
            with self.assertRaises(ValueError):
                PipelinePaths.resolve(corpus, inventory)


if __name__ == "__main__":
    unittest.main()
