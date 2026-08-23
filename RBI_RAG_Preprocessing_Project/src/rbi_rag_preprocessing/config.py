from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelinePaths:
    corpus_dir: Path
    inventory_file: Path
    project_root: Path
    raw_output_dir: Path
    cleaned_output_dir: Path
    hierarchy_output_dir: Path
    chunk_output_dir: Path
    vector_output_dir: Path
    bm25_output_dir: Path
    hybrid_output_dir: Path
    rerank_output_dir: Path
    context_output_dir: Path
    answer_output_dir: Path
    citation_output_dir: Path
    evaluation_output_dir: Path

    @classmethod
    def resolve(
        cls,
        corpus_dir: str | Path,
        inventory_file: str | Path,
        project_root: str | Path | None = None,
    ) -> "PipelinePaths":
        corpus = Path(corpus_dir).expanduser().resolve()
        inventory = Path(inventory_file).expanduser().resolve()
        if project_root is None:
            if corpus.parent != inventory.parent:
                raise ValueError(
                    "Corpus and inventory do not share a parent. Supply --project-root explicitly."
                )
            root = corpus.parent
        else:
            root = Path(project_root).expanduser().resolve()
        return cls(
            corpus_dir=corpus,
            inventory_file=inventory,
            project_root=root,
            raw_output_dir=root / "RBI_Raw_Extraction",
            cleaned_output_dir=root / "RBI_Cleaned_Extraction",
            hierarchy_output_dir=root / "RBI_Document_Hierarchy",
            chunk_output_dir=root / "RBI_Chunked_Documents",
            vector_output_dir=root / "RBI_Vector_Indexing",
            bm25_output_dir=root / "RBI_BM25_Index",
            hybrid_output_dir=root / "RBI_Hybrid_Retrieval",
            rerank_output_dir=root / "RBI_Reranked_Retrieval",
            context_output_dir=root / "RBI_Expanded_Context",
            answer_output_dir=root / "RBI_Grounded_Answers",
            citation_output_dir=root / "RBI_Citations",
            evaluation_output_dir=root / "RBI_Evaluation_Dataset",
        )

    def validate_inputs(self) -> list[Path]:
        if not self.corpus_dir.is_dir():
            raise FileNotFoundError(f"Corpus directory not found: {self.corpus_dir}")
        if not self.inventory_file.is_file():
            raise FileNotFoundError(f"Inventory workbook not found: {self.inventory_file}")
        if self.inventory_file.suffix.lower() != ".xlsx":
            raise ValueError("The inventory must be an .xlsx workbook.")
        pdfs = sorted(self.corpus_dir.glob("*.pdf"))
        if not pdfs:
            raise ValueError(f"No PDF files found in: {self.corpus_dir}")
        self.project_root.mkdir(parents=True, exist_ok=True)
        return pdfs
