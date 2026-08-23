from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import PipelinePaths


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        prog="rbi-preprocess",
        description="Run RBI preprocessing and semantic, lexical, or hybrid retrieval stages.",
    )
    command.add_argument("--corpus-dir", required=True, help="Directory containing the original RBI PDFs.")
    command.add_argument("--inventory-file", required=True, help="Path to RBI_Document_Inventory.xlsx.")
    command.add_argument(
        "--project-root",
        help="Hybrid_RAG_Project path. Defaults to the shared parent of the corpus and inventory.",
    )
    command.add_argument(
        "--step",
        choices=("all", "raw", "clean", "hierarchy", "chunks", "index", "bm25", "hybrid", "rerank", "expand", "answer", "cite", "evaluate"),
        default="all",
        help="Run the full pipeline or one stage. Default: all.",
    )
    command.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output folder(s) for the selected stage(s).",
    )
    command.add_argument("--target-tokens", type=int, default=700, help="Preferred chunk size. Default: 700.")
    command.add_argument("--max-tokens", type=int, default=1000, help="Approximate hard maximum. Default: 1000.")
    command.add_argument("--min-tokens", type=int, default=150, help="Preferred minimum. Default: 150.")
    command.add_argument("--overlap-tokens", type=int, default=100, help="Chunk overlap. Default: 100.")
    command.add_argument(
        "--pinecone-index-name",
        help="Existing 512-dimensional cosine Pinecone index. Can also use PINECONE_INDEX_NAME.",
    )
    command.add_argument("--namespace", default="RBI_RAG", help="Pinecone namespace. Default: RBI_RAG.")
    command.add_argument("--retriever-k", type=int, default=5, help="Retriever result count. Default: 5.")
    command.add_argument("--batch-size", type=int, default=100, help="Pinecone upload batch size. Default: 100.")
    command.add_argument("--query", help="Optional verification query for Pinecone or BM25 indexing.")
    command.add_argument("--bm25-k1", type=float, default=1.5, help="BM25 term saturation. Default: 1.5.")
    command.add_argument("--bm25-b", type=float, default=0.75, help="BM25 length normalization. Default: 0.75.")
    command.add_argument("--bm25-top-k", type=int, default=5, help="Optional BM25 query result count. Default: 5.")
    command.add_argument("--pinecone-candidates", type=int, default=20, help="Step-9 semantic candidates. Default: 20.")
    command.add_argument("--bm25-candidates", type=int, default=20, help="Step-9 lexical candidates. Default: 20.")
    command.add_argument("--rrf-constant", type=int, default=60, help="Step-9 RRF rank constant. Default: 60.")
    command.add_argument("--merged-candidates", type=int, default=20, help="Step-9 final pool size. Default: 20.")
    command.add_argument("--rerank-model", default="cohere-rerank-3.5", help="Pinecone hosted reranker. Default: cohere-rerank-3.5.")
    command.add_argument("--rerank-candidates", type=int, default=20, help="Step-11 fused candidates to rerank. Default: 20.")
    command.add_argument("--rerank-top-n", type=int, default=8, help="Step-11 chunks retained for generation. Default: 8.")
    command.add_argument("--llm-model", default="gemma3:4b", help="Step-13 Ollama model. Default: gemma3:4b.")
    command.add_argument("--ollama-host", default="http://localhost:11434", help="Local Ollama API URL.")
    command.add_argument("--llm-context-tokens", type=int, default=32768, help="Step-13 context window. Default: 32768.")
    command.add_argument("--llm-max-output-tokens", type=int, default=1800, help="Step-13 output limit. Default: 1800.")
    command.add_argument("--llm-evidence-token-budget", type=int, default=22000, help="Step-13 approximate evidence budget. Default: 22000.")
    command.add_argument("--llm-temperature", type=float, default=0.1, help="Step-13 temperature. Default: 0.1.")
    return command


def selected_outputs(paths: PipelinePaths, step: str) -> list[Path]:
    if step == "raw":
        return [paths.raw_output_dir]
    if step == "clean":
        return [paths.cleaned_output_dir]
    if step == "hierarchy":
        return [paths.hierarchy_output_dir]
    if step == "chunks":
        return [paths.chunk_output_dir]
    if step == "index":
        return [paths.vector_output_dir]
    if step == "bm25":
        return [paths.bm25_output_dir]
    if step == "hybrid":
        # Step 9 is query-time and safely maintains latest plus query-specific results.
        return []
    if step == "rerank":
        return []
    if step == "expand":
        return []
    if step == "answer":
        return []
    if step == "cite":
        return []
    if step == "evaluate":
        return [paths.evaluation_output_dir]
    return [paths.raw_output_dir, paths.cleaned_output_dir, paths.hierarchy_output_dir, paths.chunk_output_dir]


def validate_stage_inputs(paths: PipelinePaths, step: str) -> None:
    if step == "clean" and not (paths.raw_output_dir / "raw_extractions").is_dir():
        raise FileNotFoundError("Run --step raw first; RBI_Raw_Extraction/raw_extractions is missing.")
    if step == "hierarchy":
        if not (paths.raw_output_dir / "raw_extractions").is_dir():
            raise FileNotFoundError("RBI_Raw_Extraction/raw_extractions is required.")
        if not (paths.cleaned_output_dir / "cleaned_documents").is_dir():
            raise FileNotFoundError("Run --step clean first; cleaned_documents is missing.")
    if step == "chunks" and not (paths.hierarchy_output_dir / "structured_documents").is_dir():
        raise FileNotFoundError("Run --step hierarchy first; structured_documents is missing.")
    if step in ("index", "bm25") and not (paths.chunk_output_dir / "chunks.jsonl").is_file():
        raise FileNotFoundError("Run --step chunks first; RBI_Chunked_Documents/chunks.jsonl is missing.")
    if step == "hybrid":
        if not (paths.bm25_output_dir / "bm25_index.pkl").is_file():
            raise FileNotFoundError("Run --step bm25 first; RBI_BM25_Index/bm25_index.pkl is missing.")
        if not (paths.bm25_output_dir / "chunk_store.jsonl").is_file():
            raise FileNotFoundError("Run --step bm25 first; RBI_BM25_Index/chunk_store.jsonl is missing.")
    if step == "rerank" and not (paths.hybrid_output_dir / "hybrid_candidates.json").is_file():
        raise FileNotFoundError(
            "Run --step hybrid first; RBI_Hybrid_Retrieval/hybrid_candidates.json is missing."
        )
    if step == "expand":
        if not (paths.rerank_output_dir / "reranked_candidates.json").is_file():
            raise FileNotFoundError(
                "Run --step rerank first; RBI_Reranked_Retrieval/reranked_candidates.json is missing."
            )
        if not (paths.chunk_output_dir / "node_hierarchy.jsonl").is_file():
            raise FileNotFoundError(
                "Run --step chunks first; RBI_Chunked_Documents/node_hierarchy.jsonl is missing."
            )
    if step == "answer" and not (paths.context_output_dir / "expanded_context.json").is_file():
        raise FileNotFoundError(
            "Run --step expand first; RBI_Expanded_Context/expanded_context.json is missing."
        )
    if step == "cite":
        if not (paths.answer_output_dir / "latest_answer.json").is_file():
            raise FileNotFoundError(
                "Run --step answer first; RBI_Grounded_Answers/latest_answer.json is missing."
            )
    if step == "evaluate" and not (paths.chunk_output_dir / "chunks.jsonl").is_file():
        raise FileNotFoundError(
            "Run --step chunks first; RBI_Chunked_Documents/chunks.jsonl is missing."
        )


def configure_environment(paths: PipelinePaths, args: argparse.Namespace) -> None:
    os.environ.update(
        {
            "RBI_CORPUS_DIR": str(paths.corpus_dir),
            "RBI_INVENTORY_FILE": str(paths.inventory_file),
            "RBI_RAW_OUTPUT_DIR": str(paths.raw_output_dir),
            "RBI_CLEAN_OUTPUT_DIR": str(paths.cleaned_output_dir),
            "RBI_HIERARCHY_OUTPUT_DIR": str(paths.hierarchy_output_dir),
            "RBI_CHUNK_OUTPUT_DIR": str(paths.chunk_output_dir),
            "RBI_VECTOR_OUTPUT_DIR": str(paths.vector_output_dir),
            "RBI_BM25_OUTPUT_DIR": str(paths.bm25_output_dir),
            "RBI_HYBRID_OUTPUT_DIR": str(paths.hybrid_output_dir),
            "RBI_RERANK_OUTPUT_DIR": str(paths.rerank_output_dir),
            "RBI_CONTEXT_OUTPUT_DIR": str(paths.context_output_dir),
            "RBI_ANSWER_OUTPUT_DIR": str(paths.answer_output_dir),
            "RBI_CITATION_OUTPUT_DIR": str(paths.citation_output_dir),
            "RBI_EVALUATION_OUTPUT_DIR": str(paths.evaluation_output_dir),
            "RBI_CHUNK_TARGET_TOKENS": str(args.target_tokens),
            "RBI_CHUNK_MAX_TOKENS": str(args.max_tokens),
            "RBI_CHUNK_MIN_TOKENS": str(args.min_tokens),
            "RBI_CHUNK_OVERLAP_TOKENS": str(args.overlap_tokens),
            "RBI_PINECONE_NAMESPACE": str(args.namespace),
            "RBI_RETRIEVER_K": str(args.retriever_k),
            "RBI_VECTOR_BATCH_SIZE": str(args.batch_size),
            "RBI_BM25_K1": str(args.bm25_k1),
            "RBI_BM25_B": str(args.bm25_b),
            "RBI_BM25_TOP_K": str(args.bm25_top_k),
            "RBI_HYBRID_PINECONE_K": str(args.pinecone_candidates),
            "RBI_HYBRID_BM25_K": str(args.bm25_candidates),
            "RBI_HYBRID_RRF_CONSTANT": str(args.rrf_constant),
            "RBI_HYBRID_MERGED_K": str(args.merged_candidates),
            "RBI_RERANK_MODEL": str(args.rerank_model),
            "RBI_RERANK_INPUT_K": str(args.rerank_candidates),
            "RBI_RERANK_TOP_N": str(args.rerank_top_n),
            "RBI_LLM_MODEL": str(args.llm_model),
            "OLLAMA_HOST": str(args.ollama_host),
            "RBI_LLM_CONTEXT_TOKENS": str(args.llm_context_tokens),
            "RBI_LLM_MAX_OUTPUT_TOKENS": str(args.llm_max_output_tokens),
            "RBI_LLM_EVIDENCE_TOKEN_BUDGET": str(args.llm_evidence_token_budget),
            "RBI_LLM_TEMPERATURE": str(args.llm_temperature),
        }
    )
    if args.pinecone_index_name:
        os.environ["PINECONE_INDEX_NAME"] = args.pinecone_index_name
    if args.query:
        os.environ["RBI_RETRIEVER_QUERY"] = args.query
        os.environ["RBI_BM25_QUERY"] = args.query
        os.environ["RBI_HYBRID_QUERY"] = args.query
        os.environ["RBI_RERANK_QUERY"] = args.query
        os.environ["RBI_CONTEXT_QUERY"] = args.query
        os.environ["RBI_ANSWER_QUERY"] = args.query
        os.environ["RBI_CITATION_QUERY"] = args.query


def run(paths: PipelinePaths, step: str, args: argparse.Namespace) -> None:
    configure_environment(paths, args)
    # Import after setting paths: step modules resolve their configured paths at import time.
    if step in ("all", "raw"):
        from .steps import extract_raw

        extract_raw.main()
    if step in ("all", "clean"):
        from .steps import clean_extracted_text

        clean_extracted_text.main()
    if step in ("all", "hierarchy"):
        from .steps import build_document_hierarchy

        build_document_hierarchy.main()
    if step in ("all", "chunks"):
        from .steps import chunk_regulatory_documents

        chunk_regulatory_documents.main()
    if step == "index":
        from .steps import index_pinecone

        index_pinecone.main()
    if step == "bm25":
        from .steps import build_bm25_index

        build_bm25_index.main()
    if step == "hybrid":
        from .steps import hybrid_retrieval

        hybrid_retrieval.main()
    if step == "rerank":
        from .steps import rerank_candidates

        rerank_candidates.main()
    if step == "expand":
        from .steps import expand_parent_context

        expand_parent_context.main()
    if step == "answer":
        from .steps import generate_grounded_answer

        generate_grounded_answer.main()
    if step == "cite":
        from .steps import generate_citations

        generate_citations.main()
    if step == "evaluate":
        from .steps import build_evaluation_dataset

        build_evaluation_dataset.main()


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    paths = PipelinePaths.resolve(args.corpus_dir, args.inventory_file, args.project_root)
    pdfs = paths.validate_inputs()
    validate_stage_inputs(paths, args.step)
    existing = [path for path in selected_outputs(paths, args.step) if path.exists()]
    if existing and not args.overwrite:
        joined = "\n".join(f"  - {path}" for path in existing)
        raise SystemExit(f"Output folder(s) already exist:\n{joined}\nUse --overwrite to replace them.")
    run(paths, args.step, args)
    print(
        json.dumps(
            {
                "status": "completed",
                "step": args.step,
                "pdf_count": len(pdfs),
                "project_root": str(paths.project_root),
                "raw_output": str(paths.raw_output_dir),
                "cleaned_output": str(paths.cleaned_output_dir),
                "hierarchy_output": str(paths.hierarchy_output_dir),
                "chunk_output": str(paths.chunk_output_dir),
                "vector_output": str(paths.vector_output_dir),
                "bm25_output": str(paths.bm25_output_dir),
                "hybrid_output": str(paths.hybrid_output_dir),
                "rerank_output": str(paths.rerank_output_dir),
                "context_output": str(paths.context_output_dir),
                "answer_output": str(paths.answer_output_dir),
                "citation_output": str(paths.citation_output_dir),
                "evaluation_output": str(paths.evaluation_output_dir),
            },
            indent=2,
        )
    )
