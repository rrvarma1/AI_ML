"""Step 9: retrieve from Pinecone and BM25, then fuse candidates with RRF."""
from __future__ import annotations

import json
import hashlib
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BM25_ROOT = Path(os.environ["RBI_BM25_OUTPUT_DIR"]).expanduser().resolve()
INDEX_FILE = BM25_ROOT / "bm25_index.pkl"
CHUNK_STORE_FILE = BM25_ROOT / "chunk_store.jsonl"
OUT = Path(os.environ["RBI_HYBRID_OUTPUT_DIR"]).expanduser().resolve()
NAMESPACE = os.getenv("RBI_PINECONE_NAMESPACE", "RBI_RAG")
QUERY = os.getenv("RBI_HYBRID_QUERY", "").strip()
PINECONE_CANDIDATES = int(os.getenv("RBI_HYBRID_PINECONE_K", "20"))
BM25_CANDIDATES = int(os.getenv("RBI_HYBRID_BM25_K", "20"))
RRF_CONSTANT = int(os.getenv("RBI_HYBRID_RRF_CONSTANT", "60"))
MERGED_CANDIDATES = int(os.getenv("RBI_HYBRID_MERGED_K", "20"))
EMBEDDING_MODEL = "llama-text-embed-v2"
EMBEDDING_DIMENSIONS = 512


def require_setting(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. Add it to the project-root .env file or export it in Terminal."
        )
    return value


def model_field(model: Any, name: str) -> Any:
    if isinstance(model, dict):
        return model.get(name)
    return getattr(model, name, None)


def verify_existing_index(pc, index_name: str) -> dict[str, Any]:
    description = pc.describe_index(index_name)
    dimension = int(model_field(description, "dimension") or 0)
    metric = str(model_field(description, "metric") or "").lower()
    if dimension != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Pinecone index {index_name!r} has dimension {dimension}; expected {EMBEDDING_DIMENSIONS}."
        )
    if metric != "cosine":
        raise ValueError(f"Pinecone index {index_name!r} uses metric {metric!r}; expected 'cosine'.")
    return {"name": index_name, "dimension": dimension, "metric": metric}


def validate_parameters() -> None:
    values = {
        "Pinecone candidates": PINECONE_CANDIDATES,
        "BM25 candidates": BM25_CANDIDATES,
        "RRF constant": RRF_CONSTANT,
        "merged candidates": MERGED_CANDIDATES,
    }
    for label, value in values.items():
        if value <= 0:
            raise ValueError(f"{label} must be greater than zero; got {value}.")


def load_bm25_artifacts() -> tuple[Any, list[dict[str, Any]]]:
    if not INDEX_FILE.is_file() or not CHUNK_STORE_FILE.is_file():
        raise FileNotFoundError(
            "Run --step bm25 first; bm25_index.pkl and chunk_store.jsonl are required."
        )
    records: list[dict[str, Any]] = []
    with CHUNK_STORE_FILE.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("chunk_id") or not isinstance(record.get("tokenized_text"), list):
                raise ValueError(f"Invalid BM25 chunk record at {CHUNK_STORE_FILE}:{line_number}")
            records.append(record)

    # This is a trusted artifact created locally by Step 8. Never load arbitrary pickle files.
    with INDEX_FILE.open("rb") as stream:
        artifact = pickle.load(stream)
    if not isinstance(artifact, dict) or artifact.get("format_version") != 1:
        raise ValueError(f"Unsupported BM25 artifact format: {INDEX_FILE}")
    artifact_ids = [str(value) for value in artifact.get("chunk_ids", [])]
    store_ids = [str(record["chunk_id"]) for record in records]
    if artifact_ids != store_ids:
        raise ValueError("BM25 index and chunk_store.jsonl chunk IDs are not aligned.")
    return artifact["index"], records


def min_max_normalize(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        return []
    scores = [float(candidate["raw_score"]) for candidate in candidates]
    low, high = min(scores), max(scores)
    normalized = []
    for candidate in candidates:
        result = dict(candidate)
        result["normalized_score"] = (
            1.0 if high == low else (float(candidate["raw_score"]) - low) / (high - low)
        )
        normalized.append(result)
    return normalized


def lexical_candidates(index, records: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    from .build_bm25_index import tokenize

    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    scores = index.get_scores(query_tokens)
    positions = sorted(range(len(records)), key=lambda pos: (-float(scores[pos]), pos))
    results = []
    for position in positions:
        score = float(scores[position])
        # Zero-score documents contain none of the query terms and should not gain an RRF vote.
        if score <= 0:
            continue
        record = records[position]
        results.append({
            "chunk_id": str(record["chunk_id"]),
            "raw_score": score,
            "content": record["content"],
            "metadata": record["metadata"],
        })
        if len(results) == BM25_CANDIDATES:
            break
    return min_max_normalize(results)


def semantic_candidates(vector_store, query: str) -> list[dict[str, Any]]:
    results = []
    for document, score in vector_store.similarity_search_with_score(
        query=query, k=PINECONE_CANDIDATES
    ):
        chunk_id = str(document.metadata.get("chunk_id") or "").strip()
        if not chunk_id:
            raise ValueError("A Pinecone result is missing chunk_id metadata.")
        results.append({
            "chunk_id": chunk_id,
            "raw_score": float(score),
            "content": document.page_content,
            "metadata": document.metadata,
        })
    return min_max_normalize(results)


def reciprocal_rank_fusion(
    semantic: list[dict[str, Any]],
    lexical: list[dict[str, Any]],
    rrf_constant: int,
    top_k: int,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for source_name, candidates in (("semantic", semantic), ("lexical", lexical)):
        for rank, candidate in enumerate(candidates, start=1):
            chunk_id = candidate["chunk_id"]
            entry = merged.setdefault(chunk_id, {
                "chunk_id": chunk_id,
                "content": candidate["content"],
                "metadata": candidate["metadata"],
                "semantic": None,
                "lexical": None,
                "rrf_score": 0.0,
            })
            # Prefer the local Step-8 content/metadata when a chunk occurs in both lists.
            if source_name == "lexical":
                entry["content"] = candidate["content"]
                entry["metadata"] = candidate["metadata"]
            entry[source_name] = {
                "rank": rank,
                "raw_score": candidate["raw_score"],
                "normalized_score": candidate["normalized_score"],
            }
            entry["rrf_score"] += 1.0 / (rrf_constant + rank)

    for entry in merged.values():
        available = [
            source["normalized_score"]
            for source in (entry["semantic"], entry["lexical"])
            if source is not None
        ]
        entry["normalized_score_mean"] = sum(available) / len(available)
        entry["matched_by"] = [
            name for name in ("semantic", "lexical") if entry[name] is not None
        ]

    ranked = sorted(
        merged.values(),
        key=lambda item: (-item["rrf_score"], -item["normalized_score_mean"], item["chunk_id"]),
    )
    output = []
    for rank, item in enumerate(ranked[:top_k], start=1):
        item["rank"] = rank
        output.append(item)
    return output


def main() -> None:
    validate_parameters()
    if not QUERY:
        raise ValueError("Step 9 requires --query with a non-empty question.")

    from dotenv import load_dotenv

    project_root = BM25_ROOT.parent
    software_root = Path(__file__).resolve().parents[3]
    load_dotenv(project_root / ".env")
    load_dotenv(software_root / ".env")
    pinecone_api_key = require_setting("PINECONE_API_KEY")
    index_name = require_setting("PINECONE_INDEX_NAME")

    from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
    from pinecone import Pinecone

    bm25_index, chunk_records = load_bm25_artifacts()
    pc = Pinecone(api_key=pinecone_api_key)
    index_description = verify_existing_index(pc, index_name)
    embeddings = PineconeEmbeddings(
        model=EMBEDDING_MODEL,
        pinecone_api_key=pinecone_api_key,
        document_params={"input_type": "passage", "truncate": "END", "dimension": 512},
        query_params={"input_type": "query", "truncate": "END", "dimension": 512},
        batch_size=96,
    )
    vector_store = PineconeVectorStore(
        index=pc.Index(index_name), embedding=embeddings, namespace=NAMESPACE
    )

    semantic = semantic_candidates(vector_store, QUERY)
    lexical = lexical_candidates(bm25_index, chunk_records, QUERY)
    candidates = reciprocal_rank_fusion(
        semantic, lexical, RRF_CONSTANT, MERGED_CANDIDATES
    )

    OUT.mkdir(parents=True, exist_ok=True)
    query_output_dir = OUT / "queries"
    query_output_dir.mkdir(exist_ok=True)
    result = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "query": QUERY,
        "fusion_method": "reciprocal_rank_fusion",
        "parameters": {
            "pinecone_candidates": PINECONE_CANDIDATES,
            "bm25_candidates": BM25_CANDIDATES,
            "rrf_constant": RRF_CONSTANT,
            "merged_candidates": MERGED_CANDIDATES,
        },
        "pinecone": {
            "index": index_description,
            "namespace": NAMESPACE,
            "embedding_model": EMBEDDING_MODEL,
            "returned_candidates": len(semantic),
        },
        "bm25": {
            "index_file": str(INDEX_FILE),
            "indexed_chunks": len(chunk_records),
            "returned_nonzero_candidates": len(lexical),
        },
        "normalization": "min_max_per_retriever",
        "merged_unique_candidates_before_limit": len(
            {item["chunk_id"] for item in semantic + lexical}
        ),
        "returned_candidates": len(candidates),
        "candidates": candidates,
    }
    (OUT / "hybrid_candidates.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    query_id = hashlib.sha256(QUERY.encode("utf-8")).hexdigest()[:16]
    (query_output_dir / f"{query_id}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "hybrid_retrieval_report.json").write_text(
        json.dumps({key: value for key, value in result.items() if key != "candidates"},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("-" * 100)
    print(f"Question:                  {QUERY}")
    print(f"Pinecone candidates:       {len(semantic)}/{PINECONE_CANDIDATES}")
    print(f"BM25 non-zero candidates:  {len(lexical)}/{BM25_CANDIDATES}")
    print(f"Merged candidates:         {len(candidates)}/{MERGED_CANDIDATES}")
    print(f"RRF constant:              {RRF_CONSTANT}")
    print(f"Saved query ID:            {query_id}")
    print("Top hybrid candidates:")
    for candidate in candidates:
        title = candidate["metadata"].get("title") or "Untitled"
        print(
            f"  {candidate['rank']}. {candidate['chunk_id']} | "
            f"RRF={candidate['rrf_score']:.8f} | {title}"
        )


if __name__ == "__main__":
    main()
