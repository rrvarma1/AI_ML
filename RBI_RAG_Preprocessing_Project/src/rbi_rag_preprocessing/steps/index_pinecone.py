"""Step 7: embed final chunks and index them in an existing Pinecone index."""
from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CHUNK_ROOT = Path(os.environ["RBI_CHUNK_OUTPUT_DIR"]).expanduser().resolve()
CHUNKS_FILE = CHUNK_ROOT / "chunks.jsonl"
OUT = Path(os.environ["RBI_VECTOR_OUTPUT_DIR"]).expanduser().resolve()
NAMESPACE = os.getenv("RBI_PINECONE_NAMESPACE", "RBI_RAG")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "").strip()
EMBEDDING_MODEL = "llama-text-embed-v2"
EMBEDDING_DIMENSIONS = 512
EMBEDDING_BATCH_SIZE = int(os.getenv("RBI_EMBEDDING_BATCH_SIZE", "96"))
BATCH_SIZE = int(os.getenv("RBI_VECTOR_BATCH_SIZE", "100"))
RETRIEVER_K = int(os.getenv("RBI_RETRIEVER_K", "5"))
SEARCH_QUERY = os.getenv("RBI_RETRIEVER_QUERY", "").strip()


def require_setting(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. Add it to the project-root .env file or export it in Terminal."
        )
    return value


def serializable_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Convert chunk metadata to Pinecone's supported flat scalar/list format."""
    cleaned: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            cleaned[key] = value
        elif isinstance(value, list):
            if all(isinstance(item, str) for item in value):
                cleaned[key] = value
            else:
                cleaned[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        elif isinstance(value, dict):
            cleaned[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        else:
            cleaned[key] = str(value)
    return cleaned


def load_chunks(document_cls) -> tuple[list[Any], list[str], str]:
    if not CHUNKS_FILE.is_file():
        raise FileNotFoundError(f"Chunk ingestion file not found: {CHUNKS_FILE}")
    documents, ids = [], []
    first_document_id = ""
    with CHUNKS_FILE.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            content, metadata = record.get("content"), record.get("metadata")
            if not isinstance(content, str) or not content.strip() or not isinstance(metadata, dict):
                raise ValueError(f"Invalid content/metadata at {CHUNKS_FILE}:{line_number}")
            chunk_id = str(metadata.get("chunk_id") or "").strip()
            document_id = str(metadata.get("document_id") or "").strip()
            if not chunk_id or not document_id:
                raise ValueError(f"Missing chunk_id/document_id at {CHUNKS_FILE}:{line_number}")
            if not first_document_id:
                first_document_id = document_id
            documents.append(document_cls(page_content=content, metadata=serializable_metadata(metadata)))
            ids.append(chunk_id)
    if not documents:
        raise ValueError(f"No chunks found in {CHUNKS_FILE}")
    if len(ids) != len(set(ids)):
        raise ValueError("Chunk IDs are not unique; refusing to overwrite ambiguous Pinecone vectors.")
    return documents, ids, first_document_id


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


def fetch_contains(index, vector_id: str, attempts: int = 6) -> bool:
    """Allow for Pinecone's eventual consistency before failing verification."""
    for attempt in range(attempts):
        response = index.fetch(ids=[vector_id], namespace=NAMESPACE)
        vectors = model_field(response, "vectors") or {}
        if vector_id in vectors:
            return True
        if attempt + 1 < attempts:
            time.sleep(2)
    return False


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    from dotenv import load_dotenv

    project_root = CHUNK_ROOT.parent
    software_root = Path(__file__).resolve().parents[3]
    # Support an existing .env in either the Hybrid_RAG_Project parent or software folder.
    load_dotenv(project_root / ".env")
    load_dotenv(software_root / ".env")
    pinecone_api_key = require_setting("PINECONE_API_KEY")
    index_name = INDEX_NAME or require_setting("PINECONE_INDEX_NAME")

    from langchain_core.documents import Document
    from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
    from pinecone import Pinecone

    documents, vector_ids, first_document_id = load_chunks(Document)
    first_document_vector_ids = [
        vector_id
        for document, vector_id in zip(documents, vector_ids)
        if document.metadata["document_id"] == first_document_id
    ]

    embeddings = PineconeEmbeddings(
        model=EMBEDDING_MODEL,
        pinecone_api_key=pinecone_api_key,
        document_params={
            "input_type": "passage",
            "truncate": "END",
            "dimension": EMBEDDING_DIMENSIONS,
        },
        query_params={
            "input_type": "query",
            "truncate": "END",
            "dimension": EMBEDDING_DIMENSIONS,
        },
        batch_size=EMBEDDING_BATCH_SIZE,
    )
    pc = Pinecone(api_key=pinecone_api_key)
    index_description = verify_existing_index(pc, index_name)
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=NAMESPACE,
    )

    for start in range(0, len(documents), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(documents))
        vector_store.add_documents(documents=documents[start:stop], ids=vector_ids[start:stop])
        print(f"Indexed chunks: {stop}/{len(documents)}")

    first_vector_id = first_document_vector_ids[0]
    verified = fetch_contains(index, first_vector_id)
    if not verified:
        raise RuntimeError(f"Pinecone verification fetch did not return vector {first_vector_id}")

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )
    search_results = []
    if SEARCH_QUERY:
        for rank, result in enumerate(retriever.invoke(SEARCH_QUERY), start=1):
            search_results.append({
                "rank": rank,
                "chunk_id": result.metadata.get("chunk_id"),
                "document_id": result.metadata.get("document_id"),
                "title": result.metadata.get("title"),
                "page_start": result.metadata.get("page_start"),
                "page_end": result.metadata.get("page_end"),
                "content_preview": result.page_content[:500],
            })

    report = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "pinecone_index": index_description,
        "namespace": NAMESPACE,
        "embedding": {
            "provider": "Pinecone Inference",
            "model": EMBEDDING_MODEL,
            "dimensions": EMBEDDING_DIMENSIONS,
            "document_input_type": "passage",
            "query_input_type": "query",
        },
        "chunks_indexed": len(documents),
        "retriever": {"type": "vector_similarity", "k": RETRIEVER_K},
        "first_document_id": first_document_id,
        "first_document_vector_count": len(first_document_vector_ids),
        "first_document_vector_ids": first_document_vector_ids,
        "first_vector_id_verified": verified,
        "query": SEARCH_QUERY or None,
        "search_results": search_results,
    }
    (OUT / "pinecone_indexing_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.copy2(Path(__file__), OUT / "index_pinecone.py")

    print("-" * 100)
    print(f"Documents/chunks indexed: {len(documents)}")
    print(f"Pinecone namespace:       {NAMESPACE}")
    print(f"First document:           {first_document_id}")
    print(f"First document vectors:   {len(first_document_vector_ids)}")
    print(f"First vector ID:          {first_vector_id}")
    print(f"Verification fetch:       {'PASSED' if verified else 'FAILED'}")
    if search_results:
        print("Semantic search results:")
        for result in search_results:
            print(f"  {result['rank']}. {result['chunk_id']} | {result['title']}")


if __name__ == "__main__":
    main()
