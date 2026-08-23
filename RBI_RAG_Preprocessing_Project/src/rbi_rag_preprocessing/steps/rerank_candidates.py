"""Step 11: rerank Step-9 hybrid candidates with a Pinecone-hosted model."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HYBRID_ROOT = Path(os.environ["RBI_HYBRID_OUTPUT_DIR"]).expanduser().resolve()
HYBRID_RESULTS_FILE = HYBRID_ROOT / "hybrid_candidates.json"
OUT = Path(os.environ["RBI_RERANK_OUTPUT_DIR"]).expanduser().resolve()
QUERY_OVERRIDE = os.getenv("RBI_RERANK_QUERY", "").strip()
RERANK_MODEL = os.getenv("RBI_RERANK_MODEL", "cohere-rerank-3.5").strip()
INPUT_CANDIDATES = int(os.getenv("RBI_RERANK_INPUT_K", "20"))
TOP_N = int(os.getenv("RBI_RERANK_TOP_N", "8"))


def require_setting(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. Add it to the project-root .env file or export it in Terminal."
        )
    return value


def model_field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    for method_name in ("model_dump", "to_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return jsonable(method())
            except (TypeError, ValueError):
                # Pinecone SDK model implementations differ by release; fall through
                # to public attributes when an advertised converter is unusable.
                pass
    public_attributes = {}
    try:
        public_attributes = {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_") and not callable(item)
        }
    except TypeError:
        pass
    if public_attributes:
        return jsonable(public_attributes)
    rerank_units = getattr(value, "rerank_units", None)
    if rerank_units is not None:
        return {"rerank_units": jsonable(rerank_units)}
    return str(value)


def validate_parameters() -> None:
    if INPUT_CANDIDATES <= 0:
        raise ValueError("Reranker input candidate count must be greater than zero.")
    if TOP_N <= 0:
        raise ValueError("Reranker top-N must be greater than zero.")
    if not RERANK_MODEL:
        raise ValueError("Reranker model cannot be empty.")


def load_hybrid_candidates(path: Path) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            "Run --step hybrid first; RBI_Hybrid_Retrieval/hybrid_candidates.json is missing."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    query = str(payload.get("query") or "").strip()
    candidates = payload.get("candidates")
    if not query or not isinstance(candidates, list) or not candidates:
        raise ValueError(f"Invalid Step-9 hybrid result: {path}")
    seen_ids: set[str] = set()
    for position, candidate in enumerate(candidates, start=1):
        chunk_id = str(candidate.get("chunk_id") or "").strip()
        if not chunk_id or not isinstance(candidate.get("content"), str):
            raise ValueError(f"Invalid hybrid candidate at position {position}")
        if chunk_id in seen_ids:
            raise ValueError(f"Duplicate hybrid chunk_id: {chunk_id}")
        seen_ids.add(chunk_id)
    return query, candidates, payload


def rerank_text(candidate: dict[str, Any]) -> str:
    metadata = candidate.get("metadata") or {}
    title = str(metadata.get("title") or "").strip()
    heading = metadata.get("heading_path") or metadata.get("full_heading_path") or []
    if isinstance(heading, list):
        heading = " > ".join(str(value) for value in heading if value)
    heading = str(heading).strip()
    parts = []
    if title:
        parts.append(f"Document title: {title}")
    if heading:
        parts.append(f"Heading path: {heading}")
    parts.append(f"Content:\n{candidate['content']}")
    return "\n".join(parts)


def result_candidate_index(item: Any, document_ids: dict[str, int]) -> int:
    index = model_field(item, "index")
    if index is not None:
        return int(index)
    document = model_field(item, "document")
    document_id = model_field(document, "id") if document is not None else None
    if document_id is not None and str(document_id) in document_ids:
        return document_ids[str(document_id)]
    raise ValueError("Pinecone rerank result has neither a valid index nor document ID.")


def rerank(pc, query: str, candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Any]:
    selected = candidates[:INPUT_CANDIDATES]
    top_n = min(TOP_N, len(selected))
    documents = [
        {"id": candidate["chunk_id"], "text": rerank_text(candidate)}
        for candidate in selected
    ]
    kwargs: dict[str, Any] = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": documents,
        "rank_fields": ["text"],
        "top_n": top_n,
        "return_documents": True,
    }
    # Cohere 3.5 does not expose Pinecone's truncate parameter. BGE does.
    if RERANK_MODEL == "bge-reranker-v2-m3":
        kwargs["parameters"] = {"truncate": "END"}
    response = pc.inference.rerank(**kwargs)
    data = model_field(response, "data") or []
    document_ids = {document["id"]: index for index, document in enumerate(documents)}
    reranked: list[dict[str, Any]] = []
    used_indices: set[int] = set()
    for rank, item in enumerate(data, start=1):
        source_index = result_candidate_index(item, document_ids)
        if source_index < 0 or source_index >= len(selected):
            raise ValueError(f"Pinecone returned out-of-range candidate index: {source_index}")
        if source_index in used_indices:
            raise ValueError(f"Pinecone returned candidate index more than once: {source_index}")
        used_indices.add(source_index)
        source = selected[source_index]
        output = dict(source)
        output["hybrid_rank"] = source.get("rank")
        output["reranker_rank"] = rank
        output["reranker_score"] = float(model_field(item, "score") or 0.0)
        output["rank"] = rank
        reranked.append(output)
    if len(reranked) != top_n:
        raise ValueError(f"Pinecone returned {len(reranked)} results; expected {top_n}.")
    return reranked, model_field(response, "usage")


def main() -> None:
    validate_parameters()
    stored_query, hybrid_candidates, hybrid_payload = load_hybrid_candidates(
        HYBRID_RESULTS_FILE
    )
    if QUERY_OVERRIDE and QUERY_OVERRIDE != stored_query:
        raise ValueError(
            "--query does not match the latest Step-9 query. Run --step hybrid with this question first."
        )
    query = QUERY_OVERRIDE or stored_query

    from dotenv import load_dotenv

    project_root = HYBRID_ROOT.parent
    software_root = Path(__file__).resolve().parents[3]
    load_dotenv(project_root / ".env")
    load_dotenv(software_root / ".env")
    pinecone_api_key = require_setting("PINECONE_API_KEY")

    from pinecone import Pinecone

    pc = Pinecone(api_key=pinecone_api_key)
    reranked, usage = rerank(pc, query, hybrid_candidates)

    OUT.mkdir(parents=True, exist_ok=True)
    query_output_dir = OUT / "queries"
    query_output_dir.mkdir(exist_ok=True)
    result = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "query": query,
        "provider": "Pinecone Inference",
        "model": RERANK_MODEL,
        "input_hybrid_candidates": min(INPUT_CANDIDATES, len(hybrid_candidates)),
        "retained_candidates": len(reranked),
        "configured_top_n": TOP_N,
        "usage": jsonable(usage),
        "source_hybrid_parameters": hybrid_payload.get("parameters"),
        "candidates": reranked,
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    (OUT / "reranked_candidates.json").write_text(serialized, encoding="utf-8")
    query_id = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    (query_output_dir / f"{query_id}.json").write_text(serialized, encoding="utf-8")
    (OUT / "reranking_report.json").write_text(
        json.dumps({key: value for key, value in result.items() if key != "candidates"},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("-" * 100)
    print(f"Question:                 {query}")
    print(f"Pinecone reranker:        {RERANK_MODEL}")
    print(f"Hybrid candidates input: {result['input_hybrid_candidates']}")
    print(f"Candidates retained:     {len(reranked)}")
    print(f"Saved query ID:          {query_id}")
    print("Top reranked candidates:")
    for candidate in reranked:
        title = candidate["metadata"].get("title") or "Untitled"
        print(
            f"  {candidate['reranker_rank']}. {candidate['chunk_id']} | "
            f"score={candidate['reranker_score']:.8f} | "
            f"hybrid_rank={candidate['hybrid_rank']} | {title}"
        )


if __name__ == "__main__":
    main()
