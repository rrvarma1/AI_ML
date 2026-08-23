"""Step 8: build a persistent BM25 lexical index from Step-6 leaf chunks."""
from __future__ import annotations

import json
import os
import pickle
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

CHUNK_ROOT = Path(os.environ["RBI_CHUNK_OUTPUT_DIR"]).expanduser().resolve()
CHUNKS_FILE = CHUNK_ROOT / "chunks.jsonl"
OUT = Path(os.environ["RBI_BM25_OUTPUT_DIR"]).expanduser().resolve()
QUERY = os.getenv("RBI_BM25_QUERY", "").strip()
TOP_K = int(os.getenv("RBI_BM25_TOP_K", "5"))
BM25_K1 = float(os.getenv("RBI_BM25_K1", "1.5"))
BM25_B = float(os.getenv("RBI_BM25_B", "0.75"))

# Retains regulatory abbreviations, dotted/decimal numbers, dates, percentages,
# monetary symbols and words. Query text is processed by this same tokenizer.
TOKEN_PATTERN = re.compile(
    r"₹|\b(?:rs\.?|inr)\b|\b\d+(?:\.\d+)*(?:[/-]\d+)*%?|\b[^\W_]+(?:[-'][^\W_]+)*\b",
    flags=re.IGNORECASE | re.UNICODE,
)


def tokenize(text: str) -> list[str]:
    return [match.group(0).casefold().rstrip(".") for match in TOKEN_PATTERN.finditer(text)]


def string_values(value: Any) -> Iterable[str]:
    if isinstance(value, str) and value.strip():
        yield value.strip()
    elif isinstance(value, list):
        for item in value:
            yield from string_values(item)


def retrieval_text(content: str, metadata: dict[str, Any]) -> str:
    """Make each BM25 document self-contained without altering stored content."""
    fields = (
        "title",
        "document_type",
        "domain",
        "heading_path",
        "full_heading_path",
        "section_number",
        "section",
        "clause",
        "page_or_section",
    )
    prefixes: list[str] = []
    seen: set[str] = set()
    for field in fields:
        for value in string_values(metadata.get(field)):
            normalized = value.casefold()
            if normalized not in seen:
                seen.add(normalized)
                prefixes.append(value)
    return "\n".join([*prefixes, content])


def load_chunk_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Step-6 chunks file not found: {path}")
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            content = record.get("content")
            metadata = record.get("metadata")
            if not isinstance(content, str) or not content.strip() or not isinstance(metadata, dict):
                raise ValueError(f"Invalid content/metadata at {path}:{line_number}")
            chunk_id = str(metadata.get("chunk_id") or "").strip()
            if not chunk_id:
                raise ValueError(f"Missing deterministic chunk_id at {path}:{line_number}")
            if chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk_id {chunk_id!r} at {path}:{line_number}")
            seen_ids.add(chunk_id)
            tokens = tokenize(retrieval_text(content, metadata))
            if not tokens:
                raise ValueError(f"No indexable tokens for chunk {chunk_id!r}")
            records.append({
                "chunk_id": chunk_id,
                "content": content,
                "metadata": metadata,
                "tokenized_text": tokens,
            })
    if not records:
        raise ValueError(f"No chunks found in {path}")
    return records


def search(index, records: list[dict[str, Any]], query: str, top_k: int) -> list[dict[str, Any]]:
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    scores = index.get_scores(query_tokens)
    ranked = sorted(range(len(records)), key=lambda position: (-float(scores[position]), position))
    results = []
    for position in ranked[: min(top_k, len(ranked))]:
        record = records[position]
        results.append({
            "rank": len(results) + 1,
            "chunk_id": record["chunk_id"],
            "score": float(scores[position]),
            "title": record["metadata"].get("title"),
            "heading_path": record["metadata"].get("heading_path"),
            "content_preview": record["content"][:500],
        })
    return results


def main() -> None:
    from rank_bm25 import BM25Okapi

    records = load_chunk_records(CHUNKS_FILE)
    tokenized_corpus = [record["tokenized_text"] for record in records]
    index = BM25Okapi(tokenized_corpus, k1=BM25_K1, b=BM25_B)

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    with (OUT / "bm25_index.pkl").open("wb") as stream:
        pickle.dump({
            "format_version": 1,
            "algorithm": "BM25Okapi",
            "k1": BM25_K1,
            "b": BM25_B,
            "chunk_ids": [record["chunk_id"] for record in records],
            "index": index,
        }, stream, protocol=pickle.HIGHEST_PROTOCOL)

    with (OUT / "chunk_store.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    document_counts = Counter(
        str(record["metadata"].get("document_id") or "unknown") for record in records
    )
    token_counts = [len(tokens) for tokens in tokenized_corpus]
    query_results = search(index, records, QUERY, TOP_K) if QUERY else []
    report = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_chunks_file": str(CHUNKS_FILE),
        "algorithm": "BM25Okapi",
        "parameters": {"k1": BM25_K1, "b": BM25_B},
        "chunk_count": len(records),
        "unique_chunk_id_count": len({record["chunk_id"] for record in records}),
        "document_count": len(document_counts),
        "chunks_by_document": dict(sorted(document_counts.items())),
        "token_statistics": {
            "total": sum(token_counts),
            "minimum_per_chunk": min(token_counts),
            "maximum_per_chunk": max(token_counts),
            "average_per_chunk": round(sum(token_counts) / len(token_counts), 2),
        },
        "id_alignment": "chunk_id values copied unchanged from Step-6 chunks.jsonl",
        "query": QUERY or None,
        "query_results": query_results,
    }
    (OUT / "bm25_index_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("-" * 100)
    print(f"BM25 chunks indexed: {len(records)}")
    print(f"Source documents:    {len(document_counts)}")
    print(f"Unique chunk IDs:    {report['unique_chunk_id_count']}")
    print(f"BM25 output:         {OUT}")
    if query_results:
        print("Lexical search results:")
        for result in query_results:
            print(f"  {result['rank']}. {result['chunk_id']} | score={result['score']:.6f}")


if __name__ == "__main__":
    main()
