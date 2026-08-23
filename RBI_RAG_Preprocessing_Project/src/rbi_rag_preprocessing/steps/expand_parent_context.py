"""Step 12: expand reranked leaf chunks with regulatory hierarchy context."""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CHUNK_ROOT = Path(os.environ["RBI_CHUNK_OUTPUT_DIR"]).expanduser().resolve()
NODE_HIERARCHY_FILE = CHUNK_ROOT / "node_hierarchy.jsonl"
RERANK_ROOT = Path(os.environ["RBI_RERANK_OUTPUT_DIR"]).expanduser().resolve()
RERANKED_FILE = RERANK_ROOT / "reranked_candidates.json"
OUT = Path(os.environ["RBI_CONTEXT_OUTPUT_DIR"]).expanduser().resolve()
QUERY_OVERRIDE = os.getenv("RBI_CONTEXT_QUERY", "").strip()

LEGAL_DEPENDENCY = re.compile(
    r"(?:^|\n)\s*(?:provided\s+(?:further\s+)?that|except\s+where|notwithstanding|"
    r"subject\s+to|unless\s+otherwise|save\s+as(?:\s+otherwise)?|exception\s*:)",
    flags=re.IGNORECASE,
)
TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}.*\|.*$")
FOOTNOTE_REFERENCE = re.compile(r"\[\^?(\d{1,3})\]|(?<!\w)([†‡])")


def relationship_ids(node: dict[str, Any], name: str) -> list[str]:
    value = (node.get("relationships") or {}).get(name)
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)]


def load_node_hierarchy(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Step-6 node hierarchy is missing: {path}")
    nodes: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            node = json.loads(line)
            node_id = str(node.get("node_id") or "").strip()
            if not node_id or not isinstance(node.get("text"), str):
                raise ValueError(f"Invalid hierarchy node at {path}:{line_number}")
            if node_id in nodes:
                raise ValueError(f"Duplicate hierarchy node_id {node_id!r}")
            nodes[node_id] = node
    if not nodes:
        raise ValueError(f"No hierarchy nodes found in {path}")
    return nodes


def load_reranked(path: Path) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            "Run --step rerank first; RBI_Reranked_Retrieval/reranked_candidates.json is missing."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    query = str(payload.get("query") or "").strip()
    candidates = payload.get("candidates")
    if not query or not isinstance(candidates, list) or not candidates:
        raise ValueError(f"Invalid Step-11 reranked result: {path}")
    return query, candidates, payload


def same_document(node: dict[str, Any], document_id: str) -> bool:
    return str(node.get("document_id") or node.get("metadata", {}).get("document_id") or "") == document_id


def related_node(
    node_id: str | None,
    nodes: dict[str, dict[str, Any]],
    document_id: str,
) -> dict[str, Any] | None:
    if not node_id:
        return None
    node = nodes.get(str(node_id))
    return node if node and same_document(node, document_id) else None


def sibling_id(
    leaf: dict[str, Any], parent: dict[str, Any] | None, direction: str
) -> str | None:
    direct = relationship_ids(leaf, direction)
    if direct:
        return direct[0]
    if not parent:
        return None
    children = relationship_ids(parent, "children")
    leaf_id = str(leaf["node_id"])
    if leaf_id not in children:
        return None
    position = children.index(leaf_id) + (-1 if direction == "previous" else 1)
    return children[position] if 0 <= position < len(children) else None


def table_headers(text: str) -> list[str]:
    lines = text.splitlines()
    headers: list[str] = []
    for index in range(len(lines) - 1):
        if TABLE_ROW.match(lines[index]) and TABLE_SEPARATOR.match(lines[index + 1]):
            header = f"{lines[index].strip()}\n{lines[index + 1].strip()}"
            if header not in headers:
                headers.append(header)
    return headers


def contains_table(text: str) -> bool:
    return sum(bool(TABLE_ROW.match(line)) for line in text.splitlines()) >= 2


def footnote_markers(text: str) -> list[str]:
    markers = []
    for match in FOOTNOTE_REFERENCE.finditer(text):
        marker = match.group(1) or match.group(2)
        if marker not in markers:
            markers.append(marker)
    return markers


def matching_footnotes(text: str, markers: list[str]) -> list[str]:
    if not markers:
        return []
    lines = text.splitlines()
    results = []
    for marker in markers:
        pattern = re.compile(
            rf"^\s*(?:\[\^?{re.escape(marker)}\]|{re.escape(marker)}[.)]|{re.escape(marker)})\s+.+"
        )
        for index, line in enumerate(lines):
            if pattern.match(line):
                block = [line.strip()]
                for following in lines[index + 1:index + 4]:
                    if not following.strip() or re.match(r"^\s*(?:\[\^?\d+\]|\d+[.)])\s+", following):
                        break
                    block.append(following.strip())
                footnote = " ".join(block)
                if footnote not in results:
                    results.append(footnote)
                break
    return results


def looks_like_footnote(text: str, markers: list[str]) -> bool:
    return bool(matching_footnotes(text, markers))


def context_entry(node: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "node_id": node["node_id"],
        "document_id": node.get("document_id") or node.get("metadata", {}).get("document_id"),
        "reason": reason,
        "text": node["text"],
        "metadata": node.get("metadata") or {},
        "relationships": node.get("relationships") or {},
    }


def expand_candidate(
    candidate: dict[str, Any], nodes: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = candidate.get("metadata") or {}
    document_id = str(metadata.get("document_id") or "").strip()
    leaf_id = str(metadata.get("llamaindex_node_id") or "").strip()
    parent_id = str(metadata.get("parent_node_id") or "").strip()
    if not document_id or not leaf_id:
        raise ValueError(f"Candidate {candidate.get('chunk_id')} lacks document/node metadata.")
    leaf = related_node(leaf_id, nodes, document_id)
    if leaf is None:
        raise ValueError(f"Leaf node {leaf_id!r} for {candidate.get('chunk_id')} is missing.")
    if not parent_id:
        parent_ids = relationship_ids(leaf, "parent")
        parent_id = parent_ids[0] if parent_ids else ""
    parent = related_node(parent_id, nodes, document_id)
    if parent is None:
        raise ValueError(f"Parent node {parent_id!r} for {candidate.get('chunk_id')} is missing.")

    primary_text = candidate["content"]
    legal_dependency = bool(
        metadata.get("starts_with_legal_dependency") or LEGAL_DEPENDENCY.search(primary_text)
    )
    has_table = contains_table(primary_text)
    markers = footnote_markers(primary_text)
    contexts: list[dict[str, Any]] = [context_entry(parent, "parent_section")]
    included = {str(parent["node_id"]), leaf_id}

    previous = related_node(sibling_id(leaf, parent, "previous"), nodes, document_id)
    following = related_node(sibling_id(leaf, parent, "next"), nodes, document_id)
    if legal_dependency and previous and previous["node_id"] not in included:
        contexts.append(context_entry(previous, "governing_rule_for_legal_dependency"))
        included.add(previous["node_id"])
    if has_table and previous and previous["node_id"] not in included:
        contexts.append(context_entry(previous, "adjacent_table_heading_or_prior_clause"))
        included.add(previous["node_id"])
    if has_table and following and contains_table(following["text"]) and following["node_id"] not in included:
        contexts.append(context_entry(following, "adjacent_table_continuation"))
        included.add(following["node_id"])

    footnotes = matching_footnotes(parent["text"], markers)
    if markers and not footnotes:
        for adjacent, reason in (
            (previous, "adjacent_footnote"), (following, "adjacent_footnote")
        ):
            if adjacent and looks_like_footnote(adjacent["text"], markers):
                footnotes.extend(matching_footnotes(adjacent["text"], markers))
                if adjacent["node_id"] not in included:
                    contexts.append(context_entry(adjacent, reason))
                    included.add(adjacent["node_id"])

    headers = table_headers(primary_text) or table_headers(parent["text"])
    parts = [
        f"REGULATORY PATH: {metadata.get('heading_path_text') or ' > '.join(metadata.get('heading_path') or [])}",
        f"PARENT SECTION [{parent['node_id']}]:\n{parent['text']}",
    ]
    for context in contexts[1:]:
        parts.append(
            f"RELATED CONTEXT ({context['reason']}) [{context['node_id']}]:\n{context['text']}"
        )
    if headers:
        parts.append("REPEATED TABLE HEADER(S):\n" + "\n\n".join(headers))
    if footnotes:
        parts.append("RELEVANT FOOTNOTE(S):\n" + "\n".join(footnotes))
    parts.append(f"PRIMARY RETRIEVED CHUNK [{candidate['chunk_id']}]:\n{primary_text}")
    assembled = "\n\n".join(parts)

    expanded = dict(candidate)
    expanded["expansion"] = {
        "leaf_node_id": leaf_id,
        "parent_node_id": parent_id,
        "context_node_ids": [context["node_id"] for context in contexts],
        "reasons": [context["reason"] for context in contexts],
        "legal_dependency_detected": legal_dependency,
        "table_detected": has_table,
        "repeated_table_headers": headers,
        "footnote_markers": markers,
        "relevant_footnotes": footnotes,
        "assembled_token_count_estimate": len(assembled.split()),
    }
    expanded["expanded_content"] = assembled
    return expanded, contexts


def expand_all(
    candidates: list[dict[str, Any]], nodes: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    expanded = []
    catalog: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        item, contexts = expand_candidate(candidate, nodes)
        expanded.append(item)
        for context in contexts:
            node_id = context["node_id"]
            if node_id not in catalog:
                catalog[node_id] = context
            elif context["reason"] not in catalog[node_id]["reason"].split("|"):
                catalog[node_id]["reason"] += f"|{context['reason']}"
    return expanded, catalog


def main() -> None:
    query, candidates, rerank_payload = load_reranked(RERANKED_FILE)
    if QUERY_OVERRIDE and QUERY_OVERRIDE != query:
        raise ValueError(
            "--query does not match the latest Step-11 query. Run --step rerank for this question first."
        )
    nodes = load_node_hierarchy(NODE_HIERARCHY_FILE)
    expanded, catalog = expand_all(candidates, nodes)

    OUT.mkdir(parents=True, exist_ok=True)
    query_output_dir = OUT / "queries"
    query_output_dir.mkdir(exist_ok=True)
    result = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "query": query,
        "strategy": "post_rerank_hierarchy_expansion",
        "source_reranker": rerank_payload.get("model"),
        "reranked_candidates": len(candidates),
        "expanded_candidates": len(expanded),
        "unique_context_nodes": len(catalog),
        "legal_dependencies_expanded": sum(
            item["expansion"]["legal_dependency_detected"] for item in expanded
        ),
        "table_chunks_expanded": sum(item["expansion"]["table_detected"] for item in expanded),
        "footnotes_resolved": sum(
            len(item["expansion"]["relevant_footnotes"]) for item in expanded
        ),
        "evidence": expanded,
        "context_catalog": catalog,
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    (OUT / "expanded_context.json").write_text(serialized, encoding="utf-8")
    query_id = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    (query_output_dir / f"{query_id}.json").write_text(serialized, encoding="utf-8")
    (OUT / "context_expansion_report.json").write_text(
        json.dumps({key: value for key, value in result.items()
                    if key not in ("evidence", "context_catalog")},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("-" * 100)
    print(f"Question:                     {query}")
    print(f"Reranked chunks expanded:     {len(expanded)}")
    print(f"Unique hierarchy nodes added: {len(catalog)}")
    print(f"Legal dependencies expanded:  {result['legal_dependencies_expanded']}")
    print(f"Table chunks expanded:         {result['table_chunks_expanded']}")
    print(f"Footnotes resolved:            {result['footnotes_resolved']}")
    print(f"Saved query ID:                {query_id}")
    for item in expanded:
        print(
            f"  {item['rank']}. {item['chunk_id']} | "
            f"contexts={len(item['expansion']['context_node_ids'])} | "
            f"tokens~{item['expansion']['assembled_token_count_estimate']}"
        )


if __name__ == "__main__":
    main()
