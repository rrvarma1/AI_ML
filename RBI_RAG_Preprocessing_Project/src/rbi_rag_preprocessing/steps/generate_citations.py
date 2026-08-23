"""Step 14: generate deterministic citations and claim-to-chunk mappings."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ANSWER_ROOT = Path(os.environ["RBI_ANSWER_OUTPUT_DIR"]).expanduser().resolve()
LATEST_ANSWER_FILE = ANSWER_ROOT / "latest_answer.json"
OUT = Path(os.environ["RBI_CITATION_OUTPUT_DIR"]).expanduser().resolve()
QUERY_OVERRIDE = os.getenv("RBI_CITATION_QUERY", "").strip()
EVIDENCE_REFERENCE = re.compile(r"\[E(\d+)\]")
STRUCTURE_HEADING = re.compile(
    r"\b(?:chapter|section|clause|annex(?:ure)?|part|schedule|appendix|paragraph|para\.?)[\s\-:]*[A-Za-z0-9()./-]+",
    flags=re.IGNORECASE,
)


def metadata_value(metadata: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = metadata.get(name)
        if value not in (None, "", []):
            return value
    return None


def load_inputs(answer_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not answer_path.is_file():
        raise FileNotFoundError(
            "Run --step answer first; RBI_Grounded_Answers/latest_answer.json is missing."
        )
    audit = json.loads(answer_path.read_text(encoding="utf-8"))
    query_id = str(audit.get("query_id") or "").strip()
    answer = audit.get("answer")
    if not query_id or not isinstance(answer, dict):
        raise ValueError(f"Invalid Step-13 answer audit: {answer_path}")
    evidence_path = ANSWER_ROOT / "queries" / query_id / "evidence.json"
    if not evidence_path.is_file():
        raise FileNotFoundError(f"Step-13 evidence snapshot is missing: {evidence_path}")
    evidence_payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    selected = evidence_payload.get("selected")
    if not isinstance(selected, list):
        raise ValueError(f"Invalid Step-13 evidence snapshot: {evidence_path}")
    return audit, selected


def heading_parts(metadata: dict[str, Any], title: str) -> list[str]:
    value = metadata_value(metadata, "heading_path", "full_heading_path", "heading_path_text")
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
    elif isinstance(value, str):
        parts = [part.strip() for part in re.split(r"\s*(?:>|›|→)\s*", value) if part.strip()]
    else:
        parts = []
    if parts and parts[0].casefold() == title.casefold():
        parts = parts[1:]
    return parts


def section_or_clause(metadata: dict[str, Any], headings: list[str]) -> str | None:
    explicit = []
    for name in ("page_or_section", "section_number", "section", "clause"):
        value = metadata.get(name)
        if value not in (None, "") and str(value) not in explicit:
            explicit.append(str(value))
    if explicit:
        return " → ".join(explicit)
    structural = []
    for heading in headings:
        matches = STRUCTURE_HEADING.findall(heading)
        for match in matches:
            cleaned = match.strip()
            if cleaned not in structural:
                structural.append(cleaned)
    if structural:
        return " → ".join(structural)
    return headings[-1] if headings else None


def valid_source_url(metadata: dict[str, Any]) -> str | None:
    value = metadata_value(metadata, "source_url", "url", "document_url")
    if not value:
        return None
    text = str(value).strip()
    return text if text.startswith(("https://", "http://")) else None


def make_citation(record: dict[str, Any]) -> dict[str, Any]:
    evidence_id = str(record.get("evidence_id") or "").strip()
    chunk_id = str(record.get("chunk_id") or "").strip()
    metadata = record.get("metadata") or {}
    title_value = metadata_value(metadata, "title", "document_title")
    title = str(title_value).strip() if title_value else None
    headings = heading_parts(metadata, title or "")
    section = section_or_clause(metadata, headings)
    page_start, page_end = metadata.get("page_start"), metadata.get("page_end")
    source_url = valid_source_url(metadata)
    warnings = []
    if not title:
        warnings.append("missing_document_title")
    if not headings:
        warnings.append("missing_full_heading_path")
    if not section:
        warnings.append("missing_section_or_clause")
    if page_start in (None, "") or page_end in (None, ""):
        warnings.append("missing_page_range")
    if not source_url:
        warnings.append("missing_or_invalid_source_url")
    if not chunk_id:
        warnings.append("missing_chunk_id")

    if page_start not in (None, "") and page_end not in (None, ""):
        page_range = (
            f"page {page_start}" if str(page_start) == str(page_end)
            else f"pages {page_start}–{page_end}"
        )
    else:
        page_range = "pages not available in stored metadata"
    heading_text = " → ".join(headings) if headings else "heading path not available"
    url_text = source_url or "source URL not available in stored metadata"
    formatted = (
        f"[{evidence_id}] {title or 'Untitled RBI document'} — {heading_text} — "
        f"{section or 'section or clause not available'} — {page_range} — "
        f"{url_text} — chunk {chunk_id or 'not available'}"
    )
    return {
        "evidence_id": evidence_id,
        "document_id": metadata.get("document_id"),
        "document_title": title,
        "full_heading_path": headings,
        "full_heading_path_text": heading_text,
        "section_or_clause": section,
        "page_start": page_start,
        "page_end": page_end,
        "page_range": page_range,
        "source_url": source_url,
        "chunk_id": chunk_id or None,
        "document_type": metadata_value(metadata, "document_type", "doc_type"),
        "regulatory_status": metadata_value(metadata, "regulatory_status", "status"),
        "formatted_citation": formatted,
        "metadata_complete": not warnings,
        "warnings": warnings,
    }


def inline_evidence_ids(text: str) -> list[str]:
    result = []
    for number in EVIDENCE_REFERENCE.findall(text or ""):
        evidence_id = f"E{number}"
        if evidence_id not in result:
            result.append(evidence_id)
    return result


def claim_records(answer: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    answer_ids = inline_evidence_ids(str(answer.get("answer") or ""))
    records.append({
        "claim_id": "ANS-001", "claim_type": "answer_summary",
        "text": answer.get("answer") or "", "evidence_ids": answer_ids,
    })
    configurations = (
        ("claims", "CLM", "material_claim"),
        ("conditions_and_exceptions", "COND", "condition_or_exception"),
        ("conflicts", "CONF", "conflicting_evidence"),
    )
    for field, prefix, claim_type in configurations:
        for index, item in enumerate(answer.get(field) or [], start=1):
            records.append({
                "claim_id": f"{prefix}-{index:03d}", "claim_type": claim_type,
                "text": item.get("text") or "", "evidence_ids": item.get("evidence_ids") or [],
            })
    status_note = str(answer.get("regulatory_status_note") or "").strip()
    if status_note:
        records.append({
            "claim_id": "STAT-001", "claim_type": "regulatory_status_note",
            "text": status_note, "evidence_ids": answer.get("evidence_ids_used") or [],
        })
    return records


def build_claim_mappings(
    answer: dict[str, Any], citations: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    mappings, errors = [], []
    substantive = answer.get("answer_status") in {
        "answered", "partially_answered", "conflicting_evidence"
    }
    for claim in claim_records(answer):
        evidence_ids = []
        for evidence_id in claim["evidence_ids"]:
            if evidence_id not in evidence_ids:
                evidence_ids.append(evidence_id)
        unknown = [item for item in evidence_ids if item not in citations]
        if unknown:
            errors.append(
                f"{claim['claim_id']} references unknown evidence IDs: {', '.join(unknown)}"
            )
        resolved = [citations[item] for item in evidence_ids if item in citations]
        chunk_ids = []
        for citation in resolved:
            if citation["chunk_id"] and citation["chunk_id"] not in chunk_ids:
                chunk_ids.append(citation["chunk_id"])
        requires_support = substantive and claim["claim_type"] in {
            "answer_summary", "material_claim", "condition_or_exception",
            "conflicting_evidence", "regulatory_status_note",
        }
        if requires_support and not chunk_ids:
            errors.append(f"{claim['claim_id']} has no supporting chunk ID")
        mappings.append({
            **claim,
            "evidence_ids": evidence_ids,
            "supporting_chunk_ids": chunk_ids,
            "citations": resolved,
            "mapping_complete": not unknown and (bool(chunk_ids) or not requires_support),
        })
    return mappings, errors


def markdown_citation(citation: dict[str, Any]) -> str:
    source = (
        f"[RBI source]({citation['source_url']})" if citation["source_url"]
        else "Source URL unavailable in stored metadata"
    )
    return (
        f"[{citation['evidence_id']}] {citation['document_title'] or 'Untitled RBI document'} — "
        f"{citation['full_heading_path_text']} — "
        f"{citation['section_or_clause'] or 'section or clause unavailable'} — "
        f"{citation['page_range']} — {source} — chunk `{citation['chunk_id'] or 'unavailable'}`"
    )


def render_markdown(
    answer: dict[str, Any], citations: dict[str, dict[str, Any]], mappings: list[dict[str, Any]]
) -> str:
    lines = ["# Answer with verified citations", "", str(answer.get("answer") or "").strip()]
    lines.extend(["", "## Conditions and exceptions", ""])
    conditions = answer.get("conditions_and_exceptions") or []
    if conditions:
        for item in conditions:
            text = item["text"].strip()
            suffix = "".join(
                f"[{evidence_id}]" for evidence_id in item.get("evidence_ids", [])
                if f"[{evidence_id}]" not in text
            )
            lines.append(f"- {text}{suffix}")
    else:
        lines.append("No additional conditions or exceptions were established by the supplied evidence.")
    if answer.get("conflicts"):
        lines.extend(["", "## Conflicting evidence", ""])
        for item in answer["conflicts"]:
            lines.append(f"- {item['text']}")
    lines.extend(["", "## Sources", ""])
    for evidence_id in answer.get("evidence_ids_used") or []:
        if evidence_id in citations:
            lines.append(f"- {markdown_citation(citations[evidence_id])}")
    if not any(line.startswith("- [E") for line in lines):
        lines.append("- No source was cited because no grounded conclusion was generated.")
    lines.extend(["", "## Claim-to-chunk mapping", ""])
    for mapping in mappings:
        lines.extend([
            f"### {mapping['claim_id']} — {mapping['claim_type']}", "",
            mapping["text"], "",
            "Supporting chunks: " + (
                ", ".join(f"`{item}`" for item in mapping["supporting_chunk_ids"])
                if mapping["supporting_chunk_ids"] else "None"
            ), "",
        ])
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    audit, evidence = load_inputs(LATEST_ANSWER_FILE)
    query = str(audit.get("query") or "").strip()
    if QUERY_OVERRIDE and QUERY_OVERRIDE != query:
        raise ValueError(
            "--query does not match the latest Step-13 question. Run --step answer for this question first."
        )
    citations: dict[str, dict[str, Any]] = {}
    for record in evidence:
        citation = make_citation(record)
        evidence_id = citation["evidence_id"]
        if not evidence_id:
            raise ValueError("A Step-13 evidence record is missing evidence_id.")
        if evidence_id in citations:
            raise ValueError(f"Duplicate Step-13 evidence ID: {evidence_id}")
        citations[evidence_id] = citation
    answer = audit["answer"]
    mappings, mapping_errors = build_claim_mappings(answer, citations)
    used_ids = [item for item in answer.get("evidence_ids_used") or [] if item in citations]
    used_citations = [citations[item] for item in used_ids]
    metadata_warnings = [
        {"evidence_id": item["evidence_id"], "warnings": item["warnings"]}
        for item in used_citations if item["warnings"]
    ]
    status = (
        "no_citable_answer" if answer.get("answer_status") in {
            "insufficient_evidence", "generation_error"
        } and not used_ids
        else "review_required" if mapping_errors or metadata_warnings
        else "passed"
    )
    result = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "query_id": audit["query_id"], "query": query,
        "answer_status": answer.get("answer_status"),
        "citation_status": status,
        "citation_method": "deterministic_metadata_rendering",
        "claim_mapping_errors": mapping_errors,
        "metadata_warnings": metadata_warnings,
        "unique_citations_used": used_citations,
        "claim_to_chunk_mapping": mappings,
    }
    markdown = render_markdown(answer, citations, mappings)

    OUT.mkdir(parents=True, exist_ok=True)
    query_dir = OUT / "queries" / audit["query_id"]
    query_dir.mkdir(parents=True, exist_ok=True)
    (OUT / "citation_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "claim_citation_map.json").write_text(
        json.dumps({"query_id": audit["query_id"], "mappings": mappings},
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "answer_with_citations.md").write_text(markdown, encoding="utf-8")
    report = {
        "generated_at_utc": result["generated_at_utc"],
        "query_id": audit["query_id"], "citation_status": status,
        "claims_mapped": len(mappings),
        "unique_chunks_cited": len({
            chunk_id for mapping in mappings for chunk_id in mapping["supporting_chunk_ids"]
        }),
        "unique_evidence_ids_cited": len(used_ids),
        "claim_mapping_error_count": len(mapping_errors),
        "metadata_warning_count": len(metadata_warnings),
    }
    (OUT / "citation_generation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name, content in (
        ("citation_manifest.json", json.dumps(result, ensure_ascii=False, indent=2)),
        ("claim_citation_map.json", json.dumps({"mappings": mappings}, ensure_ascii=False, indent=2)),
        ("answer_with_citations.md", markdown),
    ):
        (query_dir / name).write_text(content, encoding="utf-8")

    print("-" * 100)
    print(f"Question:                 {query}")
    print(f"Citation status:          {status}")
    print(f"Claims mapped:            {report['claims_mapped']}")
    print(f"Unique chunks cited:      {report['unique_chunks_cited']}")
    print(f"Metadata warnings:        {report['metadata_warning_count']}")
    print(f"Mapping errors:           {report['claim_mapping_error_count']}")
    print(f"Saved query ID:           {audit['query_id']}")
    print("-" * 100)
    print(markdown)


if __name__ == "__main__":
    main()
