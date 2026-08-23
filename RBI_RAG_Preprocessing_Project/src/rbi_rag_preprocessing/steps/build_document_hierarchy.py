from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import statistics
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import fitz

CLEAN_ROOT = Path(os.environ["RBI_CLEAN_OUTPUT_DIR"]).expanduser().resolve()
CLEAN_DOCS = CLEAN_ROOT / "cleaned_documents"
RAW_DOCS = Path(os.environ["RBI_RAW_OUTPUT_DIR"]).expanduser().resolve() / "raw_extractions"
PDF_ROOT = Path(os.environ["RBI_CORPUS_DIR"]).expanduser().resolve()
OUT = Path(os.environ["RBI_HIERARCHY_OUTPUT_DIR"]).expanduser().resolve()
HIERARCHIES = OUT / "document_hierarchies"
STRUCTURED = OUT / "structured_documents"

EXPLICIT_MAJOR = re.compile(
    r"^(?P<kind>CHAPTER|PART|ANNEX(?:URE)?|APPENDIX|SCHEDULE)\b\s*[-–—:]?\s*(?P<number>[A-Z0-9IVXLCDM.-]+)?\b",
    re.I,
)
NUMERIC = re.compile(r"^(?P<number>\d+(?:\.\d+){0,5})[.)]?\s+(?P<label>.+)", re.S)
LETTER = re.compile(r"^(?P<number>\([a-z]\)|[a-z][.)])\s+(?P<label>.+)", re.I | re.S)
ROMAN = re.compile(r"^(?P<number>\([ivxlcdm]+\)|[ivxlcdm]+[.)])\s+(?P<label>.+)", re.I | re.S)
BOILERPLATE = re.compile(
    r"^(?:reserve bank of india|भारतीय|rbi/|dear sir|dear madam|yours faithfully|encl|the chairman|the director|central office|mumbai[- ]?\d|warning|चेतावनी)",
    re.I,
)
DOT_LEADER = re.compile(r"\.{5,}\s*\d*\s*$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\.{3,}\s*\d*\s*$", "", text)
    text = re.sub(r"^\d+[A-Z]\s+", "", text)
    text = re.sub(r"[^a-zA-Z0-9()]+", " ", text).lower()
    return " ".join(text.split())


def pdf_for_document(metadata):
    wanted = metadata.get("source_file_name")
    exact = PDF_ROOT / wanted
    if exact.exists():
        return exact
    key = normalize(wanted)
    scored = [(SequenceMatcher(None, key, normalize(p.name)).ratio(), p) for p in PDF_ROOT.glob("*.pdf")]
    return max(scored)[1]


def body_font_size(raw_document):
    weighted = []
    for page in raw_document["pages"]:
        for block in page.get("blocks", []):
            if block.get("block_type") != "text":
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if text:
                        weighted.extend([float(span.get("size", 0))] * min(100, max(1, len(text))))
    return statistics.median(weighted) if weighted else 0.0


def is_bold_span(span):
    return "bold" in (span.get("font") or "").lower() or bool(span.get("flags", 0) & 16)


def block_style(raw_block):
    spans = [
        span
        for line in raw_block.get("lines", [])
        for span in line.get("spans", [])
        if span.get("text", "").strip()
    ]
    total = sum(len(s["text"].strip()) for s in spans) or 1
    bold_chars = sum(len(s["text"].strip()) for s in spans if is_bold_span(s))
    return {
        "max_font_size": max([float(s.get("size", 0)) for s in spans] or [0]),
        "bold_ratio": bold_chars / total,
        "fonts": sorted({s.get("font") for s in spans if s.get("font")}),
    }


def leading_bold_text(raw_block):
    parts = []
    started = False
    for line in raw_block.get("lines", []):
        spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
        if not spans:
            continue
        chars = sum(len(s["text"].strip()) for s in spans) or 1
        bold_ratio = sum(len(s["text"].strip()) for s in spans if is_bold_span(s)) / chars
        if bold_ratio >= 0.60:
            parts.append("".join(s.get("text", "") for s in spans).strip())
            started = True
        elif started:
            break
        else:
            break
    return " ".join(parts).strip()


def clean_heading_label(text: str, maximum=220):
    text = re.sub(r"\s+", " ", text).strip(" :-–—\n\t")
    if len(text) > maximum:
        text = text[:maximum].rsplit(" ", 1)[0] + "…"
    return text


def heading_candidate(content, raw_block, body_size, within_table):
    if within_table:
        return None
    flat = clean_heading_label(content, maximum=500)
    if not flat or DOT_LEADER.search(flat) or BOILERPLATE.search(flat):
        return None
    style = block_style(raw_block)
    bold_prefix = clean_heading_label(leading_bold_text(raw_block))
    major = EXPLICIT_MAJOR.match(flat)
    if major:
        words = re.findall(r"[A-Za-z]+", flat)
        upper_ratio = sum(word.isupper() for word in words) / max(1, len(words))
        if len(flat) > 150 or (style["bold_ratio"] < 0.45 and upper_ratio < 0.65):
            return None
        kind = major.group("kind").lower()
        return {
            "label": clean_heading_label(bold_prefix or flat),
            "kind": "annexure" if kind.startswith("annex") else kind,
            "number": major.group("number"),
            "suggested_level": 1,
            "confidence": 0.98,
            "detection": "explicit_structural_marker",
            "style": style,
        }
    numeric = NUMERIC.match(flat)
    if numeric:
        number = numeric.group("number")
        first_integer = int(number.split(".")[0])
        remainder = numeric.group("label").strip()
        if first_integer > 500 or re.match(r"^(?:of\b|years?\b|days?\b|months?\b|%|₹|rs\.?\b)", remainder, re.I):
            return None
        depth = number.count(".") + 1
        preferred = bold_prefix if bold_prefix and normalize(bold_prefix).startswith(normalize(number)) else flat
        if len(preferred) > 220:
            # Long numbered normative paragraphs are content unless typography
            # isolates a concise bold heading prefix.
            if not bold_prefix or len(bold_prefix) > 180:
                return None
            preferred = bold_prefix
        heading_text = clean_heading_label(preferred)
        heading_words = len(re.findall(r"[A-Za-z]+", heading_text))
        style_support = style["bold_ratio"] >= 0.50 or style["max_font_size"] >= body_size + 0.5
        if heading_words > (18 if depth > 1 else 14):
            return None
        if not style_support and (heading_words > 14 or heading_text.endswith((".", ";"))):
            return None
        return {
            "label": heading_text,
            "kind": "section" if depth == 1 else "subsection",
            "number": number,
            "suggested_level": min(6, depth + 1),
            "confidence": 0.92 if style_support else 0.78,
            "detection": "numbering_and_typography",
            "style": style,
        }
    for pattern, kind in ((LETTER, "clause"), (ROMAN, "subclause")):
        match = pattern.match(flat)
        if match and len(flat) <= 180 and style["bold_ratio"] >= 0.55:
            return {
                "label": clean_heading_label(bold_prefix or flat),
                "kind": kind,
                "number": match.group("number"),
                "suggested_level": None,
                "confidence": 0.82,
                "detection": "clause_marker_and_typography",
                "style": style,
            }
    words = re.findall(r"[A-Za-z]+", flat)
    if 1 <= len(words) <= 16 and len(flat) <= 140:
        upper_ratio = sum(word.isupper() for word in words) / len(words)
        title_like = flat == flat.title()
        if upper_ratio >= 0.75 and style["bold_ratio"] >= 0.50:
            return {
                "label": flat,
                "kind": "heading",
                "number": None,
                "suggested_level": 2,
                "confidence": 0.84,
                "detection": "uppercase_and_typography",
                "style": style,
            }
        if title_like and style["bold_ratio"] >= 0.75 and not flat.endswith((".", ";", ",")):
            return {
                "label": flat,
                "kind": "heading",
                "number": None,
                "suggested_level": 2,
                "confidence": 0.72,
                "detection": "title_case_and_typography",
                "style": style,
            }
    return None


def clean_toc(toc):
    result = []
    for level, title, page in toc:
        title = clean_heading_label(title)
        if not title or re.fullmatch(r"[-_.\s]+", title):
            continue
        title = re.sub(r"^\d+[A-Z]\s+", "", title)
        if re.fullmatch(r"\d+[.)]?", title):
            continue
        if len(title) > 180:
            title = clean_heading_label(title, maximum=180)
        result.append({"level": max(1, min(6, int(level))), "title": title, "page_number": int(page)})
    return result


def add_node(nodes, stack, *, document_id, label, level, page_number, block_id, kind, number, source, confidence, style=None):
    level = max(1, min(6, level))
    while stack and nodes[stack[-1]]["level"] >= level:
        stack.pop()
    parent_id = stack[-1] if stack else f"{document_id}__root"
    node_id = f"{document_id}__h{len(nodes):04d}"
    node = {
        "node_id": node_id,
        "parent_node_id": parent_id,
        "level": level,
        "label": label,
        "number": number,
        "kind": kind,
        "page_number": page_number,
        "source_block_id": block_id,
        "detection_source": source,
        "confidence": round(confidence, 3),
        "style": style,
        "children": [],
    }
    nodes[node_id] = node
    nodes[parent_id]["children"].append(node_id)
    stack.append(node_id)
    return node_id


def path_for(nodes, node_id):
    ids = []
    while node_id:
        node = nodes[node_id]
        ids.append(node_id)
        node_id = node.get("parent_node_id")
    ids.reverse()
    return [
        {
            "node_id": node_id,
            "level": nodes[node_id]["level"],
            "label": nodes[node_id]["label"],
            "number": nodes[node_id].get("number"),
            "kind": nodes[node_id]["kind"],
            "page_number": nodes[node_id].get("page_number"),
        }
        for node_id in ids
    ]


def ancestry_stack(nodes, node_id):
    ids = []
    while node_id:
        ids.append(node_id)
        node_id = nodes[node_id].get("parent_node_id")
    return list(reversed(ids))


def regulatory_reference(content):
    flat = re.sub(r"\s+", " ", content).strip()
    match = NUMERIC.match(flat)
    if match:
        first_integer = int(match.group("number").split(".")[0])
        remainder = match.group("label").strip()
        if first_integer <= 500 and not re.match(r"^(?:of\b|years?\b|days?\b|months?\b|%|₹|rs\.?\b)", remainder, re.I):
            return {"type": "numbered_provision", "number": match.group("number")}
    match = LETTER.match(flat)
    if match:
        return {"type": "clause", "number": match.group("number")}
    match = ROMAN.match(flat)
    if match:
        return {"type": "subclause", "number": match.group("number")}
    return None


def build_document(clean_path):
    clean_doc = json.loads(clean_path.read_text(encoding="utf-8"))
    raw_doc = json.loads((RAW_DOCS / clean_path.name).read_text(encoding="utf-8"))
    metadata = clean_doc["document_metadata"]
    document_id = metadata["document_id"]
    body_size = body_font_size(raw_doc)
    pdf_path = pdf_for_document(metadata)
    pdf = fitz.open(pdf_path)
    toc = clean_toc(pdf.get_toc(simple=True))
    pdf.close()
    toc_by_page = defaultdict(list)
    for entry in toc:
        toc_by_page[entry["page_number"]].append(entry)

    root_id = f"{document_id}__root"
    nodes = {
        root_id: {
            "node_id": root_id,
            "parent_node_id": None,
            "level": 0,
            "label": metadata.get("title") or metadata.get("source_file_name"),
            "number": None,
            "kind": "document",
            "page_number": 1,
            "source_block_id": None,
            "detection_source": "document_metadata",
            "confidence": 1.0,
            "style": None,
            "children": [],
        }
    }
    stack = [root_id]
    block_mappings = []
    structured_pages = []

    for page, raw_page in zip(clean_doc["pages"], raw_doc["pages"]):
        page_number = page["page_number"]
        page_toc_nodes = []
        for entry in toc_by_page.get(page_number, []):
            node_id = add_node(
                nodes, stack,
                document_id=document_id,
                label=entry["title"],
                level=entry["level"],
                page_number=page_number,
                block_id=None,
                kind="bookmark_heading",
                number=None,
                source="pdf_bookmark",
                confidence=1.0,
            )
            page_toc_nodes.append(node_id)
        if page_toc_nodes:
            # Start the page at its first bookmark. Later matched headings reset
            # the active stack to the exact bookmark encountered in reading order.
            stack = ancestry_stack(nodes, page_toc_nodes[0])

        structured_blocks = []
        for ordinal, block in enumerate(page["cleaned_blocks"]):
            block_id = f"{document_id}__p{page_number:04d}__b{ordinal:04d}"
            raw_block_index = block["source_block_index"]
            raw_block = raw_page["blocks"][raw_block_index]
            candidate = heading_candidate(
                block["content"], raw_block, body_size, block.get("within_detected_table", False)
            )
            heading_node_id = None
            if candidate:
                candidate_norm = normalize(candidate["label"])
                same_page_toc = [
                    node_id
                    for node_id in page_toc_nodes
                    if SequenceMatcher(None, candidate_norm, normalize(nodes[node_id]["label"])).ratio() >= 0.66
                ]
                if same_page_toc:
                    heading_node_id = max(
                        same_page_toc,
                        key=lambda node_id: SequenceMatcher(None, candidate_norm, normalize(nodes[node_id]["label"])).ratio(),
                    )
                    nodes[heading_node_id]["source_block_id"] = block_id
                    nodes[heading_node_id]["style"] = candidate["style"]
                    stack = ancestry_stack(nodes, heading_node_id)
                else:
                    level = candidate["suggested_level"]
                    if level is None:
                        level = min(6, nodes[stack[-1]]["level"] + 1)
                    heading_node_id = add_node(
                        nodes, stack,
                        document_id=document_id,
                        label=candidate["label"],
                        level=level,
                        page_number=page_number,
                        block_id=block_id,
                        kind=candidate["kind"],
                        number=candidate["number"],
                        source=candidate["detection"],
                        confidence=candidate["confidence"],
                        style=candidate["style"],
                    )
            active_node_id = heading_node_id or stack[-1]
            heading_path = path_for(nodes, active_node_id)
            mapping = {
                "block_id": block_id,
                "document_id": document_id,
                "page_number": page_number,
                "source_block_index": raw_block_index,
                "is_heading": heading_node_id is not None,
                "heading_node_id": heading_node_id,
                "active_heading_node_id": active_node_id,
                "heading_path": heading_path,
                "heading_path_text": " > ".join(item["label"] for item in heading_path),
                "regulatory_reference": regulatory_reference(block["content"]),
                "citation_anchor": {
                    "title": metadata.get("title"),
                    "heading_path": [item["label"] for item in heading_path[1:]],
                    "page_number": page_number,
                    "regulatory_reference": regulatory_reference(block["content"]),
                    "source_url": metadata.get("source_url"),
                },
            }
            block_mappings.append(mapping)
            structured_blocks.append({**block, **mapping})
        structured_pages.append({
            "page_number": page_number,
            "blocks": structured_blocks,
            "tables": page.get("tables", []),
        })

    # A compact tree with nested children objects for inspection.
    def nested(node_id):
        node = {k: v for k, v in nodes[node_id].items() if k != "children"}
        node["children"] = [nested(child_id) for child_id in nodes[node_id]["children"]]
        return node

    hierarchy = {
        "document_metadata": {
            **metadata,
            "source_cleaned_file": str(clean_path.relative_to(CLEAN_ROOT)),
            "source_cleaned_sha256": sha256_file(clean_path),
            "hierarchy_format_version": "1.0.0",
            "hierarchy_generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        },
        "hierarchy_summary": {
            "body_font_size": round(body_size, 3),
            "pdf_bookmark_count": len(toc),
            "hierarchy_node_count": len(nodes) - 1,
            "maximum_level": max(node["level"] for node in nodes.values()),
            "detection_source_counts": dict(Counter(node["detection_source"] for node in nodes.values() if node["level"] > 0)),
        },
        "tree": nested(root_id),
        "nodes": [nodes[node_id] for node_id in nodes],
        "block_heading_map": block_mappings,
    }
    structured_document = {
        "document_metadata": hierarchy["document_metadata"],
        "hierarchy_reference": f"document_hierarchies/{document_id}.hierarchy.json",
        "pages": structured_pages,
    }
    return hierarchy, structured_document


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    HIERARCHIES.mkdir(parents=True)
    STRUCTURED.mkdir(parents=True)
    combined = []
    combined_map = []
    manifest = []
    for clean_path in sorted(CLEAN_DOCS.glob("*.json")):
        hierarchy, structured = build_document(clean_path)
        document_id = hierarchy["document_metadata"]["document_id"]
        hierarchy_path = HIERARCHIES / f"{document_id}.hierarchy.json"
        structured_path = STRUCTURED / f"{document_id}.structured.json"
        hierarchy_path.write_text(json.dumps(hierarchy, ensure_ascii=False, indent=2), encoding="utf-8")
        structured_path.write_text(json.dumps(structured, ensure_ascii=False, indent=2), encoding="utf-8")
        combined.append({
            "document_id": document_id,
            "title": hierarchy["document_metadata"].get("title"),
            "tree": hierarchy["tree"],
        })
        combined_map.extend(hierarchy["block_heading_map"])
        summary = hierarchy["hierarchy_summary"]
        manifest.append({
            "document_id": document_id,
            "title": hierarchy["document_metadata"].get("title"),
            "hierarchy_file": str(hierarchy_path.relative_to(OUT)),
            "structured_file": str(structured_path.relative_to(OUT)),
            **summary,
            "hierarchy_sha256": sha256_file(hierarchy_path),
            "structured_sha256": sha256_file(structured_path),
        })

    (OUT / "document_hierarchy.json").write_text(json.dumps({"documents": combined}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "block_heading_map.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in combined_map), encoding="utf-8")
    (OUT / "document_manifest.json").write_text(json.dumps({"documents": manifest}, ensure_ascii=False, indent=2), encoding="utf-8")

    level_counts = Counter()
    source_counts = Counter()
    integrity_errors = []
    for path in HIERARCHIES.glob("*.json"):
        hierarchy = json.loads(path.read_text(encoding="utf-8"))
        node_registry = {node["node_id"]: node for node in hierarchy["nodes"]}
        for node in hierarchy["nodes"]:
            if node["level"]:
                level_counts[str(node["level"])] += 1
                source_counts[node["detection_source"]] += 1
            parent_id = node.get("parent_node_id")
            if parent_id:
                if parent_id not in node_registry:
                    integrity_errors.append(f"{path.name}: missing parent {parent_id}")
                elif node_registry[parent_id]["level"] >= node["level"]:
                    integrity_errors.append(f"{path.name}: non-increasing level for {node['node_id']}")
    expected_blocks = sum(
        len(page["cleaned_blocks"])
        for path in CLEAN_DOCS.glob("*.json")
        for page in json.loads(path.read_text(encoding="utf-8"))["pages"]
    )
    validation = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "documents_generated": len(manifest),
        "hierarchy_nodes": sum(x["hierarchy_node_count"] for x in manifest),
        "mapped_blocks": len(combined_map),
        "level_counts": dict(sorted(level_counts.items())),
        "detection_source_counts": dict(sorted(source_counts.items())),
        "checks": {
            "ten_hierarchies_generated": len(manifest) == 10,
            "all_blocks_have_paths": all(row["heading_path"] for row in combined_map),
            "all_paths_start_with_document": all(row["heading_path"][0]["kind"] == "document" for row in combined_map),
            "all_citations_have_page": all(row["citation_anchor"]["page_number"] >= 1 for row in combined_map),
            "all_documents_have_nodes": all(x["hierarchy_node_count"] > 0 for x in manifest),
            "all_cleaned_blocks_mapped": len(combined_map) == expected_blocks,
            "block_ids_unique": len({row["block_id"] for row in combined_map}) == len(combined_map),
            "parent_child_levels_valid": not integrity_errors,
        },
        "integrity_errors": integrity_errors,
        "documents": manifest,
    }
    (OUT / "validation_report.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(Path(__file__), OUT / "build_document_hierarchy.py")
    example = max(combined_map, key=lambda row: len(row["heading_path"]))
    chunk_contract = {
        "purpose": "Metadata fields to copy from structural blocks into every Step 6 chunk.",
        "required_chunk_metadata": [
            "document_id",
            "title",
            "heading_node_ids",
            "heading_path",
            "heading_path_text",
            "regulatory_reference",
            "page_start",
            "page_end",
            "source_url",
        ],
        "example": {
            "document_id": example["document_id"],
            "title": example["citation_anchor"]["title"],
            "heading_node_ids": [item["node_id"] for item in example["heading_path"]],
            "heading_path": [item["label"] for item in example["heading_path"]],
            "heading_path_text": example["heading_path_text"],
            "regulatory_reference": example["regulatory_reference"],
            "page_start": example["page_number"],
            "page_end": example["page_number"],
            "source_url": example["citation_anchor"]["source_url"],
        },
    }
    (OUT / "chunk_metadata_contract.json").write_text(json.dumps(chunk_contract, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# RBI Regulatory Document Hierarchy\n\n"
        "## Where the hierarchy is stored\n\n"
        "- `document_hierarchy.json`: combined nested hierarchy for all documents.\n"
        "- `document_hierarchies/<document_id>.hierarchy.json`: authoritative per-document tree, flat node registry, and block mapping.\n"
        "- `block_heading_map.jsonl`: fast lookup from every cleaned block to its full heading path and citation anchor.\n"
        "- `structured_documents/<document_id>.structured.json`: cleaned blocks with `heading_path`, `heading_path_text`, and `citation_anchor` embedded directly. This is the recommended input to Step 6 chunking.\n\n"
        "During chunking, copy the block-derived `heading_path`, `active_heading_node_id`, page range, title, and source URL into each chunk's metadata. Retrieval can then return this metadata directly, and citation generation can display `title > heading path > page` without reparsing the PDF.\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
