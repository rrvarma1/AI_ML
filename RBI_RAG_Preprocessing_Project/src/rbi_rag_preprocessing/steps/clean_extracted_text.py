from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

RAW_ROOT = Path(os.environ["RBI_RAW_OUTPUT_DIR"]).expanduser().resolve()
RAW_DOCS = RAW_ROOT / "raw_extractions"
OUT = Path(os.environ["RBI_CLEAN_OUTPUT_DIR"]).expanduser().resolve()
CLEAN_DOCS = OUT / "cleaned_documents"

PAGE_NUMBER_PATTERNS = [
    re.compile(r"^\s*(?:page\s*)?\d+\s*(?:of\s*\d+)?\s*$", re.I),
    re.compile(r"^\s*[-–—]\s*\d+\s*[-–—]\s*$"),
]
STANDALONE_WEB_PATTERNS = [
    re.compile(r"^\s*(?:https?://)?(?:www\.)?rbi\.org\.in/?\s*$", re.I),
    re.compile(r"^\s*(?:reserve bank of india\s+)?website\s*:?\s*(?:https?://)?(?:www\.)?rbi\.org\.in/?\s*$", re.I),
    re.compile(r"^\s*(?:home|about us|notifications|press releases|master directions|master circulars)(?:\s*[|>]\s*(?:home|about us|notifications|press releases|master directions|master circulars))*\s*$", re.I),
]
ENUMERATOR = re.compile(
    r"^\s*(?:\(?[a-z]\)|\(?[ivxlcdm]+\)|\(?\d+(?:\.\d+)*\)?[.)]?|[a-z]\.|[ivxlcdm]+\.)\s+",
    re.I,
)
LEGAL_QUALIFIER = re.compile(
    r"\b(?:provided\s+that|provided\s+further\s+that|except\s+where|notwithstanding|subject\s+to|save\s+as|unless\s+otherwise|proviso|exception)\b",
    re.I,
)
STRUCTURAL = re.compile(
    r"\b(?:chapter|section|annex(?:ure)?|appendix|schedule|part|definition|explanation|amend(?:ed|ment)|substitut(?:ed|ion)|inserted|omitted)\b",
    re.I,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def normalize_space(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    return text.strip()


def comparison_key(text: str, normalize_numbers: bool = False) -> str:
    text = unicodedata.normalize("NFKC", normalize_space(text)).lower()
    if normalize_numbers:
        text = re.sub(r"\bpage\s+\d+\s+of\s+\d+\b", "page # of #", text)
        text = re.sub(r"^\s*[-–—]?\s*\d+\s*[-–—]?\s*$", "<page_number>", text)
    return re.sub(r"\s+", " ", text).strip()


def is_page_number(text: str) -> bool:
    return any(pattern.fullmatch(text) for pattern in PAGE_NUMBER_PATTERNS)


def is_margin_page_number(text: str, bbox, page_width: float, page_height: float) -> bool:
    """Remove page numbers only when their geometry confirms a true margin item.

    The wider header/footer band is useful for repeated running text, but it is
    too broad for bare numbers because RBI tables can begin near the top margin.
    "Page N of M" is explicit; a bare number must be in the extreme 7% band and
    approximately centered horizontally.
    """
    if not is_page_number(text) or not bbox or len(bbox) < 4:
        return False
    explicit = bool(re.search(r"\bpage\s*\d+|\bof\s*\d+", text, re.I))
    x0, y0, x1, y1 = bbox
    if explicit:
        return y1 <= page_height * 0.14 or y0 >= page_height * 0.86
    horizontally_centered = page_width * 0.38 <= (x0 + x1) / 2 <= page_width * 0.62
    extreme_margin = y1 <= page_height * 0.07 or y0 >= page_height * 0.92
    return horizontally_centered and extreme_margin


def is_standalone_navigation(text: str) -> bool:
    return any(pattern.fullmatch(text) for pattern in STANDALONE_WEB_PATTERNS)


def margin_zone(line_bbox, page_height: float) -> str | None:
    if not line_bbox or len(line_bbox) < 4:
        return None
    y0, y1 = line_bbox[1], line_bbox[3]
    if y1 <= page_height * 0.14:
        return "header"
    if y0 >= page_height * 0.86:
        return "footer"
    return None


def rectangles_intersect(first, second) -> bool:
    if not first or not second or len(first) < 4 or len(second) < 4:
        return False
    return not (
        first[2] <= second[0]
        or first[0] >= second[2]
        or first[3] <= second[1]
        or first[1] >= second[3]
    )


def line_records(page):
    for block_index, block in enumerate(page.get("blocks", [])):
        if block.get("block_type") != "text":
            continue
        for line_index, line in enumerate(block.get("lines", [])):
            text = normalize_space(line.get("text", ""))
            yield {
                "block_index": block_index,
                "line_index": line_index,
                "text": text,
                "bbox": line.get("bbox", block.get("bbox")),
                "zone": margin_zone(line.get("bbox", block.get("bbox")), page["height"]),
            }


def detect_repeated_margin_lines(pages):
    occurrences = defaultdict(set)
    examples = {}
    for page in pages:
        for line in line_records(page):
            if not line["zone"] or not line["text"]:
                continue
            key = (line["zone"], comparison_key(line["text"], normalize_numbers=True))
            occurrences[key].add(page["page_number"])
            examples.setdefault(key, line["text"])

    page_count = len(pages)
    minimum = max(3, int(page_count * 0.30 + 0.999))
    repeated = {}
    for key, page_numbers in occurrences.items():
        zone, normalized = key
        if normalized == "<page_number>":
            continue
        if len(page_numbers) >= minimum:
            repeated[key] = {
                "zone": zone,
                "normalized_text": normalized,
                "example": examples[key],
                "page_occurrences": len(page_numbers),
            }
    return repeated


def should_preserve_repeated_text(text: str) -> bool:
    # A repeated margin line containing a legal qualifier or structural amendment
    # is retained even if it meets the frequency rule.
    return bool(LEGAL_QUALIFIER.search(text) or STRUCTURAL.search(text) and len(text) > 140)


def is_heading(line: str) -> bool:
    words = re.findall(r"[A-Za-z]+", line)
    if not words or len(words) > 16:
        return False
    upper_ratio = sum(word.isupper() for word in words) / len(words)
    return upper_ratio >= 0.75 or bool(re.match(r"^(?:CHAPTER|ANNEX(?:URE)?|APPENDIX|SCHEDULE|PART)\b", line, re.I))


def join_wrapped_lines(lines):
    if not lines:
        return "", {"line_joins": 0, "dehyphenations": 0}
    output = normalize_space(lines[0])
    joins = dehyphenations = 0
    for raw_next in lines[1:]:
        nxt = normalize_space(raw_next)
        if not nxt:
            continue
        if output.endswith("-") and re.search(r"[A-Za-z]-$", output) and re.match(r"^[a-z]", nxt):
            output = output[:-1] + nxt
            dehyphenations += 1
            continue
        # Keep clause and list boundaries visible; do not merge a new enumerated item
        # into the previous clause. Within one block, a newline is retained here.
        if ENUMERATOR.match(nxt) or is_heading(nxt):
            output += "\n" + nxt
        else:
            output += " " + nxt
            joins += 1
    output = re.sub(r"[ \t]+", " ", output)
    output = re.sub(r" *\n *", "\n", output).strip()
    return output, {"line_joins": joins, "dehyphenations": dehyphenations}


def clean_document(raw_path: Path):
    source = json.loads(raw_path.read_text(encoding="utf-8"))
    pages = source["pages"]
    repeated = detect_repeated_margin_lines(pages)
    audit = []
    cleaned_pages = []
    counts = Counter()
    previous_page_last_key = None

    for page in pages:
        page_blocks = []
        page_seen = set()
        page_audit = []
        for block_index, block in enumerate(page.get("blocks", [])):
            if block.get("block_type") != "text":
                continue
            within_detected_table = any(
                rectangles_intersect(block.get("bbox"), table.get("bbox"))
                for table in page.get("tables", [])
            )
            retained_lines = []
            source_line_indices = []
            for line_index, line in enumerate(block.get("lines", [])):
                text = normalize_space(line.get("text", ""))
                zone = margin_zone(line.get("bbox", block.get("bbox")), page["height"])
                reason = None
                if not text:
                    reason = "empty_line"
                elif zone and is_margin_page_number(
                    text,
                    line.get("bbox", block.get("bbox")),
                    page["width"],
                    page["height"],
                ):
                    reason = "margin_page_number"
                elif is_standalone_navigation(text):
                    reason = "rbi_website_navigation"
                elif zone:
                    key = (zone, comparison_key(text, normalize_numbers=True))
                    if key in repeated and not should_preserve_repeated_text(text):
                        reason = f"repeated_{zone}"
                if reason:
                    entry = {
                        "document_id": source["document_metadata"]["document_id"],
                        "page_number": page["page_number"],
                        "block_index": block_index,
                        "line_index": line_index,
                        "type": reason,
                        "text": text,
                        "bbox": line.get("bbox", block.get("bbox")),
                    }
                    page_audit.append(entry)
                    audit.append(entry)
                    counts[reason] += 1
                else:
                    retained_lines.append(text)
                    source_line_indices.append(line_index)

            cleaned, transformations = join_wrapped_lines(retained_lines)
            counts.update(transformations)
            if not cleaned:
                counts["empty_blocks_removed"] += 1
                continue
            key = comparison_key(cleaned)
            # Exact duplicate blocks on the same page are extraction artifacts.
            # Legal qualifiers are still preserved at least once.
            if key in page_seen and not within_detected_table:
                entry = {
                    "document_id": source["document_metadata"]["document_id"],
                    "page_number": page["page_number"],
                    "block_index": block_index,
                    "type": "duplicate_block_same_page",
                    "text": cleaned,
                    "bbox": block.get("bbox"),
                }
                page_audit.append(entry)
                audit.append(entry)
                counts["duplicate_blocks_removed"] += 1
                continue
            # Remove only exact first-block overlap across a page boundary.
            if not page_blocks and previous_page_last_key and key == previous_page_last_key and len(key) >= 80:
                entry = {
                    "document_id": source["document_metadata"]["document_id"],
                    "page_number": page["page_number"],
                    "block_index": block_index,
                    "type": "duplicate_block_page_boundary",
                    "text": cleaned,
                    "bbox": block.get("bbox"),
                }
                page_audit.append(entry)
                audit.append(entry)
                counts["duplicate_boundary_blocks_removed"] += 1
                continue
            page_seen.add(key)
            page_blocks.append({
                "source_block_index": block_index,
                "source_line_indices": source_line_indices,
                "bbox": block.get("bbox"),
                "content": cleaned,
                "contains_legal_qualifier": bool(LEGAL_QUALIFIER.search(cleaned)),
                "contains_structural_marker": bool(STRUCTURAL.search(cleaned)),
                "within_detected_table": within_detected_table,
            })

        previous_page_last_key = comparison_key(page_blocks[-1]["content"]) if page_blocks else previous_page_last_key
        cleaned_text = "\n\n".join(block["content"] for block in page_blocks).strip()
        cleaned_pages.append({
            "page_number": page["page_number"],
            "width": page["width"],
            "height": page["height"],
            "cleaned_text": cleaned_text,
            "cleaned_blocks": page_blocks,
            "tables": page.get("tables", []),
            "image_references": page.get("image_references", []),
            "links": page.get("links", []),
            "cleaning_audit": page_audit,
        })

    metadata = dict(source["document_metadata"])
    metadata.update({
        "source_raw_file": str(raw_path.relative_to(RAW_ROOT)),
        "source_raw_sha256": sha256_file(raw_path),
        "cleaning_format_version": "1.0.0",
        "cleaned_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "cleaning_policy": "conservative_regulatory_text_v1",
    })
    document = {
        "document_metadata": metadata,
        "cleaning_summary": {
            "detected_repeated_margin_lines": list(repeated.values()),
            "operation_counts": dict(sorted(counts.items())),
            "removed_element_count": len(audit),
        },
        "pages": cleaned_pages,
    }
    return document, audit


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    CLEAN_DOCS.mkdir(parents=True)
    all_audit = []
    manifest = []
    raw_manifest = json.loads((RAW_ROOT / "document_manifest.json").read_text(encoding="utf-8"))
    manifest_by_id = {x["document_id"]: x for x in raw_manifest["documents"]}

    for raw_path in sorted(RAW_DOCS.glob("*.json")):
        cleaned, audit = clean_document(raw_path)
        document_id = cleaned["document_metadata"]["document_id"]
        out_path = CLEAN_DOCS / f"{document_id}.json"
        out_path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
        all_audit.extend(audit)
        raw_info = manifest_by_id[document_id]
        manifest.append({
            "document_id": document_id,
            "title": cleaned["document_metadata"].get("title"),
            "cleaned_file": str(out_path.relative_to(OUT)),
            "page_count": len(cleaned["pages"]),
            "raw_character_count": raw_info["text_character_count"],
            "cleaned_character_count": sum(len(p["cleaned_text"]) for p in cleaned["pages"]),
            "removed_element_count": cleaned["cleaning_summary"]["removed_element_count"],
            "table_count": sum(len(p["tables"]) for p in cleaned["pages"]),
            "cleaned_json_sha256": sha256_file(out_path),
        })

    audit_path = OUT / "cleaning_audit.jsonl"
    audit_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in all_audit), encoding="utf-8")
    (OUT / "document_manifest.json").write_text(json.dumps({"documents": manifest}, ensure_ascii=False, indent=2), encoding="utf-8")

    required_patterns = {
        "section_numbers": re.compile(r"\b\d+(?:\.\d+)+\b"),
        "clause_letters": re.compile(r"(?:^|\n)\s*\([a-z]\)", re.I),
        "roman_clauses": re.compile(r"(?:^|\n)\s*\([ivxlcdm]+\)", re.I),
        "definitions": re.compile(r"\b(?:means|definition|defined\s+as)\b", re.I),
        "legal_qualifiers": LEGAL_QUALIFIER,
        "annexures": re.compile(r"\bannex(?:ure)?\b", re.I),
        "amendment_notes": re.compile(r"\b(?:amended|substituted|inserted|omitted|corrigendum)\b", re.I),
    }
    corpus_text = "\n".join(
        p["cleaned_text"]
        for path in CLEAN_DOCS.glob("*.json")
        for p in json.loads(path.read_text(encoding="utf-8"))["pages"]
    )
    validation = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "documents_expected": 10,
        "documents_generated": len(manifest),
        "pages_generated": sum(x["page_count"] for x in manifest),
        "raw_character_count": sum(x["raw_character_count"] for x in manifest),
        "cleaned_character_count": sum(x["cleaned_character_count"] for x in manifest),
        "removed_element_count": len(all_audit),
        "audit_type_counts": dict(sorted(Counter(x["type"] for x in all_audit).items())),
        "preservation_signal_counts": {name: len(pattern.findall(corpus_text)) for name, pattern in required_patterns.items()},
        "checks": {
            "ten_documents_generated": len(manifest) == 10,
            "all_237_pages_preserved": sum(x["page_count"] for x in manifest) == 237,
            "all_documents_nonempty": all(x["cleaned_character_count"] > 0 for x in manifest),
            "tables_preserved": sum(x["table_count"] for x in manifest) == 83,
            "legal_qualifiers_present": bool(LEGAL_QUALIFIER.search(corpus_text)),
            "annexures_present": bool(re.search(r"\bannex(?:ure)?\b", corpus_text, re.I)),
            "amendment_notes_present": bool(re.search(r"\b(?:amended|substituted|inserted|omitted|corrigendum)\b", corpus_text, re.I)),
        },
        "documents": manifest,
    }
    (OUT / "validation_report.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(Path(__file__), OUT / "clean_extracted_text.py")
    (OUT / "README.md").write_text(
        "# RBI Cleaned Extraction Layer\n\n"
        "This layer is derived from the loss-minimized raw extraction. It removes repeated positional headers/footers, margin page numbers, standalone RBI navigation text, exact duplicate extraction blocks, broken line wraps, soft hyphens, line-end hyphenation, empty blocks, and excess whitespace. It conservatively preserves regulatory numbering, clause lettering, definitions, legal qualifiers, footnotes, tables, annexures, and amendment notes.\n\n"
        "Every removed line/block appears in `cleaning_audit.jsonl` with its source page and bounding box. The raw extraction remains unchanged.\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
