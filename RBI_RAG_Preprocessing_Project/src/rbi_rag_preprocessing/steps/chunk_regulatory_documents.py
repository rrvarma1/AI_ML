"""Step 6: Docling conversion followed by LlamaIndex hierarchical chunking."""
from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CORPUS = Path(os.environ["RBI_CORPUS_DIR"]).expanduser().resolve()
HIERARCHY_ROOT = Path(os.environ["RBI_HIERARCHY_OUTPUT_DIR"]).expanduser().resolve()
STRUCTURED_DOCS = HIERARCHY_ROOT / "structured_documents"
OUT = Path(os.environ["RBI_CHUNK_OUTPUT_DIR"]).expanduser().resolve()
DOC_OUTPUT = OUT / "documents"
DOCLING_OUTPUT = OUT / "docling_documents"
TARGET_TOKENS = int(os.getenv("RBI_CHUNK_TARGET_TOKENS", "700"))
MAX_TOKENS = int(os.getenv("RBI_CHUNK_MAX_TOKENS", "1000"))
MIN_TOKENS = int(os.getenv("RBI_CHUNK_MIN_TOKENS", "150"))
OVERLAP_TOKENS = int(os.getenv("RBI_CHUNK_OVERLAP_TOKENS", "100"))
PAGE_MARKER = re.compile(r"<!-- RBI_PAGE:(\d+) -->")
MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
LEGAL_DEPENDENCY = re.compile(
    r"^\s*(?:provided\s+(?:further\s+)?that|except\s+where|notwithstanding|"
    r"subject\s+to|unless\s+otherwise|save\s+as(?:\s+otherwise)?|exception\s*:)", re.I
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[\\/:*?\"<>|]", " ", text)
    return " ".join(re.sub(r"[^a-zA-Z0-9]+", " ", text).lower().split())


def resolve_source_pdf(metadata: dict[str, Any], corpus: Path = CORPUS) -> tuple[Path, str]:
    """Resolve a PDF without trusting inventory titles as literal filesystem paths."""
    pdfs = sorted(corpus.glob("*.pdf"))
    for key in ("source_file_name", "file_name", "inventory_file_name"):
        value = metadata.get(key)
        if value:
            candidate = corpus / str(value)
            if candidate.is_file():
                return candidate, f"exact:{key}"

    expected_hash = str(metadata.get("pdf_sha256") or "").lower()
    if expected_hash:
        hash_matches = [pdf for pdf in pdfs if sha256_file(pdf) == expected_hash]
        if len(hash_matches) == 1:
            return hash_matches[0], "sha256"

    wanted_names = {
        normalized_name(metadata.get(key))
        for key in ("source_file_name", "file_name", "inventory_file_name", "title")
        if metadata.get(key)
    }
    name_matches = [pdf for pdf in pdfs if normalized_name(pdf.name) in wanted_names]
    if len(name_matches) == 1:
        return name_matches[0], "normalized_filename"

    details = ", ".join(str(metadata.get(key)) for key in ("source_file_name", "file_name") if metadata.get(key))
    raise FileNotFoundError(
        f"Could not uniquely resolve the PDF for {metadata.get('document_id')} from: {details}. "
        f"Found {len(name_matches)} normalized filename matches in {corpus}."
    )


def load_frameworks():
    """Import the heavyweight Step-6 frameworks only when this stage runs."""
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from llama_index.core import Document
        from llama_index.core.node_parser import HierarchicalNodeParser, SentenceSplitter, get_leaf_nodes
        from llama_index.core.schema import NodeRelationship
    except ImportError as exc:
        raise RuntimeError(
            "Step 6 requires Docling and LlamaIndex. Run `python -m pip install -e .` "
            "inside the project virtual environment."
        ) from exc
    return locals()


def load_document_metadata() -> list[dict[str, Any]]:
    records = []
    for path in sorted(STRUCTURED_DOCS.glob("*.structured.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = dict(payload["document_metadata"])
        metadata["structured_source"] = str(path.relative_to(HIERARCHY_ROOT))
        records.append(metadata)
    return records


def build_converter(f):
    options = f["PdfPipelineOptions"]()
    options.do_ocr = True
    options.do_table_structure = True
    return f["DocumentConverter"](
        allowed_formats=[f["InputFormat"].PDF],
        format_options={f["InputFormat"].PDF: f["PdfFormatOption"](pipeline_options=options)},
    )


def markdown_with_page_markers(docling_document, page_count: int) -> str:
    pages = []
    for page_number in range(1, page_count + 1):
        page = docling_document.export_to_markdown(
            page_no=page_number, traverse_pictures=True
        ).strip()
        if page:
            pages.append(f"<!-- RBI_PAGE:{page_number} -->\n\n{page}")
    return "\n\n".join(pages)


def make_hierarchy_parser(f):
    """Build two explicit SentenceSplitter levels under HierarchicalNodeParser."""
    parent_id = f"sentence_parent_{MAX_TOKENS}"
    leaf_id = f"sentence_leaf_{TARGET_TOKENS}"
    parser_map = {
        parent_id: f["SentenceSplitter"](
            chunk_size=MAX_TOKENS, chunk_overlap=0,
            include_metadata=True, include_prev_next_rel=True,
        ),
        leaf_id: f["SentenceSplitter"](
            chunk_size=TARGET_TOKENS, chunk_overlap=OVERLAP_TOKENS,
            include_metadata=True, include_prev_next_rel=True,
        ),
    }
    return f["HierarchicalNodeParser"].from_defaults(
        node_parser_ids=[parent_id, leaf_id], node_parser_map=parser_map,
        include_metadata=True, include_prev_next_rel=True,
    )


def page_for_offset(markdown: str, offset: int) -> int:
    page = 1
    for match in PAGE_MARKER.finditer(markdown, 0, max(0, offset) + 1):
        page = int(match.group(1))
    return page


def heading_path_for_offset(markdown: str, offset: int, title: str) -> list[str]:
    levels: dict[int, str] = {}
    for match in MARKDOWN_HEADING.finditer(markdown, 0, max(0, offset) + 1):
        level = len(match.group(1))
        levels[level] = match.group(2).strip()
        for deeper in [value for value in levels if value > level]:
            del levels[deeper]
    return [title] + [levels[level] for level in sorted(levels)]


def locate_leaf(markdown: str, text: str, cursor: int) -> tuple[int, int]:
    probe = text.strip()
    if not probe:
        return cursor, cursor
    start = markdown.find(probe, max(0, cursor - 6000))
    if start < 0:
        start = markdown.find(probe[: min(160, len(probe))], max(0, cursor - 6000))
    if start < 0:
        start = cursor
    return start, min(len(markdown), start + len(probe))


def relation_ids(node, relationship) -> str | list[str] | None:
    related = node.relationships.get(relationship)
    if related is None:
        return None
    if isinstance(related, list):
        return [item.node_id for item in related]
    return related.node_id


def node_record(node, relationship_cls) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "text": node.get_content(),
        "metadata": dict(node.metadata),
        "relationships": {
            "parent": relation_ids(node, relationship_cls.PARENT),
            "children": relation_ids(node, relationship_cls.CHILD),
            "previous": relation_ids(node, relationship_cls.PREVIOUS),
            "next": relation_ids(node, relationship_cls.NEXT),
        },
    }


def process_document(metadata, converter, parser, f):
    source, source_resolution_method = resolve_source_pdf(metadata)
    docling_document = converter.convert(source).document
    (DOCLING_OUTPUT / f"{metadata['document_id']}.docling.json").write_text(
        json.dumps(docling_document.export_to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = markdown_with_page_markers(docling_document, int(metadata["page_count"]))
    (DOCLING_OUTPUT / f"{metadata['document_id']}.md").write_text(markdown, encoding="utf-8")
    inherited = {
        key: value for key, value in metadata.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }
    inherited["resolved_source_file_name"] = source.name
    inherited["source_resolution_method"] = source_resolution_method
    document = f["Document"](text=markdown, metadata=inherited, id_=metadata["document_id"])
    all_nodes = parser.get_nodes_from_documents([document], show_progress=False)
    leaves = f["get_leaf_nodes"](all_nodes)
    chunks = []
    cursor = 0
    for index, node in enumerate(leaves, start=1):
        raw_text = node.get_content()
        body = PAGE_MARKER.sub("", raw_text).strip()
        start, end = locate_leaf(markdown, raw_text, cursor)
        cursor = max(cursor, start)
        page_start, page_end = page_for_offset(markdown, start), page_for_offset(markdown, end)
        heading_path = heading_path_for_offset(markdown, start, metadata["title"])
        content = (
            f"Document: {metadata['title']}\nRegulatory path: {' > '.join(heading_path)}\n"
            f"Source pages: {page_start}-{page_end}\n\n{body}"
        ).strip()
        chunks.append({
            "content": content,
            "metadata": {
                **inherited,
                "chunk_id": f"{metadata['document_id']}__chunk_{index:04d}",
                "llamaindex_node_id": node.node_id,
                "parent_node_id": relation_ids(node, f["NodeRelationship"].PARENT),
                "start_index": start,
                "end_index": end,
                "page_start": page_start, "page_end": page_end,
                "heading_path": heading_path, "heading_path_text": " > ".join(heading_path),
                "chunk_strategy": "docling_llamaindex_hierarchical_sentence",
                "token_count": len(raw_text.split()),
                "token_count_unit": "whitespace_words_estimate",
                "starts_with_legal_dependency": bool(LEGAL_DEPENDENCY.match(body)),
            },
        })
    hierarchy = [node_record(node, f["NodeRelationship"]) for node in all_nodes]
    return chunks, hierarchy, len(markdown)


def print_chunking_summary(manifest: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> None:
    """Print corpus totals and a complete split trace for the longest input document."""
    line = "-" * 100
    print(f"Documents in:  {len(manifest)}")
    print(f"Chunks out:    {len(chunks)}")
    print(line)
    if not manifest:
        print("No documents were processed.")
        return

    longest = max(manifest, key=lambda item: item["source_character_count"])
    child_chunks = [
        chunk for chunk in chunks
        if chunk["metadata"]["document_id"] == longest["document_id"]
    ]
    print(
        f'Longest doc: "{longest["title"]}" '
        f'({longest["source_character_count"]} chars)\n'
    )
    print(f"It split into {len(child_chunks)} final leaf chunks:\n")
    for index, chunk in enumerate(child_chunks):
        metadata = chunk["metadata"]
        print(
            f'[chunk {index} | start_index={metadata["start_index"]} | '
            f'end_index={metadata["end_index"]} | {len(chunk["content"])} chars]'
        )
        print(chunk["content"])
        print(line)


def main() -> None:
    if not STRUCTURED_DOCS.is_dir():
        raise FileNotFoundError(f"Step 5 structured documents not found: {STRUCTURED_DOCS}")
    if OUT.exists():
        shutil.rmtree(OUT)
    DOC_OUTPUT.mkdir(parents=True)
    DOCLING_OUTPUT.mkdir(parents=True)
    f = load_frameworks()
    converter, parser = build_converter(f), make_hierarchy_parser(f)
    all_chunks, all_nodes, manifest = [], [], []
    for metadata in load_document_metadata():
        chunks, nodes, source_character_count = process_document(metadata, converter, parser, f)
        document_id = metadata["document_id"]
        (DOC_OUTPUT / f"{document_id}.chunks.json").write_text(
            json.dumps({"document_id": document_id, "chunks": chunks}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        all_chunks.extend(chunks)
        all_nodes.extend({"document_id": document_id, **node} for node in nodes)
        manifest.append({
            "document_id": document_id, "title": metadata["title"],
            "chunk_count": len(chunks), "hierarchy_node_count": len(nodes),
            "source_character_count": source_character_count,
            "chunk_file": f"documents/{document_id}.chunks.json",
            "docling_json": f"docling_documents/{document_id}.docling.json",
            "docling_markdown": f"docling_documents/{document_id}.md",
        })
    (OUT / "chunks.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in all_chunks), encoding="utf-8"
    )
    (OUT / "node_hierarchy.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in all_nodes), encoding="utf-8"
    )
    (OUT / "document_manifest.json").write_text(
        json.dumps({"documents": manifest}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    checks = {
        "documents_processed": len(manifest) == 10,
        "chunks_nonempty": all(item["content"].strip() for item in all_chunks),
        "chunk_ids_unique": len({item["metadata"]["chunk_id"] for item in all_chunks}) == len(all_chunks),
        "all_chunks_have_parent": all(item["metadata"]["parent_node_id"] for item in all_chunks),
        "all_chunks_have_heading_path": all(item["metadata"]["heading_path"] for item in all_chunks),
    }
    validation = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "pipeline": "Docling -> LlamaIndex HierarchicalNodeParser -> SentenceSplitter",
        "parameters": {"parent_tokens": MAX_TOKENS, "leaf_target_tokens": TARGET_TOKENS,
                       "minimum_tokens_advisory": MIN_TOKENS, "overlap_tokens": OVERLAP_TOKENS},
        "documents_processed": len(manifest), "chunks_generated": len(all_chunks),
        "hierarchy_nodes_generated": len(all_nodes),
        "legal_dependency_start_warnings": sum(
            item["metadata"]["starts_with_legal_dependency"] for item in all_chunks
        ),
        "checks": checks, "status": "passed" if all(checks.values()) else "review_required",
    }
    (OUT / "validation_report.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "README.md").write_text(
        "# RBI Step-6 Chunks\n\n`chunks.jsonl` contains leaf nodes for BM25/vector indexing. "
        "`node_hierarchy.jsonl` preserves all LlamaIndex nodes for parent-child retrieval. "
        "`docling_documents/` stores lossless Docling JSON and page-marked Markdown.\n",
        encoding="utf-8",
    )
    shutil.copy2(Path(__file__), OUT / "chunk_regulatory_documents.py")
    print(json.dumps(validation, indent=2))
    print_chunking_summary(manifest, all_chunks)


if __name__ == "__main__":
    main()
