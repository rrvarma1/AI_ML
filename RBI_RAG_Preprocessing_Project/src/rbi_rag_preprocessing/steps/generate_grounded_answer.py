"""Step 13: generate a citation-validated answer with local Ollama Gemma 3 4B."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CONTEXT_ROOT = Path(os.environ["RBI_CONTEXT_OUTPUT_DIR"]).expanduser().resolve()
EXPANDED_CONTEXT_FILE = CONTEXT_ROOT / "expanded_context.json"
OUT = Path(os.environ["RBI_ANSWER_OUTPUT_DIR"]).expanduser().resolve()
QUERY_OVERRIDE = os.getenv("RBI_ANSWER_QUERY", "").strip()
MODEL = os.getenv("RBI_LLM_MODEL", "gemma3:4b").strip()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").strip()
CONTEXT_TOKENS = int(os.getenv("RBI_LLM_CONTEXT_TOKENS", "32768"))
MAX_OUTPUT_TOKENS = int(os.getenv("RBI_LLM_MAX_OUTPUT_TOKENS", "1800"))
EVIDENCE_TOKEN_BUDGET = int(os.getenv("RBI_LLM_EVIDENCE_TOKEN_BUDGET", "22000"))
TEMPERATURE = float(os.getenv("RBI_LLM_TEMPERATURE", "0.1"))
SEED = int(os.getenv("RBI_LLM_SEED", "42"))
MAX_REPAIR_ATTEMPTS = 1

ALLOWED_STATUSES = {
    "answered", "partially_answered", "insufficient_evidence", "conflicting_evidence"
}
EVIDENCE_REFERENCE = re.compile(r"\[E(\d+)\]")
LOOSE_EVIDENCE_REFERENCE = re.compile(
    r"(?:\[|\()\s*(?:evidence\s*)?e\s*[-:#]?\s*(\d+)\s*(?:\]|\))",
    flags=re.IGNORECASE,
)
EVIDENCE_BLOCK = re.compile(r"(?m)^\[(E\d+)\]\s*$")
NUMBER_TOKEN = re.compile(r"(?<![A-Za-z])(?:₹\s*)?\d+(?:[.,]\d+)?%?")
WORD_TOKEN = re.compile(r"[A-Za-z][A-Za-z-]{2,}")
STOPWORDS = {
    "about", "after", "also", "and", "answer", "are", "been", "being", "between",
    "could", "does", "each", "evidence", "from", "have", "into", "must", "only",
    "other", "shall", "should", "that", "the", "their", "these", "this", "those",
    "under", "using", "were", "what", "when", "where", "which", "while", "with",
}
NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "hundred", "thousand",
    "lakh", "crore", "million", "billion",
}

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer_status": {"type": "string", "enum": sorted(ALLOWED_STATUSES)},
        "answer": {"type": "string"},
        "conditions_and_exceptions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "evidence_ids"],
                "additionalProperties": False,
            },
        },
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "evidence_ids"],
                "additionalProperties": False,
            },
        },
        "conflicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "evidence_ids"],
                "additionalProperties": False,
            },
        },
        "regulatory_status_note": {"type": "string"},
        "evidence_ids_used": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "answer_status", "answer", "conditions_and_exceptions", "claims",
        "conflicts", "regulatory_status_note", "evidence_ids_used",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are an RBI regulatory-document question-answering assistant.

Answer solely from the EVIDENCE supplied in the user message.

Mandatory rules:
1. Do not use general knowledge, memory, assumptions, or external sources.
2. Every material factual or regulatory claim must cite one or more supplied evidence IDs, such as [E1].
3. Distinguish binding or mandatory requirements from explanatory, historical, descriptive, or background text.
4. Preserve every applicability condition, proviso, exception, exclusion, threshold, date, amount, percentage, and entity-specific qualification.
5. Do not combine provisions applying to different regulated entities unless the evidence explicitly supports the comparison.
6. Do not treat amended, withdrawn, repealed, superseded, draft, or historical provisions as currently binding.
7. If current regulatory status is not established by the evidence, state that limitation.
8. If evidence conflicts, identify the conflict and cite all conflicting evidence.
9. If evidence is insufficient, return answer_status=insufficient_evidence and do not complete the answer from memory.
10. Interpret a proviso or exception together with its supplied governing rule and parent context.
11. Use only evidence IDs present in the user message. Never invent titles, sections, clauses, pages, dates, or citations.
12. Put inline evidence IDs in the answer and in every condition or exception text.
13. Return only JSON conforming to the supplied schema."""


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
                pass
    try:
        public = {
            key: item for key, item in vars(value).items()
            if not key.startswith("_") and not callable(item)
        }
        if public:
            return jsonable(public)
    except TypeError:
        pass
    return str(value)


def validate_parameters() -> None:
    if not MODEL:
        raise ValueError("Ollama model cannot be empty.")
    if CONTEXT_TOKENS < 4096:
        raise ValueError("LLM context must be at least 4096 tokens.")
    if MAX_OUTPUT_TOKENS <= 0 or EVIDENCE_TOKEN_BUDGET <= 0:
        raise ValueError("Output and evidence token budgets must be greater than zero.")
    if EVIDENCE_TOKEN_BUDGET + MAX_OUTPUT_TOKENS >= CONTEXT_TOKENS:
        raise ValueError(
            "Evidence plus output budgets must leave room in the configured context for prompts."
        )
    if not 0 <= TEMPERATURE <= 2:
        raise ValueError("LLM temperature must be between 0 and 2.")


def load_expanded_context(path: Path) -> tuple[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            "Run --step expand first; RBI_Expanded_Context/expanded_context.json is missing."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    query = str(payload.get("query") or "").strip()
    evidence = payload.get("evidence")
    if not query or not isinstance(evidence, list) or not evidence:
        raise ValueError(f"Invalid Step-12 evidence package: {path}")
    return query, payload


def token_estimate(text: str) -> int:
    # Conservative approximation suitable for budgeting without loading a second tokenizer.
    return max(1, int(len(text.split()) * 1.35))


def metadata_value(metadata: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = metadata.get(name)
        if value not in (None, "", []):
            return value
    return None


def evidence_header(evidence_id: str, item: dict[str, Any]) -> str:
    metadata = item.get("metadata") or {}
    heading = metadata_value(metadata, "heading_path_text", "heading_path") or "Unknown"
    if isinstance(heading, list):
        heading = " > ".join(str(part) for part in heading)
    return "\n".join([
        f"[{evidence_id}]",
        f"Chunk ID: {item.get('chunk_id')}",
        f"Document title: {metadata_value(metadata, 'title') or 'Unknown'}",
        f"Document type: {metadata_value(metadata, 'document_type', 'doc_type') or 'Unknown'}",
        f"Regulatory status: {metadata_value(metadata, 'regulatory_status', 'status') or 'Unknown'}",
        f"Publication date: {metadata_value(metadata, 'publication_date') or 'Unknown'}",
        f"Updated through: {metadata_value(metadata, 'updated_through') or 'Unknown'}",
        f"Regulatory path: {heading}",
        f"Pages: {metadata.get('page_start', 'Unknown')}-{metadata.get('page_end', 'Unknown')}",
    ])


def build_evidence_package(
    evidence: list[dict[str, Any]], budget: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    selected, omitted, blocks = [], [], []
    used = 0
    for position, item in enumerate(evidence, start=1):
        evidence_id = f"E{position}"
        content = str(item.get("expanded_content") or "").strip()
        if not content:
            raise ValueError(f"Expanded evidence {item.get('chunk_id')} has no expanded_content.")
        block = f"{evidence_header(evidence_id, item)}\n\n{content}"
        estimated = token_estimate(block)
        record = {
            "evidence_id": evidence_id,
            "chunk_id": item.get("chunk_id"),
            "rank": item.get("rank"),
            "reranker_score": item.get("reranker_score"),
            "metadata": item.get("metadata") or {},
            "expansion": item.get("expansion") or {},
            "expanded_content": content,
            "token_estimate": estimated,
        }
        if selected and used + estimated > budget:
            omitted.append(record)
            continue
        selected.append(record)
        blocks.append(block)
        used += estimated
    return selected, omitted, "\n\n".join(blocks)


def build_user_prompt(query: str, evidence_text: str) -> str:
    return f"""QUESTION

{query}

EVIDENCE

{evidence_text}

TASK

Answer the question using only the evidence above. Preserve legal qualifications and entity applicability. Cite every material claim using the supplied [E#] identifiers. Return only the required structured JSON."""


def response_content(response: Any) -> str:
    message = model_field(response, "message")
    content = model_field(message, "content") if message is not None else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Ollama returned an empty response message.")
    return content.strip()


def referenced_ids(value: Any) -> set[str]:
    if isinstance(value, str):
        return {f"E{match}" for match in EVIDENCE_REFERENCE.findall(value)}
    if isinstance(value, dict):
        result: set[str] = set()
        for item in value.values():
            result.update(referenced_ids(item))
        return result
    if isinstance(value, list):
        result: set[str] = set()
        for item in value:
            result.update(referenced_ids(item))
        return result
    return set()


def validate_answer(answer: dict[str, Any], valid_ids: set[str]) -> list[str]:
    errors = []
    required = set(ANSWER_SCHEMA["required"])
    missing = sorted(required - set(answer))
    if missing:
        errors.append(f"Missing required fields: {', '.join(missing)}")
        return errors
    if answer.get("answer_status") not in ALLOWED_STATUSES:
        errors.append(f"Invalid answer_status: {answer.get('answer_status')!r}")
    if not isinstance(answer.get("answer"), str) or not answer["answer"].strip():
        errors.append("answer must be a non-empty string")
    for field in ("conditions_and_exceptions", "claims", "conflicts"):
        entries = answer.get(field)
        if not isinstance(entries, list):
            errors.append(f"{field} must be an array")
            continue
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict) or not str(entry.get("text") or "").strip():
                errors.append(f"{field}[{index}] must contain non-empty text")
                continue
            ids = entry.get("evidence_ids")
            if not isinstance(ids, list) or not ids:
                errors.append(f"{field}[{index}] must cite at least one evidence ID")
            elif any(item not in valid_ids for item in ids):
                errors.append(f"{field}[{index}] contains an unknown evidence ID")
    used = answer.get("evidence_ids_used")
    if not isinstance(used, list):
        errors.append("evidence_ids_used must be an array")
        used_set: set[str] = set()
    else:
        used_set = set(used)
        unknown = sorted(used_set - valid_ids)
        if unknown:
            errors.append(f"Unknown evidence_ids_used: {', '.join(unknown)}")
    inline = referenced_ids(answer)
    unknown_inline = sorted(inline - valid_ids)
    if unknown_inline:
        errors.append(f"Invented inline evidence IDs: {', '.join(unknown_inline)}")
    if not inline.issubset(used_set):
        errors.append("evidence_ids_used does not include every cited evidence ID")
    status = answer.get("answer_status")
    if status in {"answered", "partially_answered", "conflicting_evidence"}:
        if not answer.get("claims"):
            errors.append("A substantive answer must include cited claims")
        if not EVIDENCE_REFERENCE.search(answer.get("answer", "")):
            errors.append("A substantive answer must contain inline evidence citations")
    return errors


def parse_json_object(raw: str) -> dict[str, Any]:
    """Accept plain JSON, fenced JSON, or prose surrounding one JSON object."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("Top-level response is not an object")
    return value


def evidence_blocks(evidence_text: str, valid_ids: set[str]) -> dict[str, str]:
    matches = list(EVIDENCE_BLOCK.finditer(evidence_text))
    blocks = {}
    for index, match in enumerate(matches):
        evidence_id = match.group(1)
        if evidence_id in valid_ids:
            end = matches[index + 1].start() if index + 1 < len(matches) else len(evidence_text)
            blocks[evidence_id] = evidence_text[match.end():end].strip()
    return blocks


def grounding_tokens(text: str) -> set[str]:
    return {
        token.casefold() for token in WORD_TOKEN.findall(EVIDENCE_REFERENCE.sub("", text))
        if token.casefold() not in STOPWORDS
    }


def canonicalize_inline_citations(text: str, valid_ids: set[str]) -> str:
    def replace(match: re.Match[str]) -> str:
        evidence_id = f"E{int(match.group(1))}"
        return f"[{evidence_id}]" if evidence_id in valid_ids else match.group(0)
    return LOOSE_EVIDENCE_REFERENCE.sub(replace, text)


def numbers_in(text: str) -> set[str]:
    values = {item.replace(" ", "").casefold() for item in NUMBER_TOKEN.findall(text)}
    values.update(grounding_tokens(text) & NUMBER_WORDS)
    return values


def answer_supported_by_grounded_claims(
    answer: str, claims: list[dict[str, Any]]
) -> list[str]:
    """Allow a summary to inherit only citations from substantively overlapping grounded claims."""
    answer_tokens = grounding_tokens(answer)
    answer_numbers = numbers_in(answer)
    supported = []
    for claim in claims:
        claim_text = str(claim.get("text") or "")
        overlap = answer_tokens & grounding_tokens(claim_text)
        numbers_supported = answer_numbers.issubset(numbers_in(claim_text))
        if numbers_supported and len(overlap) >= 2:
            for evidence_id in claim.get("evidence_ids") or []:
                if evidence_id not in supported:
                    supported.append(evidence_id)
    return supported


def supporting_evidence_ids(
    claim: str, blocks: dict[str, str], minimum_overlap: float = 0.30
) -> list[str]:
    """Conservatively map a claim only when its terms and every numeric value occur in evidence."""
    claim_tokens = grounding_tokens(claim)
    claim_numbers = numbers_in(claim)
    ranked = []
    for evidence_id, content in blocks.items():
        evidence_tokens = grounding_tokens(content)
        evidence_numbers = numbers_in(content)
        overlap_count = len(claim_tokens & evidence_tokens)
        denominator = max(1, min(len(claim_tokens), 12))
        overlap = overlap_count / denominator
        numbers_supported = claim_numbers.issubset(evidence_numbers)
        if numbers_supported and overlap_count >= 2 and overlap >= minimum_overlap:
            ranked.append((overlap, overlap_count, evidence_id))
    ranked.sort(reverse=True)
    return [item[2] for item in ranked[:2]]


def normalize_entry_list(
    value: Any, blocks: dict[str, str], actions: list[str]
) -> list[dict[str, Any]]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        value = [value]
        actions.append("coerced_non_list_entry_collection")
    output = []
    for item in value:
        if isinstance(item, str):
            item = {"text": item, "evidence_ids": []}
            actions.append("coerced_string_entry")
        if not isinstance(item, dict) or not str(item.get("text") or "").strip():
            continue
        text = str(item["text"]).strip()
        ids = supporting_evidence_ids(text, blocks)
        if ids:
            output.append({"text": text, "evidence_ids": ids})
    return output


def normalize_and_ground_answer(
    parsed: dict[str, Any], evidence_text: str, valid_ids: set[str]
) -> tuple[dict[str, Any], list[str]]:
    """Repair harmless Gemma formatting defects without accepting unsupported claims."""
    actions: list[str] = []
    blocks = evidence_blocks(evidence_text, valid_ids)
    answer_text = canonicalize_inline_citations(
        str(parsed.get("answer") or parsed.get("response") or "").strip(), valid_ids
    )
    model_inline_ids = [
        item for item in referenced_ids(answer_text) if item in valid_ids
    ]
    answer_without_citations = EVIDENCE_REFERENCE.sub("", answer_text).strip()
    status = str(parsed.get("answer_status") or parsed.get("status") or "answered").strip().casefold()
    aliases = {"complete": "answered", "success": "answered", "partial": "partially_answered",
               "insufficient": "insufficient_evidence", "conflicting": "conflicting_evidence"}
    status = aliases.get(status, status)
    if status not in ALLOWED_STATUSES:
        status = "answered" if answer_text else "insufficient_evidence"
        actions.append("normalized_answer_status")

    abstaining = status == "insufficient_evidence"
    claims = normalize_entry_list(parsed.get("claims"), blocks, actions)
    conditions = normalize_entry_list(
        parsed.get("conditions_and_exceptions", parsed.get("conditions")), blocks, actions
    )
    conflicts = normalize_entry_list(parsed.get("conflicts"), blocks, actions)
    answer_ids = (
        supporting_evidence_ids(answer_without_citations, blocks)
        if answer_without_citations and not abstaining else []
    )
    if not abstaining and not claims and answer_ids:
        claims = [{"text": EVIDENCE_REFERENCE.sub("", answer_text).strip(), "evidence_ids": answer_ids}]
        actions.append("created_grounded_claim_from_answer")
    if not abstaining and not answer_ids and claims:
        answer_ids = answer_supported_by_grounded_claims(answer_without_citations, claims)
        if answer_ids:
            actions.append("inherited_verified_claim_citations_for_answer")
    if not abstaining and answer_ids:
        answer_text = answer_without_citations.rstrip() + " " + "".join(
            f"[{item}]" for item in answer_ids
        )
        if model_inline_ids != answer_ids:
            actions.append("replaced_model_citations_with_verified_citations")
        else:
            actions.append("canonicalized_verified_inline_citations")

    used = []
    for evidence_id in answer_ids:
        if evidence_id not in used:
            used.append(evidence_id)
    for entry in claims + conditions + conflicts:
        for evidence_id in entry["evidence_ids"]:
            if evidence_id not in used:
                used.append(evidence_id)
    result = {
        "answer_status": status,
        "answer": answer_text or "The supplied evidence is insufficient to answer the question.",
        "conditions_and_exceptions": conditions,
        "claims": claims,
        "conflicts": conflicts,
        "regulatory_status_note": str(parsed.get("regulatory_status_note") or "").strip(),
        "evidence_ids_used": used,
    }
    limitation_language = re.compile(
        r"\b(?:insufficient|not established|cannot determine|could not determine|"
        r"does not specify|not supported|unclear|unknown)\b",
        flags=re.IGNORECASE,
    )
    if (
        result["answer_status"] == "partially_answered"
        and result["claims"] and not result["conflicts"]
        and not limitation_language.search(result["answer"])
        and not limitation_language.search(result["regulatory_status_note"])
    ):
        result["answer_status"] = "answered"
        actions.append("reconciled_fully_grounded_status_to_answered")
    return result, actions


def rebuild_answer_from_grounded_claims(answer: dict[str, Any]) -> dict[str, Any]:
    """Replace an ungroundable summary with only the already grounded claim text."""
    rendered = []
    used = []
    for claim in answer.get("claims") or []:
        text = EVIDENCE_REFERENCE.sub("", str(claim.get("text") or "")).strip()
        evidence_ids = [
            item for item in claim.get("evidence_ids") or []
            if re.fullmatch(r"E\d+", str(item))
        ]
        if not text or not evidence_ids:
            continue
        citations = "".join(f"[{item}]" for item in evidence_ids)
        rendered.append(f"{text.rstrip()} {citations}")
        for evidence_id in evidence_ids:
            if evidence_id not in used:
                used.append(evidence_id)
    if not rendered:
        return answer
    repaired = dict(answer)
    repaired["answer"] = " ".join(rendered)
    repaired["evidence_ids_used"] = list(dict.fromkeys(
        list(answer.get("evidence_ids_used") or []) + used
    ))
    return repaired


def call_ollama(client, messages: list[dict[str, str]]) -> tuple[Any, str]:
    response = client.chat(
        model=MODEL,
        messages=messages,
        format=ANSWER_SCHEMA,
        options={
            "temperature": TEMPERATURE,
            "top_p": 0.9,
            "num_ctx": CONTEXT_TOKENS,
            "num_predict": MAX_OUTPUT_TOKENS,
            "seed": SEED,
        },
    )
    return response, response_content(response)


def generate_validated_answer(
    client, query: str, evidence_text: str, valid_ids: set[str]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    user_prompt = build_user_prompt(query, evidence_text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    attempts = []
    last_errors: list[str] = []
    for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
        response, raw = call_ollama(client, messages)
        try:
            original = parse_json_object(raw)
            original_errors = validate_answer(original, valid_ids)
            if original_errors:
                parsed, repair_actions = normalize_and_ground_answer(
                    original, evidence_text, valid_ids
                )
                errors = validate_answer(parsed, valid_ids)
                if errors == ["A substantive answer must contain inline evidence citations"]:
                    rebuilt = rebuild_answer_from_grounded_claims(parsed)
                    rebuilt_errors = validate_answer(rebuilt, valid_ids)
                    if not rebuilt_errors:
                        parsed = rebuilt
                        errors = []
                        repair_actions.append("rebuilt_answer_from_verified_grounded_claims")
            else:
                parsed, repair_actions, errors = original, [], []
        except (json.JSONDecodeError, ValueError) as exc:
            parsed = None
            repair_actions = []
            errors = [f"Invalid JSON response: {exc}"]
        attempts.append({
            "attempt": attempt + 1,
            "raw_content": raw,
            "parsed": parsed,
            "repair_actions": repair_actions,
            "validation_errors": errors,
            "response_metadata": jsonable(response),
        })
        if parsed is not None and not errors:
            return parsed, attempts, []
        last_errors = errors
        if attempt < MAX_REPAIR_ATTEMPTS:
            messages.extend([
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "Your response failed validation:\n- " + "\n- ".join(errors) +
                        "\nReturn a corrected JSON object. Use only the original evidence IDs and evidence."
                    ),
                },
            ])
    failure = {
        "answer_status": "generation_error",
        "answer": "A citation-valid grounded answer could not be generated from the selected evidence.",
        "conditions_and_exceptions": [], "claims": [], "conflicts": [],
        "regulatory_status_note": "No regulatory conclusion was generated.",
        "evidence_ids_used": [],
    }
    return failure, attempts, last_errors


def citation_for(record: dict[str, Any]) -> str:
    evidence_id = record["evidence_id"]
    metadata = record["metadata"]
    title = str(metadata_value(metadata, "title") or "Untitled RBI document")
    heading = metadata_value(metadata, "heading_path", "heading_path_text")
    if isinstance(heading, list):
        cleaned = [str(item) for item in heading if str(item) != title]
        heading_text = " › ".join(cleaned)
    else:
        heading_text = str(heading or "Section not identified")
    page_start, page_end = metadata.get("page_start"), metadata.get("page_end")
    if page_start and page_end:
        pages = f"page {page_start}" if page_start == page_end else f"pages {page_start}–{page_end}"
    else:
        pages = "pages not identified"
    chunk_id = record.get("chunk_id") or "chunk not identified"
    status = metadata_value(metadata, "regulatory_status", "status")
    status_text = f" — status: {status}" if status else ""
    return f"[{evidence_id}] {title} — {heading_text} — {pages} — {chunk_id}{status_text}"


def render_markdown(answer: dict[str, Any], evidence_by_id: dict[str, dict[str, Any]]) -> str:
    lines = ["# Answer", "", answer["answer"].strip(), "", "## Conditions and exceptions", ""]
    conditions = answer.get("conditions_and_exceptions") or []
    if conditions:
        for condition in conditions:
            text = condition["text"].strip()
            suffix = "".join(f"[{item}]" for item in condition.get("evidence_ids", []) if f"[{item}]" not in text)
            lines.append(f"- {text}{suffix}")
    else:
        lines.append("No additional conditions or exceptions were established by the supplied evidence.")
    conflicts = answer.get("conflicts") or []
    if conflicts:
        lines.extend(["", "## Conflicting evidence", ""])
        for conflict in conflicts:
            text = conflict["text"].strip()
            suffix = "".join(f"[{item}]" for item in conflict.get("evidence_ids", []) if f"[{item}]" not in text)
            lines.append(f"- {text}{suffix}")
    if answer.get("regulatory_status_note"):
        lines.extend(["", "## Regulatory status", "", answer["regulatory_status_note"].strip()])
    lines.extend(["", "## Sources", ""])
    used = [item for item in answer.get("evidence_ids_used", []) if item in evidence_by_id]
    if used:
        lines.extend(f"- {citation_for(evidence_by_id[item])}" for item in used)
    else:
        lines.append("- No source was cited because no grounded regulatory conclusion was generated.")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    validate_parameters()
    stored_query, payload = load_expanded_context(EXPANDED_CONTEXT_FILE)
    if QUERY_OVERRIDE and QUERY_OVERRIDE != stored_query:
        raise ValueError(
            "--query does not match the latest Step-12 query. Run --step expand for this question first."
        )
    query = QUERY_OVERRIDE or stored_query
    selected, omitted, evidence_text = build_evidence_package(
        payload["evidence"], EVIDENCE_TOKEN_BUDGET
    )
    evidence_by_id = {record["evidence_id"]: record for record in selected}

    try:
        from ollama import Client
    except ImportError as exc:
        raise RuntimeError("Install the Ollama Python client with `python -m pip install -e .`.") from exc
    client = Client(host=OLLAMA_HOST)
    try:
        client.show(MODEL)
    except Exception as exc:
        raise RuntimeError(
            f"Ollama model {MODEL!r} is unavailable at {OLLAMA_HOST}. "
            f"Start Ollama and run `ollama pull {MODEL}`. Original error: {exc}"
        ) from exc

    started = time.perf_counter()
    answer, attempts, final_errors = generate_validated_answer(
        client, query, evidence_text, set(evidence_by_id)
    )
    duration = round(time.perf_counter() - started, 3)
    markdown = render_markdown(answer, evidence_by_id)

    OUT.mkdir(parents=True, exist_ok=True)
    query_id = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    query_dir = OUT / "queries" / query_id
    query_dir.mkdir(parents=True, exist_ok=True)
    invalid_dir = OUT / "invalid_responses"
    invalid_dir.mkdir(exist_ok=True)

    audit = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "query_id": query_id, "query": query, "provider": "Ollama", "model": MODEL,
        "ollama_host": OLLAMA_HOST,
        "parameters": {
            "context_tokens": CONTEXT_TOKENS, "max_output_tokens": MAX_OUTPUT_TOKENS,
            "evidence_token_budget": EVIDENCE_TOKEN_BUDGET,
            "temperature": TEMPERATURE, "top_p": 0.9, "seed": SEED,
        },
        "selected_evidence_count": len(selected), "omitted_evidence_count": len(omitted),
        "selected_evidence_ids": list(evidence_by_id),
        "omitted_chunk_ids": [record["chunk_id"] for record in omitted],
        "generation_duration_seconds": duration,
        "validation_passed": not final_errors and answer["answer_status"] != "generation_error",
        "final_validation_errors": final_errors,
        "attempt_count": len(attempts), "answer": answer,
    }
    (OUT / "latest_answer.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "latest_answer.md").write_text(markdown, encoding="utf-8")
    (OUT / "answer_generation_report.json").write_text(
        json.dumps({key: value for key, value in audit.items() if key != "answer"},
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (query_dir / "prompt.txt").write_text(
        f"SYSTEM\n\n{SYSTEM_PROMPT}\n\nUSER\n\n{build_user_prompt(query, evidence_text)}",
        encoding="utf-8",
    )
    (query_dir / "evidence.json").write_text(
        json.dumps({"selected": selected, "omitted": omitted}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (query_dir / "model_response.json").write_text(
        json.dumps({"attempts": attempts, "final_answer": answer}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (query_dir / "answer.md").write_text(markdown, encoding="utf-8")
    for attempt in attempts:
        if attempt["validation_errors"]:
            (invalid_dir / f"{query_id}_attempt_{attempt['attempt']}.json").write_text(
                json.dumps(attempt, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    print("-" * 100)
    print(f"Question:                 {query}")
    print(f"Local model:              {MODEL}")
    print(f"Evidence supplied:        {len(selected)}")
    print(f"Evidence omitted:         {len(omitted)}")
    print(f"Generation attempts:      {len(attempts)}")
    print(f"Validation:               {'PASSED' if audit['validation_passed'] else 'FAILED'}")
    print(f"Generation time:          {duration:.3f}s")
    print(f"Answer status:            {answer['answer_status']}")
    print(f"Saved query ID:           {query_id}")
    print("-" * 100)
    print(markdown)


if __name__ == "__main__":
    main()
