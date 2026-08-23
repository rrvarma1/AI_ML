"""Step 16: Streamlit question-and-answer interface for the RBI Hybrid RAG pipeline."""
from __future__ import annotations

import json
import html
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


STAGES = (
    ("hybrid", "Retrieve semantic and lexical candidates"),
    ("rerank", "Rerank the fused candidate pool"),
    ("expand", "Expand regulatory parent context"),
    ("answer", "Generate a grounded answer with Gemma"),
    ("cite", "Generate deterministic citations"),
)

VISIBLE_PROGRESS_STAGES = (
    ("hybrid", "Retrieve semantic and lexical candidates"),
    ("rerank", "Rerank the fused candidate pool"),
    ("expand", "Expand regulatory parent context"),
    ("answer", "Generate a grounded answer with Gemma"),
)


def default_project_root() -> Path:
    configured = os.getenv("RBI_PROJECT_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    # Installed/editable package: <software>/src/rbi_rag_preprocessing/this_file.py
    return Path(__file__).resolve().parents[3]


def build_stage_command(
    *, stage: str, query: str, corpus_dir: Path, inventory_file: Path,
    project_root: Path, namespace: str, rerank_model: str, llm_model: str,
) -> list[str]:
    return [
        sys.executable, "-m", "rbi_rag_preprocessing",
        "--corpus-dir", str(corpus_dir),
        "--inventory-file", str(inventory_file),
        "--project-root", str(project_root),
        "--step", stage,
        "--query", query,
        "--namespace", namespace,
        "--rerank-model", rerank_model,
        "--llm-model", llm_model,
    ]


def run_stage(command: list[str], timeout_seconds: int = 1800) -> dict[str, Any]:
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout_seconds,
        env=os.environ.copy(), check=False,
    )
    def redact(text: str) -> str:
        for setting in ("PINECONE_API_KEY", "OPENAI_API_KEY"):
            secret = os.getenv(setting, "")
            if secret:
                text = text.replace(secret, f"<{setting} redacted>")
        return text

    return {
        "stage": command[command.index("--step") + 1],
        "returncode": result.returncode,
        "stdout": redact(result.stdout),
        "stderr": redact(result.stderr),
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Expected pipeline output is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_results(project_root: Path) -> dict[str, Any]:
    answer = load_json(project_root / "RBI_Grounded_Answers" / "latest_answer.json")
    query_id = str(answer.get("query_id") or "")
    return {
        "hybrid": load_json(project_root / "RBI_Hybrid_Retrieval" / "hybrid_candidates.json"),
        "rerank": load_json(project_root / "RBI_Reranked_Retrieval" / "reranked_candidates.json"),
        "expanded": load_json(project_root / "RBI_Expanded_Context" / "expanded_context.json"),
        "answer": answer,
        "evidence": load_json(
            project_root / "RBI_Grounded_Answers" / "queries" / query_id / "evidence.json"
        ),
        "citations": load_json(project_root / "RBI_Citations" / "citation_manifest.json"),
    }


def metadata_value(metadata: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = metadata.get(name)
        if value not in (None, "", []):
            return value
    return None


def heading_label(metadata: dict[str, Any]) -> str:
    value = metadata_value(metadata, "heading_path", "full_heading_path", "heading_path_text")
    if isinstance(value, list):
        return " → ".join(str(part) for part in value if str(part).strip()) or "Heading unavailable"
    return str(value).strip() if value else "Heading unavailable"


def page_label(metadata: dict[str, Any]) -> str:
    start, end = metadata.get("page_start"), metadata.get("page_end")
    if start in (None, ""):
        return "Pages unavailable"
    return f"Page {start}" if end in (None, "") or str(start) == str(end) else f"Pages {start}–{end}"


def build_score_lookup(
    hybrid_payload: dict[str, Any], rerank_payload: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}
    for candidate in hybrid_payload.get("candidates") or []:
        chunk_id = str(candidate.get("chunk_id") or "")
        scores[chunk_id] = {
            "hybrid_rank": candidate.get("rank"),
            "rrf_score": candidate.get("rrf_score"),
            "matched_by": candidate.get("matched_by") or [],
            "semantic": candidate.get("semantic"),
            "lexical": candidate.get("lexical"),
        }
    for candidate in rerank_payload.get("candidates") or []:
        chunk_id = str(candidate.get("chunk_id") or "")
        record = scores.setdefault(chunk_id, {})
        record.update({
            "reranker_rank": candidate.get("reranker_rank", candidate.get("rank")),
            "reranker_score": candidate.get("reranker_score"),
        })
    return scores


def debug_rows(payload: dict[str, Any], source: str) -> list[dict[str, Any]]:
    rows = []
    for candidate in payload.get("candidates") or []:
        metadata = candidate.get("metadata") or {}
        base = {
            "chunk_id": candidate.get("chunk_id"),
            "title": metadata_value(metadata, "title", "document_title") or "Untitled",
        }
        if source in ("semantic", "lexical"):
            score = candidate.get(source)
            if not score:
                continue
            rows.append({
                "rank": score.get("rank"), **base,
                "raw_score": score.get("raw_score"),
                "normalized_score": score.get("normalized_score"),
                "hybrid_rank": candidate.get("rank"),
            })
        elif source == "rrf":
            rows.append({
                "rank": candidate.get("rank"), **base,
                "rrf_score": candidate.get("rrf_score"),
                "matched_by": ", ".join(candidate.get("matched_by") or []),
            })
        else:
            rows.append({
                "rank": candidate.get("reranker_rank", candidate.get("rank")), **base,
                "reranker_score": candidate.get("reranker_score"),
                "hybrid_rank": candidate.get("hybrid_rank"),
            })
    return rows


def render_citation(st, citation: dict[str, Any]) -> None:
    title = citation.get("document_title") or "Untitled RBI document"
    heading = citation.get("full_heading_path_text") or "Heading unavailable"
    section = citation.get("section_or_clause") or "Section unavailable"
    pages = citation.get("page_range") or "Pages unavailable"
    chunk_id = citation.get("chunk_id") or "Chunk unavailable"
    url = citation.get("source_url")
    source = f"[Open RBI source]({url})" if url else "Source URL unavailable"
    st.markdown(f"**{title}**  \n{heading}  \n{section} · {pages} · `{chunk_id}` · {source}")


def build_download_markdown(results: dict[str, Any]) -> str:
    audit = results["answer"]
    answer = audit.get("answer") or {}
    lines = [
        "# RBI Regulation Assistant", "", "## Question", "",
        str(audit.get("query") or ""), "", "## Answer", "",
        str(answer.get("answer") or "No answer was generated."), "",
        "## Conditions and exceptions", "",
    ]
    conditions = answer.get("conditions_and_exceptions") or []
    if conditions:
        lines.extend(f"- {item.get('text', '')}" for item in conditions)
    else:
        lines.append("No additional conditions or exceptions were established by the evidence.")
    conflicts = answer.get("conflicts") or []
    if conflicts:
        lines.extend(["", "## Conflicting evidence", ""])
        lines.extend(f"- {item.get('text', '')}" for item in conflicts)
    lines.extend(["", "## Sources", ""])
    citations = results["citations"].get("unique_citations_used") or []
    if not citations:
        lines.append("- No citation was emitted.")
    for citation in citations:
        url = citation.get("source_url")
        source = f"[RBI source]({url})" if url else "Source URL unavailable"
        lines.append(
            f"- {citation.get('document_title') or 'Untitled RBI document'} — "
            f"{citation.get('full_heading_path_text') or 'Heading unavailable'} — "
            f"{citation.get('section_or_clause') or 'Section unavailable'} — "
            f"{citation.get('page_range') or 'Pages unavailable'} — "
            f"chunk `{citation.get('chunk_id') or 'unavailable'}` — {source}"
        )
    lines.extend([
        "", "---", "",
        "Generated by RBI Regulation Assistant — Grounded Regulatory Q&A.",
        "Educational use only; not affiliated with or endorsed by the Reserve Bank of India.",
    ])
    return "\n".join(lines).strip() + "\n"


def history_title(question: str, limit: int = 54) -> str:
    cleaned = " ".join(question.split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "…"


def runtime_configuration() -> dict[str, Any]:
    project_root = default_project_root()
    return {
        "project_root": project_root,
        "corpus_dir": Path(
            os.getenv("RBI_CORPUS_DIR", str(project_root / "RBI_Document_Corpus"))
        ).expanduser(),
        "inventory_file": Path(
            os.getenv("RBI_INVENTORY_FILE", str(project_root / "RBI_Document_Inventory.xlsx"))
        ).expanduser(),
        "namespace": os.getenv("RBI_PINECONE_NAMESPACE", "RBI_RAG"),
        "rerank_model": os.getenv("RBI_RERANK_MODEL", "cohere-rerank-3.5"),
        "llm_model": os.getenv("RBI_LLM_MODEL", "gemma3:4b"),
    }


def bank_emblem_html(size: str = "normal") -> str:
    return (
        f'<div class="bank-emblem bank-emblem-{size}" aria-label="Regulatory assistant emblem">'
        '<svg viewBox="0 0 24 24" role="img" aria-hidden="true">'
        '<path d="M3 9h18M5 9V19M9 9V19M15 9V19M19 9V19M3 19h18M2 22h20M12 2 3 7h18L12 2Z" '
        'fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>'
        '</svg></div>'
    )


def pipeline_progress_html(
    active_stage: str | None, *, state: str = "running"
) -> str:
    """Build the fixed, viewport-visible progress strip used during a query."""
    stage_index = {stage: index for index, (stage, _) in enumerate(VISIBLE_PROGRESS_STAGES)}
    active_index = stage_index.get(active_stage, len(VISIBLE_PROGRESS_STAGES))
    if active_stage == "cite" or state == "complete":
        active_index = len(VISIBLE_PROGRESS_STAGES)
    title = {
        "running": "Running the grounded Q&amp;A pipeline…",
        "complete": "Grounded answer and citations are ready",
        "error": "The grounded Q&amp;A pipeline stopped",
    }.get(state, "Running the grounded Q&amp;A pipeline…")
    marker = "✓" if state == "complete" else "!" if state == "error" else "◌"
    steps = []
    for index, (_, label) in enumerate(VISIBLE_PROGRESS_STAGES):
        if state == "complete" or index < active_index:
            css_class, icon = "done", "✓"
        elif index == active_index and state != "error":
            css_class, icon = "active", "●"
        else:
            css_class, icon = "pending", "○"
        steps.append(
            f'<span class="pipeline-step {css_class}"><b>{icon}</b> {html.escape(label)}</span>'
        )
    return (
        f'<div class="pipeline-progress-strip pipeline-{state}" role="status" aria-live="polite">'
        f'<div class="pipeline-progress-title"><span class="pipeline-spinner">{marker}</span>'
        f'<strong>{title}</strong></div><div class="pipeline-progress-steps">'
        + "".join(steps) + "</div></div>"
    )


def clear_evidence_states(session_state: Any) -> None:
    """Close every open evidence inspector for a new chat or submitted query."""
    for key in list(session_state.keys()):
        if str(key).startswith("evidence_details_open_"):
            del session_state[key]


def render_evidence_details(
    st, results: dict[str, Any], logs: list[dict[str, Any]]
) -> None:
    """Render expanded evidence in a full-width panel below the answer actions."""
    answer = (results.get("answer") or {}).get("answer") or {}
    citations = (results.get("citations") or {}).get("unique_citations_used") or []
    with st.container(border=True, height=560):
        st.markdown('<div class="evidence-details-title">Evidence Details</div>', unsafe_allow_html=True)
        conditions = answer.get("conditions_and_exceptions") or []
        st.markdown("**Conditions and exceptions**")
        if conditions:
            for item in conditions:
                st.markdown(f"- {item.get('text', '')}")
        else:
            st.caption("No additional conditions or exceptions were established.")
        conflicts = answer.get("conflicts") or []
        if conflicts:
            st.markdown("**Conflicting evidence**")
            for item in conflicts:
                st.markdown(f"- {item.get('text', '')}")
        st.markdown("**Sources**")
        for citation in citations:
            render_citation(st, citation)
        scores = build_score_lookup(results["hybrid"], results["rerank"])
        for record in results["evidence"].get("selected") or []:
            metadata = record.get("metadata") or {}
            chunk_id = str(record.get("chunk_id") or "")
            score = scores.get(chunk_id, {})
            title = metadata_value(metadata, "title", "document_title") or "Untitled RBI document"
            with st.expander(f"{record.get('evidence_id', '')} · {title} · {page_label(metadata)}"):
                st.markdown(f"**Heading:** {heading_label(metadata)}")
                source_url = metadata_value(metadata, "source_url", "url", "document_url")
                if source_url:
                    st.markdown(f"[Open source document]({source_url})")
                matched = ", ".join(score.get("matched_by") or []) or "retrieval metadata unavailable"
                st.caption(
                    f"Chunk `{chunk_id}` · matched by {matched} · "
                    f"RRF {score.get('rrf_score', '—')} · reranker {score.get('reranker_score', '—')}"
                )
                st.text(str(record.get("expanded_content") or ""))
        with st.expander("Retrieval details", expanded=False):
            pinecone_tab, bm25_tab, rrf_tab, rerank_tab, logs_tab = st.tabs(
                ["Pinecone", "BM25", "RRF", "Reranker", "Logs"]
            )
            with pinecone_tab:
                st.dataframe(debug_rows(results["hybrid"], "semantic"), use_container_width=True)
            with bm25_tab:
                st.dataframe(debug_rows(results["hybrid"], "lexical"), use_container_width=True)
            with rrf_tab:
                st.dataframe(debug_rows(results["hybrid"], "rrf"), use_container_width=True)
            with rerank_tab:
                st.dataframe(debug_rows(results["rerank"], "rerank"), use_container_width=True)
            with logs_tab:
                for log in logs:
                    st.markdown(f"**{log['stage']}** — exit code {log['returncode']}")
                    st.code((log.get("stdout") or "") + (log.get("stderr") or ""), language="text")


def render_answer(st, results: dict[str, Any], logs: list[dict[str, Any]]) -> bool:
    audit = results["answer"]
    answer = audit.get("answer") or {}
    status = answer.get("answer_status", "unknown")
    if status == "generation_error":
        st.error("Answer generation failed validation — inspect the Step 13 diagnostics below.")
        for error in audit.get("final_validation_errors") or []:
            st.markdown(f"- {error}")
    elif status == "insufficient_evidence":
        st.error("Insufficient evidence — the retrieved RBI material does not support a reliable answer.")
    elif status == "conflicting_evidence":
        st.warning("Conflicting evidence — review the cited provisions before relying on this answer.")
    elif status == "partially_answered":
        st.warning("Partially answered — some aspects are not supported by the retrieved evidence.")
    st.markdown('<div class="assistant-label">RBI Regulation Assistant</div>', unsafe_allow_html=True)
    st.markdown(str(answer.get("answer") or "No answer was generated."))
    citations = results["citations"].get("unique_citations_used") or []
    if citations:
        first = citations[0]
        title = first.get("document_title") or "RBI source"
        heading = first.get("full_heading_path_text") or first.get("section_or_clause") or "Heading unavailable"
        pages = first.get("page_range") or "Pages unavailable"
        st.markdown(
            '<div class="compact-source">▧&nbsp;&nbsp;'
            f'{html.escape(str(title))} · {html.escape(str(heading))} · {html.escape(str(pages))}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("No citation was emitted because no grounded conclusion was generated.")

    query_id = str(audit.get("query_id") or "rbi-answer")
    evidence_column, download_column, spacer = st.columns([0.20, 0.30, 0.50])
    with evidence_column:
        evidence_state_key = f"evidence_details_open_{query_id}"
        if evidence_state_key not in st.session_state:
            st.session_state[evidence_state_key] = False
        if st.button("▤  Evidence", key=f"evidence_{query_id}", use_container_width=True):
            st.session_state[evidence_state_key] = True
    with download_column:
        st.download_button(
            "⤓  Download answer",
            data=build_download_markdown(results),
            file_name=f"rbi_regulation_qa_{query_id}.md",
            mime="text/markdown",
            key=f"download_{query_id}",
            use_container_width=True,
        )
    return bool(st.session_state[evidence_state_key])


def inject_executive_styles(st) -> None:
    st.markdown(
        """
        <style>
        :root { --rbi-blue:#7ebcf1; --rbi-main:#252525; --rbi-side:#333333; --rbi-muted:#9d9d9d; }
        #MainMenu, footer { visibility:hidden; }
        html, body, .stApp { background:#161616; color:#eeeeee; }
        .stApp { border:1px solid #454545; border-radius:24px; overflow:hidden; }
        [data-testid="stHeader"] { background:transparent; }
        [data-testid="stSidebar"] { background:#333333; border-right:1px solid #494949; }
        [data-testid="stSidebar"] * { color:#ededed; }
        [data-testid="stSidebarContent"] { padding:1.55rem 1.3rem; }
        [data-testid="stMainBlockContainer"] {
            max-width:1180px; margin-top:0; margin-bottom:0; padding:2.2rem 3rem 3rem;
            background:#252525; min-height:100vh; color:#eeeeee;
        }
        [data-testid="stMain"] { background:#252525; }
        [data-testid="stSidebarContent"] [data-testid="stVerticalBlock"] { min-height:calc(100vh - 3rem); }
        [data-testid="stSidebarContent"] [data-testid="stVerticalBlock"] > div:last-child { margin-top:auto; }
        [data-testid="stSidebar"] .stButton > button {
            width:100%; text-align:left; justify-content:flex-start; border:0;
            background:transparent; color:#a7a7a7; border-radius:15px; min-height:3rem;
        }
        [data-testid="stSidebar"] .stButton > button:hover { background:#414141; color:#ffffff; }
        [data-testid="stSidebar"] .stButton > button[kind="primary"] { background:#7ebcf1; color:#111111; }
        [data-testid="stSidebar"] .st-key-new_chat button { background:#f0f0f0; color:#111111 !important; font-weight:500; justify-content:center; }
        [data-testid="stSidebar"] .st-key-new_chat button *,
        [data-testid="stSidebar"] [class*="st-key-history_"] button * { color:#111111 !important; }
        [data-testid="stSidebar"] .st-key-new_chat button:hover { background:#ffffff; color:#111111 !important; }
        [data-testid="stSidebar"] [class*="st-key-history_"] button {
            background:#d7d7d7; color:#111111 !important; margin-bottom:.25rem;
        }
        [data-testid="stSidebar"] [class*="st-key-history_"] button:hover { background:#eeeeee; color:#111111 !important; }
        [data-testid="stSidebar"] [class*="st-key-history_"] button[kind="primary"] { background:#7ebcf1; color:#111111 !important; }
        .rbi-brand { padding:.4rem 0 1.15rem; }
        .rbi-brand-title { font-size:1.08rem; font-weight:500; line-height:1.25; color:#f0f0f0; margin-top:1rem; }
        .rbi-brand-subtitle { font-size:.86rem; color:#9d9d9d; margin-top:.18rem; }
        .bank-emblem { display:grid; place-items:center; border-radius:50%; background:#7ebcf1; color:#111111; }
        .bank-emblem svg { width:55%; height:55%; }
        .bank-emblem-normal { width:52px; height:52px; }
        .bank-emblem-small { width:46px; height:46px; }
        .history-label { margin:1.55rem 0 .45rem; font-size:.76rem; letter-spacing:.1em; color:#a6a6a6; }
        .sidebar-foot { padding-top:1rem; font-size:.74rem; color:#a5a5a5; }
        .executive-header { display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; border-bottom:1px solid #3d3d3d; padding:.25rem 0 1.5rem; margin-bottom:1.8rem; }
        .executive-header h1 { color:#eeeeee; font-size:2rem; margin:0 0 .12rem; font-weight:400; }
        .executive-header p { color:#9d9d9d; margin:0; font-size:1rem; }
        .question-row { display:grid; grid-template-columns:46px 1fr; gap:.9rem; align-items:start; margin:1.1rem 0 2rem; }
        .user-avatar { width:46px; height:46px; border-radius:50%; display:grid; place-items:center; background:#7ebcf1; color:#111111; font-size:1rem; }
        .conversation-label, .assistant-label { color:#9d9d9d; font-size:.88rem; margin-bottom:.25rem; }
        .question-text { color:#f0f0f0; font-size:1.05rem; }
        .assistant-grid [data-testid="stVerticalBlock"] { gap:.35rem; }
        .empty-state { margin:3rem auto 1rem; max-width:560px; text-align:center; padding:1.2rem; }
        .empty-state .bank-emblem { margin:0 auto 1rem; }
        .empty-state h3 { color:#eeeeee; margin-bottom:.4rem; }
        .empty-state p { color:#9d9d9d; }
        .compact-source { color:#9d9d9d; margin:1rem 0 .8rem; font-size:.92rem; }
        .evidence-details-title { color:#f1f1f1; font-size:1.15rem; font-weight:600; margin:0 0 1rem; }
        .pipeline-progress-strip {
            position:relative; margin:1.5rem 0 1rem;
            padding:.75rem 1rem; background:#303030; border:1px solid #505050;
            border-radius:14px; box-shadow:0 8px 22px rgba(0,0,0,.22); color:#eeeeee;
        }
        .pipeline-progress-title { display:flex; align-items:center; gap:.5rem; margin-bottom:.5rem; }
        .pipeline-spinner { color:#7ebcf1; font-size:1.05rem; }
        .pipeline-running .pipeline-spinner { animation:rbi-spin 1.15s linear infinite; }
        .pipeline-complete .pipeline-spinner { color:#7fc88a; }
        .pipeline-error .pipeline-spinner { color:#f2a1a1; }
        .pipeline-progress-steps { display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:.45rem; }
        .pipeline-step { color:#8f8f8f; font-size:.76rem; line-height:1.25; }
        .pipeline-step b { margin-right:.15rem; }
        .pipeline-step.done { color:#7fc88a; }
        .pipeline-step.active { color:#f1f1f1; }
        @keyframes rbi-spin { to { transform:rotate(360deg); } }
        .stMarkdown, .stMarkdown p { color:#d2d2d2; }
        div[data-testid="stStatusWidget"] { border-color:#4a4a4a; background:#303030; }
        .stDownloadButton > button, [data-testid="stPopover"] button { background:#333333; border-color:#3c3c3c; color:#f1f1f1; border-radius:12px; }
        [data-testid="stExpander"] { border:0; }
        [data-testid="stForm"] { border:0; border-top:1px solid #3d3d3d; border-radius:0; padding:2rem 0 0; margin-top:5rem; }
        [data-testid="stForm"] [data-testid="stHorizontalBlock"] { align-items:flex-end; }
        [data-testid="stTextArea"] textarea { background:#353535; border:1px solid #383838; color:#f1f1f1; border-radius:14px; min-height:110px; }
        [data-testid="stTextArea"] textarea::placeholder { color:#9d9d9d; }
        [data-testid="stFormSubmitButton"] button { background:#f2f2f2; color:#111111; border:0; border-radius:12px; min-height:3rem; }
        @media (max-width:700px) {
            [data-testid="stMainBlockContainer"] { padding:1.4rem 1rem 2rem; }
            .executive-header { display:block; }
            .executive-header h1 { font-size:1.55rem; }
            .pipeline-progress-steps { grid-template-columns:1fr 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    import streamlit as st
    from rbi_rag_preprocessing import __version__
    from dotenv import load_dotenv

    software_root = Path(__file__).resolve().parents[2]
    load_dotenv(software_root.parent / ".env")
    load_dotenv(software_root / ".env")
    st.set_page_config(
        page_title="RBI Regulation Assistant", page_icon="🏛️", layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_executive_styles(st)
    config = runtime_configuration()
    if "rbi_chat_history" not in st.session_state:
        st.session_state["rbi_chat_history"] = []
    if "rbi_active_chat" not in st.session_state:
        st.session_state["rbi_active_chat"] = None
    if "rbi_pending_question" not in st.session_state:
        st.session_state["rbi_pending_question"] = ""
    active_index = st.session_state["rbi_active_chat"]
    history = st.session_state["rbi_chat_history"]

    with st.sidebar:
        st.markdown(
            '<div class="rbi-brand">' + bank_emblem_html() +
            '<div class="rbi-brand-title">RBI Regulation Assistant</div>'
            '<div class="rbi-brand-subtitle">Grounded Regulatory Q&amp;A</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("✎  New chat", use_container_width=True, key="new_chat"):
            clear_evidence_states(st.session_state)
            st.session_state["rbi_active_chat"] = None
            st.session_state["rbi_pending_question"] = ""
            st.rerun()
        st.markdown('<div class="history-label">EARLIER QUESTIONS</div>', unsafe_allow_html=True)
        if history:
            for index in range(len(history) - 1, -1, -1):
                item = history[index]
                if st.button(
                    history_title(item["question"]), key=f"history_{index}",
                    use_container_width=True,
                    type="primary" if active_index == index else "secondary",
                ):
                    st.session_state["rbi_active_chat"] = index
                    st.rerun()
        else:
            st.caption("Your questions will appear here during this session.")
        st.markdown(
            f'<div class="sidebar-foot">Session history · v{__version__}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="executive-header"><div><h1>Grounded Regulatory Q&amp;A</h1>'
        '<p>Answers grounded in retrieved RBI evidence</p></div></div>',
        unsafe_allow_html=True,
    )
    evidence_display_slot = None
    if active_index is not None and 0 <= active_index < len(history):
        active = history[active_index]
        st.markdown(
            pipeline_progress_html("cite", state="complete"), unsafe_allow_html=True
        )
        conversation_pane, evidence_pane = st.columns([0.64, 0.36], gap="large")
        with conversation_pane:
            st.markdown(
                '<div class="question-row"><div class="user-avatar">RV</div><div>'
                '<div class="conversation-label">You</div>'
                f'<div class="question-text">{html.escape(active["question"])}</div></div></div>',
                unsafe_allow_html=True,
            )
            avatar_column, response_column = st.columns([0.09, 0.91])
            with avatar_column:
                st.markdown(bank_emblem_html("small"), unsafe_allow_html=True)
            with response_column:
                evidence_open = render_answer(st, active["results"], active["logs"])
        with evidence_pane:
            evidence_display_slot = st.empty()
            if evidence_open:
                with evidence_display_slot.container():
                    render_evidence_details(st, active["results"], active["logs"])
    else:
        st.markdown(
            '<div class="empty-state">' + bank_emblem_html() +
            '<h3>Explore RBI regulations with grounded evidence</h3>'
            '<p>Ask about directions, circulars, clauses, thresholds, applicability, '
            'conditions or exceptions. Every material claim is linked to retrieved evidence.</p></div>',
            unsafe_allow_html=True,
        )

    pending_question_slot = st.empty()
    pending_question = str(st.session_state.get("rbi_pending_question") or "").strip()
    if pending_question:
        pending_question_slot.markdown(
            '<div class="question-row pending-question"><div class="user-avatar">RV</div><div>'
            '<div class="conversation-label">You · processing</div>'
            f'<div class="question-text">{html.escape(pending_question)}</div></div></div>',
            unsafe_allow_html=True,
        )
    progress_slot = st.empty()

    with st.form("question_form", clear_on_submit=True):
        input_column, submit_column = st.columns([0.86, 0.14])
        with input_column:
            question = st.text_area(
                "Question", placeholder="Ask you question on RBI regulations",
                label_visibility="collapsed", height=110,
            )
        with submit_column:
            submitted = st.form_submit_button("↗  Ask", use_container_width=True)
    if submitted and question.strip():
        submitted_question = question.strip()
        clear_evidence_states(st.session_state)
        if evidence_display_slot is not None:
            evidence_display_slot.empty()
        st.session_state["rbi_pending_question"] = submitted_question
        pending_question_slot.markdown(
            '<div class="question-row pending-question"><div class="user-avatar">RV</div><div>'
            '<div class="conversation-label">You · processing</div>'
            f'<div class="question-text">{html.escape(submitted_question)}</div></div></div>',
            unsafe_allow_html=True,
        )
        missing = [
            str(path) for path in (config["corpus_dir"], config["inventory_file"])
            if not path.exists()
        ]
        if missing:
            st.error("Required input path(s) not found:\n\n" + "\n".join(missing))
        else:
            logs = []
            failed = None
            progress_slot.markdown(
                pipeline_progress_html("hybrid"), unsafe_allow_html=True
            )
            for stage, label in STAGES:
                progress_slot.markdown(
                    pipeline_progress_html(stage), unsafe_allow_html=True
                )
                command = build_stage_command(
                    stage=stage, query=submitted_question, corpus_dir=config["corpus_dir"],
                    inventory_file=config["inventory_file"],
                    project_root=config["project_root"], namespace=config["namespace"],
                    rerank_model=config["rerank_model"], llm_model=config["llm_model"],
                )
                try:
                    log = run_stage(command)
                except subprocess.TimeoutExpired as exc:
                    log = {"stage": stage, "returncode": 124, "stdout": exc.stdout or "", "stderr": "Stage timed out."}
                logs.append(log)
                if log["returncode"] != 0:
                    failed = log
                    progress_slot.markdown(
                        pipeline_progress_html(stage, state="error"), unsafe_allow_html=True
                    )
                    break
            if not failed:
                progress_slot.markdown(
                    pipeline_progress_html("cite", state="complete"), unsafe_allow_html=True
                )
            if failed:
                st.error(f"The {failed['stage']} stage failed. Review the diagnostic below.")
                st.code((failed.get("stderr") or failed.get("stdout") or "No diagnostic returned.")[-6000:], language="text")
                with st.expander("Completed-stage logs"):
                    for log in logs:
                        st.code((log.get("stdout") or "") + (log.get("stderr") or ""), language="text")
            else:
                try:
                    results = load_results(config["project_root"])
                    history = st.session_state["rbi_chat_history"]
                    history.append({
                        "question": submitted_question, "results": results, "logs": logs,
                    })
                    st.session_state["rbi_active_chat"] = len(history) - 1
                    st.session_state["rbi_pending_question"] = ""
                    st.rerun()
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    st.error(f"The pipeline completed but its output could not be loaded: {exc}")
