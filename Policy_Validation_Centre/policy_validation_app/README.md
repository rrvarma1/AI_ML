# Policy Validation Center

A production-style Streamlit prototype for uploading a deal document, extracting structured terms, validating those terms with Agent 1, assessing risk with Agent 2, and retaining prior runs in SQLite.

## Run locally

```bash
cd policy_validation_app
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

The application supports text-based PDF and TXT files. Scanned PDFs require OCR, which is intentionally outside this prototype.

## Included

- Bright, responsive upload workspace
- LLM-backed deal-term extraction with a strict Pydantic output contract
- Case-specific `extractor_output.json` handoff managed by the Orchestrator
- Agent 1 deterministic validation using `rules.json` plus a Python rule engine
- Agent 2 deterministic financial-risk evaluation using `risk_rules.json`
- LangGraph `StateGraph` orchestration with parallel specialist branches
- Human-in-the-loop `interrupt()` and `Command(resume=...)` deal-officer decisions
- Gemma 3 4B narrative generation constrained to deterministic Agent 2 evidence
- Summary metrics, missing-field checks and reviewable JSON
- Downloadable JSON report
- Searchable SQLite validation history
- Safe document hashing; source document bytes are not retained

Only the deal document is uploaded. The extractor is a preprocessing component before Agent 1. The Orchestrator writes its structured output to `data/cases/<case_id>/extractor_output.json`; Agent 1 reads that file, loads `rules.json`, and executes the rules through `validation_rules.py`.

## Extractor output contract

`DealTerms` is a Pydantic v2 model used as the extractor output contract. The extractor runs at temperature zero and uses Pydantic only to enforce output shape, data types and safe serialization. It does not calculate LTV, compare financial values, apply thresholds, produce compliance findings or make a recommendation. Missing evidence remains `null`.

Financial and policy validation is performed by Agent 1 using deterministic Python logic. Agent 1 does not use an LLM to decide PASS or FAIL.

The extraction contract includes loan amount, term in years and months, rate and stated interest amount, DSCR, leverage and liquidity inputs, revenue growth, separate market and realisable collateral values, stated LTV and coverage, total facility amount, and repayment instalment details. The extractor never derives a missing field.

The default backend is local Ollama using `gemma3:4b`. OpenAI and an explicitly labelled deterministic offline fallback remain available in the UI. Put optional OpenAI credentials in `.env`; never hard-code them.

Before starting the application, install Ollama and pull the default model:

```powershell
ollama pull gemma3:4b
ollama list
```

Gemma runs in Ollama JSON mode. The application parses that JSON with Pydantic after generation instead of passing a complex grammar to Ollama. This avoids the `Failed to initialize samplers: failed to parse grammar` error seen with grammar-constrained structured output.

For reliable local extraction, the precursor uses a compact flat JSON template, rejects all-null responses, retries Gemma once when critical terms are missed, and fills remaining nulls only from explicit labelled document values. It never derives financial values or applies compliance rules.

## Orchestrator, Agent 1 and Agent 2

```text
START → Extractor ─┬→ Agent 1 Compliance ← rules.json ─┐
                  └→ Agent 2 Risk & Summary ← risk_rules.json ─┤
                                                              ├→ outcome router
                                                              ├→ success → END
                                                              └→ adverse → interrupt → Deal Officer → resume → END
```

The Orchestrator is a LangGraph `StateGraph`. It starts after the user selects a document and clicks **Validate**, invokes the Extractor, then fans out to Agent 1 and Agent 2 as parallel graph branches. Both agents receive the same extractor output and neither depends on the other agent's result. A join node evaluates both outcomes. Successful reviews finish automatically; adverse reviews call LangGraph `interrupt()` and checkpoint the full graph state in a shared `MemorySaver`. The validation ID is the stable `thread_id`. The Deal Officer reviews all prior outputs and resumes the same checkpoint with `Command(resume="Approve" | "Reject" | "On Hold")`.

`MemorySaver` is process-memory persistence suitable for this prototype. A server restart clears paused checkpoints; a production deployment should replace it with a durable database-backed LangGraph checkpointer.

`rules.json` owns thresholds, operators, severity and messages. `validation_rules.py` owns safe operator execution and rule-catalog validation. `agent1.py` owns missing-data handling, finding construction, scoring and overall compliance status. Missing fields produce `INSUFFICIENT_DATA`; any failed rule produces `NON_COMPLIANT`.

The included values are explicitly dummy demonstration thresholds and must not be treated as real lending policy.

Agent 2 consumes only the Extractor's typed `DealTerms`. It calculates LTV, realisable collateral coverage and collateral valuation haircut in Python, and evaluates DSCR, debt-to-equity, current ratio and revenue growth. It does not consume Agent 1 findings, status or score. Risk aggregation and recommendations are deterministic. Gemma 3 4B receives only the resulting Agent 2 evidence and writes the narrative summary; its response cannot add findings to the typed `Agent2Result`. If the model is unavailable or introduces an unsupported number, the application returns a deterministic grounded summary instead.

Agent 2 recommendations are limited to `PROCEED`, `PROCEED_WITH_CONDITIONS`, `MANUAL_REVIEW`, `DECLINE` and `INSUFFICIENT_INFORMATION`. They are review recommendations only and do not approve or reject a real loan.
