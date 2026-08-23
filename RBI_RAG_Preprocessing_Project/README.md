# RBI Hybrid RAG Preprocessing Project

This Python project implements preprocessing Steps 1-6 for the RBI regulatory-document corpus, Step 7 for Pinecone-hosted semantic indexing, Step 8 for local BM25 lexical indexing, Step 9 for RRF-based hybrid retrieval, Step 11 for Pinecone-hosted reranking, Step 12 for hierarchy-aware context expansion, Step 13 for grounded answers using local Ollama Gemma 3 4B, and Step 14 for deterministic metadata citations and claim-to-chunk mappings.

## Pipeline

1. Validate the PDF corpus and read `RBI_Document_Inventory.xlsx`.
2. Match inventory rows to PDF filenames and classify/extract source content.
3. Create a loss-minimized raw extraction with pages, blocks, lines, spans, coordinates, tables, images, links, metadata, warnings, and validation reports.
4. Create a conservative cleaned extraction while preserving regulatory language, numbering, clauses, definitions, provisos, footnotes, tables, annexures, and amendment notes.
5. Build document hierarchies and embed full heading paths and citation anchors into structured blocks.
6. Re-convert the source PDFs into Docling's structured document model, then create parent and leaf nodes with LlamaIndex `HierarchicalNodeParser` configured with explicit `SentenceSplitter` levels.
7. Generate 512-dimensional `llama-text-embed-v2` vectors with Pinecone Inference, store text and flattened metadata in the existing cosine Pinecone index under namespace `RBI_RAG`, and expose a LangChain vector retriever.
8. Build a persistent local BM25 index from the same final chunks, preserving the Step-6 `chunk_id` values so lexical and semantic results can be fused reliably.
9. Retrieve semantic and lexical candidates, normalize each score set, merge by `chunk_id`, and return a Reciprocal Rank Fusion candidate pool.
11. Rerank the fused candidate pool with a Pinecone-hosted cross-encoder and retain the strongest evidence chunks for answer generation.
12. Expand each retained leaf with its parent section and any governing rule, adjacent table context, repeated headers, or relevant footnote required for safe regulatory interpretation.
13. Give only the expanded evidence to local `gemma3:4b`, validate every cited evidence ID, and render human-readable sources deterministically from metadata.
14. Resolve every answer claim to its supporting evidence IDs and chunk IDs, then generate complete citations from stored metadata without an LLM call.

## Expected project layout

```text
Hybrid_RAG_Project/
├── RBI_Document_Corpus/
│   └── *.pdf
├── RBI_Document_Inventory.xlsx
├── RBI_Raw_Extraction/          # generated
├── RBI_Cleaned_Extraction/      # generated
├── RBI_Document_Hierarchy/      # generated Step-5 structure
├── RBI_Chunked_Documents/       # generated Step-6 chunks
├── RBI_Vector_Indexing/         # generated Step-7 local audit report
├── RBI_BM25_Index/              # generated Step-8 lexical index
├── RBI_Hybrid_Retrieval/        # generated Step-9 query results
├── RBI_Reranked_Retrieval/      # generated Step-11 evidence pool
├── RBI_Expanded_Context/        # generated Step-12 grounded context
├── RBI_Grounded_Answers/        # generated Step-13 answers and audit trail
└── RBI_Citations/               # generated Step-14 citation manifest
```

If the corpus directory and inventory workbook share a parent, that parent is automatically used as `Hybrid_RAG_Project`.

## macOS installation

Open Terminal in this downloaded project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Run Steps 1-6

```bash
rbi-preprocess \
  --corpus-dir "/Users/ravvarma/Documents/Personal/Mastering Agentic AI Certification/Day-2/Hybrid_RAG_Project/RBI_Document_Corpus" \
  --inventory-file "/Users/ravvarma/Documents/Personal/Mastering Agentic AI Certification/Day-2/Hybrid_RAG_Project/RBI_Document_Inventory.xlsx"
```

The command creates four output folders under `Hybrid_RAG_Project`.

To replace existing outputs deliberately:

```bash
rbi-preprocess \
  --corpus-dir "/path/to/Hybrid_RAG_Project/RBI_Document_Corpus" \
  --inventory-file "/path/to/Hybrid_RAG_Project/RBI_Document_Inventory.xlsx" \
  --overwrite
```

## Run individual stages

```bash
rbi-preprocess --corpus-dir "/path/to/RBI_Document_Corpus" --inventory-file "/path/to/RBI_Document_Inventory.xlsx" --step raw
rbi-preprocess --corpus-dir "/path/to/RBI_Document_Corpus" --inventory-file "/path/to/RBI_Document_Inventory.xlsx" --step clean
rbi-preprocess --corpus-dir "/path/to/RBI_Document_Corpus" --inventory-file "/path/to/RBI_Document_Inventory.xlsx" --step hierarchy
rbi-preprocess --corpus-dir "/path/to/RBI_Document_Corpus" --inventory-file "/path/to/RBI_Document_Inventory.xlsx" --step chunks
```

The parent level defaults to 1,000 tokens and the leaf/indexing level to 700 tokens with 100-token overlap. The 150-token minimum is retained as an advisory configuration value. Override these with `--target-tokens`, `--max-tokens`, `--min-tokens`, and `--overlap-tokens`.

Use `--project-root` when the corpus and inventory do not share the same parent.

## Run Step 7: Pinecone semantic indexing

Keep your existing `.env` file either in `Hybrid_RAG_Project/` or in this software folder:

```dotenv
PINECONE_API_KEY=your_private_key
PINECONE_INDEX_NAME=your_existing_index_name
```

The `.env` file is ignored by Git. Do not add secrets to `.env.example`.

Install the new Step-7 dependencies after updating the project:

```bash
conda activate rbi-rag
python -m pip install -e .
```

Index all final chunks into namespace `RBI_RAG`:

```bash
rbi-preprocess \
  --corpus-dir "../RBI_Document_Corpus" \
  --inventory-file "../RBI_Document_Inventory.xlsx" \
  --project-root ".." \
  --step index \
  --overwrite
```

Index and immediately run a semantic-search verification query:

```bash
rbi-preprocess \
  --corpus-dir "../RBI_Document_Corpus" \
  --inventory-file "../RBI_Document_Inventory.xlsx" \
  --project-root ".." \
  --step index \
  --namespace "RBI_RAG" \
  --retriever-k 5 \
  --query "What customer identification is required under RBI KYC directions?" \
  --overwrite
```

Step 7 validates that the existing index uses 512 dimensions and cosine similarity before requesting embeddings. Pinecone Inference embeds documents as `passage` inputs and retriever searches as `query` inputs using `llama-text-embed-v2`; no OpenAI API key or OpenAI credits are required. It uses each `chunk_id` as the deterministic Pinecone vector ID, making repeat uploads idempotent. On completion it fetches and prints the first vector ID from the first document for verification. It does not create or delete the Pinecone index.

The local audit output is:

```text
RBI_Vector_Indexing/pinecone_indexing_report.json
```

## Run Step 8: BM25 lexical indexing

Build the local lexical index from `RBI_Chunked_Documents/chunks.jsonl`:

```bash
rbi-preprocess \
  --corpus-dir "../RBI_Document_Corpus" \
  --inventory-file "../RBI_Document_Inventory.xlsx" \
  --project-root ".." \
  --step bm25 \
  --overwrite
```

Optionally run a lexical verification query immediately after building it:

```bash
rbi-preprocess \
  --corpus-dir "../RBI_Document_Corpus" \
  --inventory-file "../RBI_Document_Inventory.xlsx" \
  --project-root ".." \
  --step bm25 \
  --query "KYC clause 33 beneficial owner" \
  --bm25-top-k 5 \
  --overwrite
```

Step 8 uses `rank_bm25.BM25Okapi` with configurable defaults `k1=1.5` and `b=0.75`. It indexes the chunk content together with retrieval-critical metadata such as title, document type, heading path, section, and clause. It fails fast if any `chunk_id` is missing or duplicated.

The generated files are:

```text
RBI_BM25_Index/
├── bm25_index.pkl
├── chunk_store.jsonl
└── bm25_index_report.json
```

`chunk_store.jsonl` stores the original `chunk_id`, content, metadata, and the tokenized text used by BM25. Only load `bm25_index.pkl` from this trusted pipeline because Python pickle files must not be accepted from untrusted sources.

## Run Step 9: hybrid retrieval

Step 9 requires the completed Pinecone index, the local Step-8 BM25 index, and a question:

```bash
rbi-preprocess \
  --corpus-dir "../RBI_Document_Corpus" \
  --inventory-file "../RBI_Document_Inventory.xlsx" \
  --project-root ".." \
  --step hybrid \
  --query "What are the customer due diligence requirements under RBI KYC directions?"
```

The default hyperparameters are:

```text
Pinecone candidates: 20
BM25 candidates:     20
RRF constant:        60
Merged candidates:   20
```

Override them when evaluating retrieval:

```bash
rbi-preprocess \
  --corpus-dir "../RBI_Document_Corpus" \
  --inventory-file "../RBI_Document_Inventory.xlsx" \
  --project-root ".." \
  --step hybrid \
  --query "What does CRR mean?" \
  --pinecone-candidates 20 \
  --bm25-candidates 20 \
  --rrf-constant 60 \
  --merged-candidates 20
```

Pinecone cosine and BM25 scores are min-max normalized independently and retained for diagnostics. Final ordering uses rank-based RRF, avoiding direct arithmetic comparison between incompatible raw score scales. BM25 candidates with zero lexical score are excluded from the RRF vote.

Step 9 writes:

```text
RBI_Hybrid_Retrieval/
├── hybrid_candidates.json          # latest question and complete candidate pool
├── hybrid_retrieval_report.json    # latest run summary
└── queries/
    └── <query_hash>.json            # repeatable result for each distinct question
```

Each result contains the final rank, `chunk_id`, RRF score, normalized score summary, source-specific ranks and scores, content, metadata, and whether it was found by semantic retrieval, lexical retrieval, or both.

## Run Step 11: Pinecone-hosted reranking

First run Step 9 for the question, then rerank its latest fused candidate pool:

```bash
rbi-preprocess \
  --corpus-dir "../RBI_Document_Corpus" \
  --inventory-file "../RBI_Document_Inventory.xlsx" \
  --project-root ".." \
  --step rerank \
  --query "What are the customer due diligence requirements under RBI KYC directions?"
```

The defaults rerank the top 20 hybrid candidates with Pinecone-hosted `cohere-rerank-3.5` and retain the top 8:

```text
Reranker input:  20
Reranker output: 8
Model:           cohere-rerank-3.5
```

`cohere-rerank-3.5` is available through Pinecone on Standard and Enterprise plans. For a Starter or Builder plan, select Pinecone-hosted BGE instead:

```bash
rbi-preprocess \
  --corpus-dir "../RBI_Document_Corpus" \
  --inventory-file "../RBI_Document_Inventory.xlsx" \
  --project-root ".." \
  --step rerank \
  --query "What are the customer due diligence requirements under RBI KYC directions?" \
  --rerank-model "bge-reranker-v2-m3" \
  --rerank-candidates 20 \
  --rerank-top-n 8
```

The reranker receives each chunk together with its document title and heading path. The output preserves hybrid rank, RRF score, semantic/BM25 diagnostics, content, metadata, and `chunk_id`, then adds `reranker_rank` and `reranker_score`.

```text
RBI_Reranked_Retrieval/
├── reranked_candidates.json       # latest evidence pool
├── reranking_report.json          # latest run summary and usage
└── queries/
    └── <query_hash>.json           # result for each distinct question
```

Step 11 is query-time and does not require `--overwrite`. It does not update or rebuild Pinecone or BM25.

## Run Step 12: parent-context expansion

After Step 11 completes, expand the reranked chunks using the Step-6 LlamaIndex hierarchy:

```bash
rbi-preprocess \
  --corpus-dir "../RBI_Document_Corpus" \
  --inventory-file "../RBI_Document_Inventory.xlsx" \
  --project-root ".." \
  --step expand \
  --query "What are the customer due diligence requirements under RBI KYC directions?"
```

The question must match the latest Step-11 result. You may omit `--query` to expand that latest result directly:

```bash
rbi-preprocess \
  --corpus-dir "../RBI_Document_Corpus" \
  --inventory-file "../RBI_Document_Inventory.xlsx" \
  --project-root ".." \
  --step expand
```

Step 12 always attaches the exact parent section. When indicated by the content or Step-6 metadata, it also attaches:

- The preceding governing rule for a chunk beginning with `Provided that`, `except where`, `notwithstanding`, or a similar legal dependency.
- An adjacent clause that supplies a table heading or prior rule.
- A following node when a table continues across the chunk boundary.
- Markdown table column headers, repeated explicitly in the assembled context.
- A referenced footnote found in the parent or an adjacent hierarchy node.

Every expansion is tied to `llamaindex_node_id`, `parent_node_id`, `chunk_id`, and a reason code. Shared parent nodes are deduplicated in the context catalog.

```text
RBI_Expanded_Context/
├── expanded_context.json            # latest complete evidence package
├── context_expansion_report.json    # latest expansion summary
└── queries/
    └── <query_hash>.json              # result for each distinct question
```

Within `expanded_context.json`, each evidence record retains all Step-11 ranking fields and adds `expanded_content` plus an `expansion` object containing node IDs, reasons, table headers, footnotes, legal-dependency flags, and an estimated assembled token count.

Step 12 is local and query-time. It makes no Pinecone or model API calls and does not require `--overwrite`.

## Run Step 13: grounded answer generation with Gemma 3 4B

Step 13 uses the local Ollama API and requires no OpenAI key. Confirm Ollama is installed and pull the requested model once:

```bash
ollama --version
ollama pull gemma3:4b
ollama list
```

Ollama's `gemma3:4b` package is an instruction-following 4.3B model. The packaged project uses structured-output JSON schema support and a conservative 32K runtime context even though the model supports a larger maximum context.

Generate an answer for the same question used by Steps 9–12:

```bash
rbi-preprocess \
  --corpus-dir "../RBI_Document_Corpus" \
  --inventory-file "../RBI_Document_Inventory.xlsx" \
  --project-root ".." \
  --step answer \
  --query "What are the customer due diligence requirements under RBI KYC directions?" \
  --llm-model "gemma3:4b"
```

You may omit `--query` to answer the latest Step-12 question:

```bash
rbi-preprocess \
  --corpus-dir "../RBI_Document_Corpus" \
  --inventory-file "../RBI_Document_Inventory.xlsx" \
  --project-root ".." \
  --step answer
```

Default generation settings are:

```text
Ollama host:             http://localhost:11434
Model:                   gemma3:4b
Context window:          32768 tokens
Evidence budget:         approximately 22000 tokens
Maximum output:          1800 tokens
Temperature:             0.1
Seed:                    42
Validation repairs:      1
```

Override them when needed:

```bash
rbi-preprocess \
  --corpus-dir "../RBI_Document_Corpus" \
  --inventory-file "../RBI_Document_Inventory.xlsx" \
  --project-root ".." \
  --step answer \
  --llm-model "gemma3:4b" \
  --llm-context-tokens 32768 \
  --llm-evidence-token-budget 22000 \
  --llm-max-output-tokens 1800 \
  --llm-temperature 0.1
```

The generator assigns deterministic evidence IDs (`E1`, `E2`, and so on), supplies only Step-12 expanded evidence, and requires structured fields for answer status, claims, conditions, conflicts, regulatory status, and used evidence IDs. It rejects unknown or invented IDs. Human-readable citations are built by Python from title, heading path, pages, chunk ID, and regulatory status rather than generated by the model.

If the first response fails JSON or citation validation, the model receives one constrained correction request. Both attempts are retained. If neither passes, the program records `generation_error` and does not present an unsupported regulatory conclusion.

```text
RBI_Grounded_Answers/
├── latest_answer.json
├── latest_answer.md
├── answer_generation_report.json
├── invalid_responses/
└── queries/
    └── <query_hash>/
        ├── prompt.txt
        ├── evidence.json
        ├── model_response.json
        └── answer.md
```

Step 13 is query-time and does not require `--overwrite`. It calls only your local Ollama server and does not call OpenAI, Pinecone, or Cohere.

## Run Step 14: deterministic citation generation

After Step 13 completes, build citations and the explicit claim-to-chunk map:

```bash
rbi-preprocess \
  --corpus-dir "../RBI_Document_Corpus" \
  --inventory-file "../RBI_Document_Inventory.xlsx" \
  --project-root ".." \
  --step cite \
  --query "What are the customer due diligence requirements under RBI KYC directions?"
```

You can omit `--query` to process the latest Step-13 answer:

```bash
rbi-preprocess \
  --corpus-dir "../RBI_Document_Corpus" \
  --inventory-file "../RBI_Document_Inventory.xlsx" \
  --project-root ".." \
  --step cite
```

Step 14 resolves each LLM-selected `E#` identifier against the immutable evidence snapshot saved by Step 13. It creates citation objects containing:

- Document title
- Full heading path
- Section or clause
- Page start, end, and formatted page range
- Source URL
- Chunk ID
- Document type and regulatory status
- Metadata-completeness warnings

It maps the answer summary, every structured material claim, every condition or exception, every conflict, and the regulatory-status note to supporting evidence IDs and chunk IDs. Unknown evidence IDs or claims without supporting chunks are mapping errors.

Source URLs, headings, sections, and pages are never inferred by the model. When a field is absent from stored metadata, the citation records `null` plus a warning and sets the run to `review_required`.

```text
RBI_Citations/
├── citation_manifest.json
├── claim_citation_map.json
├── answer_with_citations.md
├── citation_generation_report.json
└── queries/
    └── <query_hash>/
        ├── citation_manifest.json
        ├── claim_citation_map.json
        └── answer_with_citations.md
```

Citation statuses are:

```text
passed              Every claim is mapped and used citation metadata is complete.
review_required     A mapping or required metadata field is incomplete.
no_citable_answer   Step 13 abstained or failed without making a grounded claim.
```

Step 14 is entirely local, makes no LLM or Pinecone calls, and does not require `--overwrite`.

## Important outputs

`RBI_Raw_Extraction/raw_extractions/` contains one loss-minimized JSON file per PDF.

`RBI_Cleaned_Extraction/cleaned_documents/` contains one cleaned JSON file per PDF plus an auditable removal log.

`RBI_Document_Hierarchy/structured_documents/` is the recommended input for the next chunking step because every block contains its full heading path and citation anchor.

`RBI_Chunked_Documents/chunks.jsonl` contains the leaf nodes for BM25/vector indexing. Every line has a `content` string and a `metadata` object.

`RBI_Chunked_Documents/node_hierarchy.jsonl` contains every parent and leaf node plus LlamaIndex relationship IDs. Store these nodes in the LlamaIndex document store if you plan to use parent expansion or `AutoMergingRetriever`.

`RBI_Chunked_Documents/docling_documents/` preserves Docling JSON and page-marked Markdown for auditing and citation tracing. The first Docling run can take longer because its layout/OCR models may need to be downloaded and cached.

`RBI_BM25_Index/` contains the local lexical index and auditable chunk store used in the future hybrid-retrieval fusion step.

`RBI_Hybrid_Retrieval/` contains the latest merged candidate pool and query-specific retrieval traces suitable for evaluation and the next reranking stage.

`RBI_Reranked_Retrieval/` contains the focused 5–8-chunk evidence pool to use for hierarchy expansion, citation assembly, and grounded answer generation.

`RBI_Expanded_Context/` contains the hierarchy-complete evidence package intended for citation assembly and grounded answer generation. Use `expanded_content`, not the isolated leaf text, when a legal dependency has been detected.

`RBI_Grounded_Answers/` contains the final Markdown answer, structured answer, prompt/evidence snapshot, model responses, validation results, and per-question audit trail.

Step 13 includes a Gemma 3 4B compatibility layer. It accepts fenced or prose-wrapped JSON, canonicalizes citation variants such as `(Evidence E 1)` to `[E1]`, normalizes harmless schema omissions, reconstructs a missing claim record, and attaches missing `[E#]` citations only when the claim has sufficient lexical overlap with that evidence and every stated number is present in it. An answer summary may inherit citations from already verified claims only when their substantive terms and numeric values agree. Unsupported, unrelated, or numerically conflicting claims remain validation failures. Every automatic change is recorded under `repair_actions` in the query's `model_response.json`.

If a Gemma summary still cannot be citation-matched but its individual claims have already passed grounding, Step 13 deterministically rebuilds the displayed answer solely from those grounded claim texts and their verified `[E#]` identifiers. This removes the missing-inline-citation failure mode without preserving an unsupported summary. The Streamlit sidebar displays the running package version so stale editable installs are immediately visible.

Model-supplied inline evidence IDs are not trusted during schema repair. Step 13 removes them, independently maps the answer to the evidence text, and emits the same verified IDs in both the answer and `evidence_ids_used`; this prevents mismatches such as an answer citing `E8` while the structured field contains only `E2`. A `partially_answered` status is reconciled to `answered` only when grounded claims exist and neither the answer nor regulatory-status note identifies an evidence limitation or conflict.

`RBI_Citations/` contains deterministic source records and the auditable mapping from every answer claim to evidence IDs and supporting Step-6/Pinecone `chunk_id` values.

Review `validation_report.json` before indexing. The report flags chunks that start with legal qualifiers; unlike the earlier custom chunker, the framework-only pipeline does not guarantee that every proviso or exception remains attached to its governing rule.

At the end of Step 6, the command prints the total input-document count, final leaf-chunk count, and a complete chunk-by-chunk trace for the longest document. Each leaf chunk also carries `start_index` and `end_index` metadata relative to its page-marked Docling Markdown source, making overlap and citation boundaries auditable.

Step 6 resolves each original PDF using the exact extracted source filename first, then its SHA-256 fingerprint, and finally a normalized filename. This prevents `/` and other punctuation in inventory titles from being interpreted as filesystem path separators.

## Step 16 — Streamlit Q&A interface

The application runs Steps 9, 11, 12, 13, and 14 for each submitted question. It shows the grounded answer, conditions and exceptions, deterministic citations, expandable hierarchy-expanded evidence, source links, retrieval modes and scores, and a prominent insufficient-evidence state. Its debug panel separates Pinecone semantic results, BM25 lexical results, RRF fusion, Pinecone reranking, and stage logs.

The production-style Executive interface is branded **RBI Regulation Assistant** with **Grounded Regulatory Q&A** as its heading. It uses the selected dark **Split workspace** design: a charcoal navigation sidebar, a conversation pane for the question and grounded response, and a dedicated right-side **Evidence Details** inspector containing citations, expanded excerpts, source links, retrieval scores and diagnostics. Evidence details remain hidden until the user selects Evidence; repeated clicks keep the inspector open, and it resets only for New chat or a newly submitted question. Download answer exports the question, response and deterministic sources as Markdown. The submitted question is preserved in session state and remains visible while the pipeline is running. Retrieval, reranking, parent-context expansion and Gemma answer-generation progress appears as an in-flow middle section beneath the guidance content instead of overlapping the screen header. After completion, the completed progress strip remains visible immediately above the answer. Earlier questions are retained only in Streamlit session state and restore their complete answer and evidence when selected. Corpus paths, inventory path, namespace, reranker and Ollama model remain server-side and are not displayed to end users. The interface deliberately does not show either a grounded-status pill or a `Hybrid · Reranked` badge.

Prerequisites:

- Complete Pinecone indexing (Step 7) and local BM25 indexing (Step 8).
- Keep `PINECONE_API_KEY` and `PINECONE_INDEX_NAME` in `.env` in `Hybrid_RAG_Project` or `RBI_RAG_Preprocessing_Project`.
- Start Ollama and make sure `ollama list` includes `gemma3:4b`.
- Reinstall the editable project after upgrading so Streamlit is installed: `python -m pip install -e .`

From `RBI_RAG_Preprocessing_Project`, run:

```bash
python -m streamlit run streamlit_app.py
```

Streamlit normally opens `http://localhost:8501`. The sidebar defaults to the enclosing `Hybrid_RAG_Project`; adjust the corpus and inventory paths there if needed. Credentials stay in the server process and are not displayed in the application.

## Step 17 — Evaluation dataset

Step 17 creates a curated evaluation set with direct lookup, exact terminology, multi-clause, multi-document, table, ambiguous, citation-sensitive, unanswerable, adversarial, supersession, applicability, entity-specific, and retrieval-precision questions. Each record stores:

- `expected_answer`
- validated `supporting_chunk_ids`
- deterministic `required_citations` built from Step-6 metadata
- `should_abstain` and `abstention_reason`
- claim-grounding and qualifier-preservation evaluation criteria

The source definitions use document-and-phrase selectors rather than hard-coded chunk IDs. This is intentional: Docling and LlamaIndex version changes can alter chunk boundaries. At execution time, the stage resolves selectors against your current `chunks.jsonl`, records the actual deterministic IDs, and fails if an answerable question cannot be grounded.

Run from `RBI_RAG_Preprocessing_Project`:

```bash
rbi-preprocess \
  --corpus-dir "../RBI_Document_Corpus" \
  --inventory-file "../RBI_Document_Inventory.xlsx" \
  --project-root ".." \
  --step evaluate \
  --overwrite
```

Outputs:

```text
RBI_Evaluation_Dataset/
├── evaluation_dataset.json
├── evaluation_dataset.jsonl
├── evaluation_dataset_review.md
└── evaluation_dataset_report.json
```

Review `evaluation_dataset_review.md` once before treating the expected answers as a final regulatory gold set. The JSONL file is the recommended input for automated retrieval, answer-grounding, citation, and abstention evaluations.
