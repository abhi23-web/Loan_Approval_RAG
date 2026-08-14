# Architecture

## The central design decision

**The language model does not decide. It explains.**

```
Applicant data ──► Rule engine ──► DECISION (deterministic, recomputable)
                        │
Policy corpus ──► Retrieval ──► clauses ──► LLM ──► EXPLANATION (cited, validated)
```

`RuleAssessment` and `GroundedExplanation` are separate types all the way to the
API response and the UI. They are never merged into one free-text blob, so a
reader can always tell *"policy says X"* from *"the model wrote X"*.

Three consequences follow, and they are the reason for the split:

1. A rephrased application cannot change the outcome. The decision is arithmetic.
2. Every decision is recomputable from the inputs and the configured thresholds,
   which is what makes it defensible to a regulator.
3. If the model is unavailable or its answer fails citation validation, the
   applicant still gets a decision and the rule checks behind it — only the
   prose is replaced.

## Request path

```
Streamlit
  └─ POST /api/v1/loan-assessment
       └─ AssessmentService.assess
            ├─ version store: reload if another process changed it
            ├─ guard: refuse if nothing is indexed  → HTTP 503
            └─ HomeLoanRagPipeline.assess_application
                 ├─ EligibilityRuleEngine.assess      → the decision
                 ├─ build_policy_query                → deterministic query
                 ├─ PolicyRetriever.retrieve          → version-filtered chunks
                 ├─ assemble_context                  → numbered [S1..Sn], budgeted
                 ├─ build_assessment_prompt
                 └─ GroundedAnswerGenerator.generate  → validated citations
```

Everything from `HomeLoanRagPipeline` down is traced as its own LangSmith span,
so the trace tree is this diagram.

## Ingestion path

```
source_registry.yaml
  └─ DocumentFetcher      conditional GET → 304? stop. byte hash match? stop.
       └─ extract_document   PDF (per page) | HTML (content container) | text
            └─ clean_document   repeated headers, page numbers, web boilerplate
                 └─ text hash match? stop.          ← the gate that matters most
                      └─ VersionStore.register_version   append + elect active
                           └─ chunk → embed → ChromaDB upsert (deterministic ids)
```

## Module boundaries and why they are where they are

| Boundary | Rule | What it buys |
| --- | --- | --- |
| `app/api` → `app/services` | Routes do HTTP only | The pipeline is reachable from the evaluator and a future worker without FastAPI |
| `app/services/container.py` | The only place collaborators are constructed | ChromaDB, embeddings and the rule engine are built once, not per request |
| `app/ingestion/vector_store.py` | The only module importing `chromadb` | Swapping the vector database is a one-file rewrite |
| `app/rag/llm.py`, `embeddings.py` | The only modules speaking to Ollama | Adding OpenAI is a new provider class, not a refactor |
| `app/rules/` | Imports nothing from `app/rag` | The decision cannot accidentally become model-dependent |
| `config/`, `documents/` | No URL or threshold in Python | Policy tuning and source changes are config edits |

## Concurrency and process model

Three processes may run at once against the same `data/` directory:

- the API (`run_backend.py`),
- the watcher (`scripts/run_watcher.py`),
- an ad-hoc CLI run (`scripts/ingest.py`).

Two mechanisms keep them consistent:

- **Atomic version-store writes.** `write_json_atomic` writes a temp file,
  `fsync`s, then renames. A crash mid-write cannot truncate the audit trail.
- **mtime-based reload.** The API calls `reload_if_changed()` on the request path.
  It is a `stat()`. Without it, a running API would keep filtering to a superseded
  version until restarted — the bug where "I ingested the new policy and nothing
  changed".

ChromaDB is embedded and single-writer. Ingestion from two processes at once is
not supported and is not needed: the watcher owns writes, and everything else
reads.

## Performance decisions

| Problem | Approach | Effect |
| --- | --- | --- |
| Re-downloading unchanged documents | Conditional GET with stored ETag/Last-Modified | A poll costs one round trip |
| Re-embedding unchanged documents | Byte hash, then text hash after cleaning | The nightly poll embeds nothing |
| Re-embedding the same query | LRU cache on `embed_query` | A 5-strategy × 4-top-k sweep embeds 10 questions once, not 200 times |
| Per-request embedding round trips | Batched `embed_documents` | 400 chunks → 25 requests, not 400 |
| Re-opening the index per request | `lru_cache` on the Chroma client, container singletons | Index open cost paid once at startup |
| Duplicate chunks after a retry | Deterministic chunk ids + `upsert` | An interrupted ingest is safe to simply repeat |
| Prompt bloat as `top_k` rises | `max_context_characters` budget | Cost and latency stay bounded during experiments |
| One document monopolising top-k | `max_chunks_per_source` | A 200-page circular cannot win on volume |

Complexity notes, with `n` chunks and `k = top_k`: ingestion is `O(n)` embedding
calls and `O(n)` storage; retrieval is one embedding call plus ChromaDB's HNSW
search, roughly `O(log n)`, then `O(3k log 3k)` to sort the over-fetched
candidates. Memory is bounded by the batch size at ingestion and by `3k` chunks at
query time — the corpus is never fully loaded into the process.

## Security decisions

- Secrets only in the environment; `.env` gitignored; `.env.example` committed.
- The registry is configuration, not a trusted source of URLs. Schemes are
  checked against an allow-list, and `file://` paths must resolve inside the
  repository — so a config edit cannot become arbitrary local file read.
- Downloads are streamed with a byte cap, so a mis-registered URL cannot exhaust
  disk or memory.
- Applicant names never reach prompts, logs or traces. Financial figures do,
  because they are what makes a trace useful for debugging a decision.
- Pydantic models use `extra="forbid"`, so an unexpected field is an error rather
  than a silently ignored input.

## What would change at production scale

| Today | At scale | Trigger |
| --- | --- | --- |
| ChromaDB embedded | pgvector / Qdrant / managed | > a few million chunks, or concurrent writers |
| Watcher polls on an interval | Queue-backed jobs, per-source schedules | Dozens of sources with different cadences |
| Synchronous `/documents/refresh` | Job id + status endpoint | Ingestion exceeding a request timeout |
| Local Ollama | A served inference tier | Concurrency beyond one machine |
| Version store as JSON | A relational table | Version history that needs querying or joins |
| No auth | OIDC + per-role access | Any real applicant data |

The boundaries above are drawn so that each of these is a contained change.
