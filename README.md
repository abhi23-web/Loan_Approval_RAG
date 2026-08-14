# Home Loan Approval Automation — RAG + Rules + LangSmith + FastAPI + Streamlit

A home loan assessment system that decides with deterministic rules and explains
with retrieval-augmented generation grounded in versioned policy documents.

**Start here:** [`RUNBOOK.md`](RUNBOOK.md) walks the whole system up block by
block, offline first, with expected output and failure modes for every step.

---

## 1. Project overview

An applicant submits a loan application in a Streamlit form. A FastAPI backend
runs deterministic eligibility rules over the figures, retrieves the policy
clauses those rules correspond to from a ChromaDB index of versioned documents,
and asks a local LLM to explain the decision using only those clauses. The
response carries the decision, the arithmetic, and citations that each name a
document and a specific version of it.

```
Streamlit form
      │ HTTP
      ▼
FastAPI  ──►  Rule engine (decides)
      │
      ├──►  Retriever ──► ChromaDB ──► version-filtered chunks
      │                                        │
      └──►  Context assembly ──► LLM ──► explanation + validated citations
      │
      ▼
Response: decision · rule checks · explanation · source · version · page

Ingestion (separate long-lived process)
   registry ──► fetch ──► extract ──► clean ──► detect change ──► version
            ──► chunk ──► embed ──► ChromaDB

LangSmith: traces every retrieval and generation; hosts evaluation runs
Golden dataset: 10 questions, run repeatedly, as the regression suite
```

## 2. Business problem

Loan eligibility is decided by policy documents that change. A bank revises its
credit policy; a regulator issues a new circular. Three failures follow, and this
system is built around preventing each:

1. **A stale answer.** An assistant that quotes last year's minimum credit score
   is not slightly wrong — it is wrong in a way the applicant cannot detect.
   Answered by document versioning with an active-version filter.
2. **An unverifiable answer.** "You are not eligible" without a clause reference
   is not actionable and not auditable. Answered by mandatory citations drawn
   from a closed set the system controls.
3. **A decision made by a language model.** A model can be argued out of an
   answer by rephrasing. Answered by putting the decision in Python and giving
   the model only the explanation.

## 3. Architecture

| Layer | Module | Responsibility |
| --- | --- | --- |
| Frontend | `frontend/streamlit_app.py` | Form, validation, result rendering |
| API | `app/api/` | HTTP only: schemas, routes, error mapping |
| Services | `app/services/` | Application operations, composition root |
| Rules | `app/rules/eligibility.py` | **The decision.** Deterministic, testable |
| RAG | `app/rag/` | Query building, retrieval, context, generation, citation validation |
| Ingestion | `app/ingestion/` | Registry, fetch, extract, clean, version, chunk, embed, store |
| Watcher | `app/watcher/` | Long-lived polling process |
| Evaluation | `evaluation/` | Golden dataset, metrics, runner, experiments |
| Config | `config/`, `documents/` | Settings, chunking strategies, source registry |

Layering rule: routes call services, services call the pipeline, the pipeline
calls collaborators built once in `app/services/container.py`. Nothing below the
service layer imports FastAPI, which is why the evaluation harness can run the
exact same pipeline the API runs.

## 4. Technology stack

| Choice | Version | Why |
| --- | --- | --- |
| Python | 3.11+ | Modern typing without `typing.Optional` noise |
| FastAPI + Uvicorn | 0.141 / 0.41 | Pydantic-native validation, free OpenAPI |
| Pydantic | 2.13 | One definition of every contract, strictly validated |
| ChromaDB | 1.5.9 | Local persistence + metadata filtering (see §6) |
| Ollama | any current | Local LLM and embeddings, no per-token cost |
| LangSmith SDK | 0.10 | Tracing and evaluation, used directly |
| Streamlit | 1.61 | Fast, professional-enough form UI |
| pypdf, BeautifulSoup, lxml | current | PDF and HTML extraction |

**No LangChain.** The LangSmith SDK is used directly. Two reasons: the trace tree
then mirrors this application's own functions rather than a framework's internal
runnables, so a bad answer maps to a file you can open; and a project whose whole
point is comparing chunkers should not also be comparing framework versions. The
chunkers here are about 150 lines and fully inspectable.

## 5. Why RAG

Home loan policy is long, numeric, and revised. Fine-tuning would bake a
snapshot into weights and make "which version said that" unanswerable. Putting
whole documents in the prompt does not scale past a couple of circulars and pays
for every irrelevant clause on every request. RAG gives per-answer provenance,
which for lending is not a nice-to-have: it is the difference between an
explanation and an assertion.

## 6. Why ChromaDB

**What it buys here**

- **Local persistence.** `data/chroma/` survives restarts. The API never
  re-embeds on boot; ingestion is the only thing that writes vectors.
- **Metadata filtering.** `where={"version_id": {"$in": [...]}}` is what makes
  version-aware retrieval a filter rather than a second index. This is the single
  biggest reason for the choice.
- **Embedded, not a service.** A reviewer clones the repo and gets an answer.
  No container, no cluster, no signup.
- **Per-strategy collections.** Five chunking strategies coexist on disk and are
  evaluated against the same dataset with no rebuild between runs.
- **Straightforward Python integration**, and easy to reason about at this scale
  (thousands of chunks, not millions).

**Trade-offs, honestly**

| | ChromaDB | FAISS | Pinecone |
| --- | --- | --- | --- |
| Metadata filtering | First-class | Manual, you build it | First-class |
| Persistence | Built in | You write it | Managed |
| Operations | None | None | A service + a bill |
| Raw ANN speed at 10M+ | Slower | Fastest | Fast, managed |
| Horizontal scale | Limited | You build it | Built in |
| Fit for this project | Best | Over-manual | Over-engineered |

**When ChromaDB stops being right:** more than a few million chunks; multiple
writers needing real concurrency; multi-tenant isolation; five-nines availability;
or a corpus large enough that index build time needs distributing. At that point
the migration is contained — `app/ingestion/vector_store.py` is the only module
that imports `chromadb`, and `VectorMatch` is the boundary type. Swapping in
pgvector or Qdrant is a rewrite of one file.

## 7. Why FastAPI

Pydantic models are the contract, so validation, OpenAPI docs and Python types
are one definition rather than three that drift. Domain exceptions map to status
codes in exactly one place (`app/api/main.py`), so a new route cannot leak a
stack trace or return 500 for a user error. Routes contain no business logic —
that is what makes the same code path testable and reusable by the evaluator.

## 8. Why Streamlit

The frontend needs to be a professional-looking form and a good result renderer,
not a web application. Streamlit gets there in one file, in Python, with no build
step. It computes nothing: two implementations of the eligibility rules would
eventually disagree, and the one the applicant sees would be the wrong one.

## 9. Why LangSmith

Used as a debugging and evaluation tool, not a logger. A trace shows question →
retrieved chunks with similarities and the version filter that produced them →
the assembled prompt → the answer → which citations survived validation. That is
enough to diagnose the four failures that actually happen: irrelevant chunks, the
right chunk ranked too low, an over-aggressive similarity threshold, and
fabricated citation markers. Evaluation runs are traced too, so a metric
regression can be opened and read rather than guessed at.

Tracing is optional. With no key the decorators become no-ops.

## 10. Document ingestion

Sources live in `documents/source_registry.yaml` — the only place a URL appears
in this project. Each entry carries institution, document type, title, authority
(primary/secondary) and an enable flag.

The pipeline is `fetch → extract → clean → detect change → version → chunk →
embed → store`, and its most important property is how much work it *skips*.
Three change gates, cheapest first:

1. **Conditional HTTP.** Stored ETag / Last-Modified. A 304 costs one round trip
   and no download.
2. **Byte hash** against the newest stored version, for servers that ignore
   conditional requests.
3. **Text hash** after extraction and cleaning. This is the one that matters for
   bank product pages: their bytes change on every request (build ids, tokens)
   while the policy wording is identical. Without this gate a nightly poll would
   re-embed the entire corpus every night.

Cleaning removes running headers and footers — lines appearing on more than 60%
of pages — page numbers, and web boilerplate. This is a retrieval-quality step,
not cosmetics: repeated furniture is the most frequent text in a document, it
embeds well, and it crowds real clauses out of the top-k.

Every chunk is stored with source, URL, institution, title, version number,
version id, effective date, page number where available, chunk id, ingestion
timestamp, chunking strategy, and embedding model.

## 11. Document versioning

- Versions are **appended, never overwritten**. A decision made under version 1
  must still be explainable after version 3 ships.
- Version identity is a **text hash**. Re-fetching an unchanged policy a hundred
  times produces one version.
- **Active is derived, not set.** The active version is the newest whose
  effective date has arrived. A policy published early is stored and retrievable
  by explicit request, and correctly does not answer "what is the rule today".
- Retrieval filters on one metadata key, `version_id`, so restricting to active
  versions is a single `$in` clause rather than a nested filter over pairs.
- Historical lookup is explicit: `version_numbers_by_source={"...": 1}`.

Try it: in the Streamlit "Ask a question" view, pin the policy to version 1 and
ask for the minimum CIBIL score. You get 700. Unpin it and you get 725.

## 12. Chunking strategies

Five strategies are configured in `config/chunking.yaml`: `fixed_500_50` (the
control), `recursive_500_50`, `recursive_800_100` (current default),
`recursive_1000_150`, and `semantic`. Each gets its own ChromaDB collection.

The default is a **starting point, not a conclusion**. See
[`docs/chunking_strategy.md`](docs/chunking_strategy.md) for the reasoning and
the results table, and run `make experiment-chunking` to fill it in with numbers
from your machine.

Chunking happens per page, so no chunk spans a page boundary and every citation
can carry an exact page number. That costs a little recall on rules that straddle
a page break and buys verifiability — the right trade for a lending system.

## 13. Retrieval strategy

Configurable `top_k` (default 5), `min_similarity` (0.25), `max_chunks_per_source`
(3), `max_context_characters` (7000), and active-version restriction. The
retriever over-fetches `3 × top_k` and then post-filters, because filters can only
remove results and asking for exactly `top_k` would quietly return fewer.

Ties are broken by chunk id, so the prompt sees the same context in the same order
on every run. Without that, reproducibility fails for reasons that have nothing to
do with the model.

Higher top-k raises recall but adds irrelevant context, prompt size, latency and
cost — and beyond a point *lowers* answer quality by diluting the real clause.
Which value wins is measured, not assumed: `make experiment-topk`.

## 14. Golden dataset

`evaluation/golden_dataset.json` — ten questions with exactly knowable ground
truth, graded against the committed Meridian policy fixture rather than a live
bank website. A live page changes without notice, so an expected answer written
from one would be wrong within weeks and the suite would be measuring the
internet.

- **Category A — deterministic questions.** Specific figures stated in the
  policy: minimum CIBIL score, LTV slabs, FOIR ceiling and its exception,
  tenure, age at maturity, processing fee and cap, sanction lapse period.
- **Category B — same-question reproducibility.** Every case is executed
  `repeat_runs` times in one evaluation, and the harness reports whether the
  answer, the retrieved chunk ordering, the citations and the cited version were
  identical. Consistency is measured on all ten questions rather than a separate
  set, because consistency on a question whose correctness is unknown proves
  nothing.
- One case is a **historical lookup** graded against superseded version 1. It is
  what proves retaining old versions is functional rather than decorative.
- Cases carry `forbidden_answer_patterns`, which catch an answer that quotes a
  superseded figure alongside the current one and would otherwise score correct.

## 15. RAG metrics

Every metric is implemented in `evaluation/metrics.py` with a docstring saying
what question it answers and why it matters here. Full write-up in
[`docs/rag_metrics.md`](docs/rag_metrics.md).

- **Retrieval:** context precision, context recall, MRR, NDCG.
- **Generation:** faithfulness, answer relevancy. Both have a cheap deterministic
  lexical proxy that always runs, and an opt-in LLM-judge implementation
  (`--judge`). The proxy is labelled a proxy in every report.
- **System:** retrieval latency, end-to-end latency (mean and p95), context
  characters, prompt and completion tokens.
- **Business:** correct answer rate, citation correctness, **version
  correctness**, grounded rate, and its complement the fabricated-citation rate.

Version correctness is the metric the whole versioning design exists to move, and
it is strict: one citation of a superseded version alongside a correct one still
misleads a reader.

## 16. Evaluation methodology

```bash
python evaluation/run_evaluation.py
```

The harness checks a precondition first — the dataset states which policy version
must be active — and refuses to run against the wrong corpus state rather than
producing a page of red that only means "you forgot to ingest". It executes each
case through the same `answer_question` path the API exposes, scores it, measures
consistency across repeats, and writes a JSON result plus a Markdown summary to
`evaluation/experiment_results/`.

Every result file records the full configuration fingerprint: model, seed,
temperature, embedding model, strategy, top-k, thresholds, prompt version. A
number in a table can always be traced back to the run that produced it.

## 17. Optimization methodology

```
baseline → run golden dataset → inspect LangSmith traces → identify the failure
        → change ONE variable → re-run → compare → keep or reject → log it
```

Variables worth sweeping: chunk size, overlap, chunking strategy, embedding
model, top-k, similarity threshold, prompt, context formatting, metadata filters,
reranking. `evaluation/experiments.py` sweeps chunking and top-k; anything else
is one `HLR__` environment override away, which keeps the repository clean during
an experiment.

Decisions and their evidence go in [`docs/experiment_log.md`](docs/experiment_log.md).

## 18. How to run locally

See [`RUNBOOK.md`](RUNBOOK.md). The short version:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m pytest                                  # offline, no Ollama needed
ollama serve & ollama pull llama3.1:8b && ollama pull nomic-embed-text
python scripts/ingest.py
python run_backend.py                             # :8000
streamlit run frontend/streamlit_app.py           # :8501
python evaluation/run_evaluation.py
```

## 19. Environment variables

Secrets and machine-specific endpoints only; all behaviour lives in
`config/settings.yaml`. See `.env.example`.

| Variable | Purpose |
| --- | --- |
| `LANGSMITH_TRACING` | `true` to trace |
| `LANGSMITH_API_KEY` | LangSmith key. Absent ⇒ tracing is a no-op |
| `LANGSMITH_PROJECT` | Project name for traces |
| `OLLAMA_BASE_URL` | Ollama endpoint |
| `HOME_LOAN_API_URL` | Backend URL used by Streamlit |
| `HLR__SECTION__KEY` | Override any settings value, e.g. `HLR__RETRIEVAL__TOP_K=8` |

## 20. API endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/health` | Configuration, index size, explicit warnings |
| `POST` | `/api/v1/loan-assessment` | Assess an application |
| `POST` | `/api/v1/policy-question` | Ask the corpus; supports version pinning |
| `GET` | `/api/v1/documents/status` | Sources, active versions, full history |
| `POST` | `/api/v1/documents/refresh` | Run ingestion now |

Interactive docs at `/docs`. Errors return `{"error": "...", "detail": "..."}`;
`503` means a dependency is not ready (empty index, Ollama down), `502` a source
fetch failure, `422` invalid input.

## 21. Testing

```bash
python -m pytest          # 86 tests, fully offline
```

Covered: configuration validation, chunking (all strategies, size and overlap
properties, no text loss), extraction and cleaning, fetch security (scheme
allow-list, path containment) and change detection, version election including
future-dated policies and cross-process reload, ChromaDB persistence and metadata
filtering, historical retrieval, retrieval determinism, all eligibility rules and
their arithmetic, citation validation and the insufficient-information guard, the
HTTP contract including the empty-index 503, and the evaluation metrics and
harness end to end.

Integration coverage runs the real pipeline over the real policy fixture with
offline embedding and generation providers.

## 22. Known limitations

- **Determinism is conditional.** Fixing `temperature=0` and a seed does not make
  an LLM mathematically deterministic; a different quantisation, different
  hardware or a different context can still change the output. What is guaranteed
  is narrower: same model build + same retrieved context in the same order + same
  prompt ⇒ reproducible output. The golden dataset measures whether that holds
  rather than asserting it.
- **Live sources may 403.** RBI and bank sites often block non-browser clients.
  Handled per source; evaluation never depends on them.
- **Scanned PDFs are not read.** No OCR. Such a source fails loudly with a
  message saying so.
- **The lexical faithfulness proxy** cannot detect a claim that reuses the
  context's vocabulary to say something the context does not. Use `--judge`.
- **The Meridian policy is fictitious.** It exists so the golden dataset has
  verifiable ground truth. It is not lending advice.
- **The top-k sweep is flat on a single-source corpus** because the per-source cap
  binds first. Enable more sources for a meaningful sweep.
- **No authentication, no rate limiting, no PII storage policy.** Local
  demonstration system.

## 23. Future improvements

- A cross-encoder reranker over the over-fetched candidates, evaluated against
  the golden dataset before adoption.
- Hybrid retrieval (BM25 + dense). Policy questions are full of exact figures,
  where lexical matching is strong and dense retrieval is weak.
- A structured extraction pass that pulls thresholds out of policy text into a
  table, so the rule engine's configuration can be *derived* from documents and
  drift between the two becomes detectable.
- Per-institution rule profiles, so one application can be assessed against
  several lenders' policies at once.
- Ingestion as a queue-backed job with per-source scheduling instead of a single
  poll interval.
- Answer caching keyed on `(question, active version ids, configuration
  fingerprint)` — safe precisely because the key contains the knowledge state.

---

## Repository layout

```
app/
  api/          routes, schemas, dependency wiring, exception mapping
  core/         config, logging, exceptions, LangSmith tracing
  models/       applicant, assessment, document and chunk contracts
  ingestion/    registry, fetcher, extractor, cleaner, versioning,
                chunking, embeddings, vector_store, pipeline
  rag/          query_builder, retriever, context_builder, prompts,
                llm, generator, pipeline
  rules/        eligibility rule engine  ← the decision lives here
  services/     container (composition root), assessment, documents
  utils/        hashing, text, timing, atomic JSON store
  watcher/      long-lived document update process
config/         settings.yaml, chunking.yaml
documents/      source_registry.yaml, local_policies/{current,versions}
evaluation/     golden_dataset.json, metrics, runner, report,
                run_evaluation.py, experiments.py, experiment_results/
frontend/       streamlit_app.py
scripts/        check_environment, ingest, run_watcher,
                simulate_policy_update, reset_data
tests/          86 offline tests
docs/           architecture, chunking_strategy, rag_metrics,
                experiment_log, decisions
```

## Disclaimer

A demonstration system. It does not make binding credit decisions. Meridian
Housing Finance Limited is fictitious, and the thresholds in
`config/settings.yaml` are illustrative defaults, not any real lender's policy.
