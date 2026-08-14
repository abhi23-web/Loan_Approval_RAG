# Decision record

Judgement calls made while building this, with the reasoning. Where the brief was
ambiguous, the choice is recorded here rather than left implicit.

---

### D-01 — The rule engine decides, the model explains

Deterministic eligibility in `app/rules/eligibility.py`; the LLM receives the
decision as an input it may not contradict. A model that can be talked into a
different answer by rephrasing has no business approving credit, and a decision
that cannot be recomputed from its inputs cannot be defended.

**Cost:** thresholds must be maintained in `config/settings.yaml` alongside the
policy documents, and the two can drift. Mitigated by surfacing both in the
response so a mismatch is visible; a future structured-extraction pass could
derive one from the other.

---

### D-02 — LangSmith SDK used directly, no LangChain

The trace tree then mirrors this application's own functions — query building,
retrieval, context assembly, generation — instead of a framework's internal
runnables, so a bad answer maps to a file you can open. It also keeps a project
whose purpose is comparing chunkers from simultaneously comparing framework
versions.

**Cost:** the chunkers, the retriever and the Ollama clients are written here.
About 400 lines, all inspectable, all tested.

---

### D-03 — A fictitious lender's policy in the corpus

The golden dataset needs expected answers that are exactly knowable and stable.
Grading against a live bank page would mean the expected answers rot within weeks
and the suite would measure the internet rather than the pipeline.

`documents/local_policies/versions/` holds three versions of the Meridian policy,
committed to the repository. Real RBI and HDFC Bank sources are registered
alongside them and are used by the live system; the dataset simply does not grade
against them. The document is labelled as illustrative in its own first paragraph.

---

### D-04 — Version filtering by a single metadata key

Each chunk carries `version_id = "{source}::v{n}"`. Retrieval restricts with one
`$in` clause. The alternative — storing an `is_active` boolean on every chunk —
would require rewriting the metadata of every chunk of the previous version on
each update, which is both slower and a chance to corrupt history.

**Consequence:** the active-version set is resolved at query time from the version
store, so the store must be readable by the query path. That is why the API
reloads it on an mtime change.

---

### D-05 — Active version is derived from effective dates

The active version is the newest whose effective date has arrived, not simply the
newest ingested. A policy published in advance is stored, is retrievable by
explicit request, and does not answer "what is the rule today".

**Consequence:** a corpus of only future-dated policies has no correct answer.
The store activates the earliest and logs a warning rather than silently having
none.

---

### D-06 — Chunking per page

No chunk spans a page boundary, so every citation from a PDF carries an exact
page number a reviewer can verify in seconds.

**Cost:** a rule split across a page break is split across chunks, costing some
recall. For a lending system, verifiability wins.

---

### D-07 — Change detection by text hash, not byte hash

Bank product pages change bytes on every request — build ids, CSRF tokens — while
the policy wording is unchanged. Byte-level detection alone would re-embed the
corpus on every poll. Extraction and cleaning run first, then the text hash
decides. Byte hash and HTTP 304 remain as cheaper earlier gates.

---

### D-08 — A deterministic offline provider pair, clearly marked

`DeterministicEmbeddingProvider` is a hashing vectoriser with real lexical
similarity, so retrieval *ordering* can be asserted in CI with no model server.
`DeterministicLLMProvider` returns a fixed citation-shaped answer so the citation
validator and the HTTP contract are testable.

**Guard against misuse:** `/health` warns when either is active, and the
evaluation runner writes a note into every result file produced with them.
Metrics from a stub run are never presentable as quality evidence.

---

### D-09 — Category B read as reproducibility over all ten questions

The brief asks for exactly ten questions and separately for ten reproducibility
questions. Read literally those conflict. The resolution: ten questions, each
executed `repeat_runs` times, with answer, retrieval, citation and version
consistency measured across the repeats.

Consistency on a question whose correctness is unknown proves nothing, so
measuring it on the same ten graded questions is strictly more informative than a
separate set. Documented inside `golden_dataset.json` itself.

---

### D-10 — Faithfulness has a proxy and a judge

The lexical proxy is deterministic, free, and always runs. The LLM judge is
opt-in via `--judge`. Every report labels the proxy as a proxy, and an unparsable
judge response yields `null` rather than zero — "the judge broke" and "the answer
was unfaithful" must not look the same in a results file.

---

### D-11 — Synchronous `/documents/refresh`

Ingestion is idempotent and change-gated, so the common case returns in well under
a second, and a caller who triggered a refresh should be told what happened rather
than handed a job id. Continuous background refresh is the watcher's job.

**When this breaks:** a first-time ingest of many large PDFs can exceed a request
timeout. At that point the endpoint should return a job id — noted in the README's
future work.

---

### D-12 — An empty index is a 503, not "insufficient information"

An un-ingested system would otherwise answer every applicant with a polite
refusal that reads like a policy outcome and hides an operational fault.
`KnowledgeBaseEmptyError` maps to 503 with the exact command to fix it.

---

### D-13 — The registry is never written back

`last_checked` in `source_registry.yaml` is a human note. Programmatic writes
would destroy the file's comments, which are documentation. The authoritative
timestamp lives in `data/metadata/version_store.json`.

---

### D-14 — Over-fetch then post-filter

The retriever asks ChromaDB for `3 × top_k` and then applies the similarity
threshold and the per-source cap. Filters can only remove results, so requesting
exactly `top_k` would silently return fewer chunks than configured — a subtle
recall loss that would look like a chunking problem.

---

### D-15 — Deterministic tie-breaking on chunk id

Equal similarities are ordered by chunk id. Without it, retrieval order could vary
between runs and reproducibility would fail for reasons unrelated to the model,
making the consistency metric useless as a signal.
