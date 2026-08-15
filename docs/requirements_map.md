# Requirements → where each one is satisfied

Every requirement, the file that implements it, and how to see it working.

| # | Requirement | Implemented in | Verify with |
|---|---|---|---|
| 1 | ChromaDB as the vector DB | `app/ingestion/vector_store.py` | `make check` |
| 2 | Chunking, then feed into vector DB | `app/ingestion/chunking.py` → `vector_store.py` | `make ingest` |
| 3 | Streamlit form feeds the data | `frontend/streamlit_app.py` | `make ui` |
| 4 | FastAPI + uvicorn always listening | `app/api/main.py`, `run_backend.py` | `make api` |
| 5 | Frontend calls backend, response renders below the form | `streamlit_app.py` → `POST /api/v1/loan-assessment` | submit the form |
| 6 | Live process checking for updated data | `app/watcher/monitor.py` | `make watcher` |
| 7 | Document versions 1 / 2 / 3, v1 first, new versions on rerun | `app/ingestion/versioning.py` | `make versions` |
| 8 | Data stored locally on the PC | `data/chroma/`, `data/metadata/` | `ls data/` |
| 9 | LangSmith | `app/core/tracing.py` | set `LANGSMITH_API_KEY`, run a query |
| 10 | RAG metrics | `evaluation/metrics.py`, `docs/rag_metrics.md` | `make eval` |
| 11 | Different chunking strategies | `config/chunking.yaml` (5 strategies) | `make experiment-chunking` |
| 12 | A way to explain the chunking choice | `docs/chunking_strategy.md` | read it |
| 13 | LangSmith data used to optimise | `docs/rag_metrics.md` § "Optimising from traces" | — |
| 14 | Why this vector DB | `docs/decisions.md` § ADR-002 | read it |
| 15 | Golden dataset, 10 questions, absolute answers | `evaluation/golden_dataset.json` | `make eval` |
| 16 | Same question → same response | temperature 0 + fixed seed + fixed retrieval order | `make eval` (3 repeats) |
| 17 | Reference to source | `app/rag/generator.py` citation validator | any answer's `citations[]` |
| 18 | Coding standards | `docs/coding_standards.md` | `ruff check app/` |

---

## The three processes, and how they relate

The requirement "1 — ChromaDB creation, 2 — Streamlit, 3 — a live process" describes
three things that run independently. They share state through the ChromaDB
directory and the metadata store on disk, not through memory, which is why each
can be restarted without the others noticing.

```
  ┌─────────────────┐         ┌──────────────────┐        ┌────────────────┐
  │ scripts/        │ writes  │  data/chroma/    │ reads  │ FastAPI        │
  │ ingest.py       ├────────►│  data/metadata/  │◄───────┤ :8000          │
  │ (1: build)      │         │  (local disk)    │        │ (always up)    │
  └─────────────────┘         └────────▲─────────┘        └───────▲────────┘
                                       │ writes                   │ HTTP
  ┌─────────────────┐                  │                  ┌───────┴────────┐
  │ watcher         ├──────────────────┘                  │ Streamlit      │
  │ (3: live poll)  │  re-ingests on change               │ :8501 (2: form)│
  └─────────────────┘                                     └────────────────┘
```

The watcher is what makes "version 1 first, new versions on later runs" true
without anyone re-running a script by hand. It polls each source on an interval,
and when a document's content hash changes it re-ingests, marks the previous
version superseded, and elects the new one active by effective date.

---

## Why the same question gives the same answer

Four things have to hold at once. Any one of them missing and repeat runs drift:

1. **`temperature: 0.0` and `top_p: 1.0`** — no sampling randomness.
2. **`seed: 42`** — pins whatever randomness remains.
3. **A fixed retrieval order** — chunks are sorted by similarity then by a stable
   tiebreaker, so equal scores never reorder between runs.
4. **A fixed context assembly order** — `context_builder.py` emits sources in
   retrieval order, so `[S1]` means the same chunk each time.

`make eval` runs all 10 golden questions three times each and reports whether
the answers matched. That is the check, not the claim.

One honest limit: on a *hosted* endpoint (`provider: openai`) the `seed`
parameter is best-effort — some providers honour it, some ignore it, and the
deployment can change underneath you. Full reproducibility is a property of the
local Ollama path. What the code guarantees in both cases is that it contributes
no randomness of its own.

---

## Why the decision cannot be hallucinated

The architectural point worth understanding before reading the code:

**The rule engine decides. The model only explains.**

`app/rules/eligibility.py` is ordinary deterministic Python — arithmetic and
comparisons against thresholds in `config/settings.yaml`. It produces a
`RuleAssessment`. The model never sees that as something it can change; it
receives it as a fact to explain, along with the retrieved policy text.

`RuleAssessment` and `GroundedExplanation` stay separate types all the way to
the UI. So the worst a hallucinating model can do is produce a bad *explanation*
of a *correct* decision — a visible bug, rather than a wrong loan outcome that
looks fine.
