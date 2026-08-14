# Runbook — run this project block by block

Fourteen blocks. Run them in order the first time. Each block states what it does,
what success looks like, and what to do when it fails. Nothing later depends on a
block you skipped without reading the note attached to it.

Blocks 1–7 need no model server and no API key. Block 8 onwards needs Ollama.

---

## Block 0 — What you need before you start

| Requirement | Why | Check |
| --- | --- | --- |
| Python 3.11 or newer | The code uses `X \| Y` type syntax and `datetime.UTC` | `python --version` |
| ~6 GB free disk | Ollama models are 4–5 GB | `df -h .` |
| Ollama (block 8) | Local LLM and embeddings | `ollama --version` |
| A LangSmith key (optional) | Tracing and evaluation views | https://smith.langchain.com |

Nothing here talks to a paid API. The only network calls are model downloads
(once) and fetching public policy documents.

---

## Block 1 — Create an isolated environment

```bash
cd home_loan_rag
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Success:** `pip list | grep chromadb` prints `chromadb 1.5.9`.

**If it fails:** a `chromadb` build error almost always means Python is older than
3.11 or the platform has no wheel. Check `python --version` first.

---

## Block 2 — Create your `.env`

```bash
cp .env.example .env
```

Open `.env`. You can leave everything as-is for now — the system runs without a
LangSmith key, it just produces no traces. Fill `LANGSMITH_API_KEY` when you get
to block 12.

**Never commit `.env`.** It is already in `.gitignore`; verify with
`git check-ignore -v .env`.

---

## Block 3 — Pre-flight check

```bash
python scripts/check_environment.py
```

**Success:** you see four lines. At this stage expect:

```
[  ok  ] configuration — strategy='recursive_800_100', top_k=5
[ FAIL ] ollama server — http://localhost:11434 unreachable
[ warn ] langsmith — no LANGSMITH_API_KEY
[ warn ] chromadb index — empty for 'recursive_800_100'
```

The Ollama FAIL is expected until block 8. The other two are warnings, not errors.

**If configuration FAILs:** the message names the exact key in
`config/settings.yaml` that is wrong. Configuration is validated strictly, so a
typo is reported rather than silently defaulted.

---

## Block 4 — Run the test suite

Run this before anything else touches your machine. The whole suite is offline:
it uses a deterministic hashing embedder and a stub LLM, so it needs no Ollama,
no network and no keys.

```bash
python -m pytest
```

**Success:** `86 passed`.

**What it just proved:** chunking, cleaning, version election, ChromaDB
persistence and metadata filtering, the eligibility rules, citation validation,
the HTTP contract, and the evaluation harness — all working.

**If it fails:** the failure name tells you the subsystem. Nothing later in this
runbook will work correctly if this block is red.

---

## Block 5 — Ingest the controlled policy (version 1)

The repository ships a fictitious lender's policy in three versions. Version 1 is
the working copy. Ingesting it exercises the entire pipeline with no network.

```bash
export HLR__LLM__PROVIDER=deterministic
export HLR__EMBEDDINGS__PROVIDER=deterministic     # Windows: set HLR__...=deterministic

python scripts/simulate_policy_update.py --version 1
python scripts/ingest.py --source meridian_home_loan_policy
```

**Success:**

```
meridian_home_loan_policy    indexed    v1    recursive_800_100=6
1 indexed, 0 unchanged, 0 skipped, 0 failed
```

Those two `HLR__` variables force the offline providers so this block works
before Ollama exists. Unset them at block 8. Any settings key can be overridden
this way — `HLR__SECTION__KEY`.

**Now run it again:**

```bash
python scripts/ingest.py --source meridian_home_loan_policy
```

**Success:** `0 indexed, 1 unchanged`. That is the change-detection working; a
poll over an unchanged document costs no embedding at all.

---

## Block 6 — Watch versioning happen

```bash
python scripts/simulate_policy_update.py --version 2
python scripts/ingest.py --source meridian_home_loan_policy

python scripts/simulate_policy_update.py --version 3
python scripts/ingest.py --source meridian_home_loan_policy

python scripts/simulate_policy_update.py --status
```

**Success:**

```
Version  Active  Effective    Declared  Chunks
1        no      2023-04-01   1         recursive_800_100=6
2        no      2024-07-01   2         recursive_800_100=7
3        yes     2026-01-01   3         recursive_800_100=7
```

**What this proves:** version 1 and 2 are still indexed and still queryable — they
were superseded, not deleted. Version 3 is active because its effective date has
arrived. The script only copies a file; the pipeline discovered the change on its
own through the ordinary path.

---

## Block 7 — Try the live regulatory sources

```bash
python scripts/ingest.py
```

**Success:** `meridian_home_loan_policy unchanged` plus whatever the network
allows for the RBI and HDFC Bank entries.

**Expect some `failed` lines.** Public bank and regulator sites frequently return
`403 Forbidden` to non-browser clients, and behaviour differs by network. That is
handled, not fatal: one unreachable source never aborts the run, and the golden
dataset deliberately grades against the local corpus so evaluation never depends
on a website being up.

To silence a source that your network always blocks, set `enabled: false` on it
in `documents/source_registry.yaml`. To add your own source, add an entry to that
file — no code changes.

---

## Block 8 — Install Ollama and pull the models

Everything above ran without a model. From here you need one.

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh
# or download from https://ollama.com/download

ollama serve                       # leave this running in its own terminal
```

In a second terminal:

```bash
ollama pull llama3.1:8b            # generation, ~4.7 GB
ollama pull nomic-embed-text       # embeddings, ~275 MB
```

**Success:**

```bash
unset HLR__LLM__PROVIDER HLR__EMBEDDINGS__PROVIDER
python scripts/check_environment.py
```

```
[  ok  ] ollama server — http://localhost:11434, 2 model(s) installed
[  ok  ] llm model — llama3.1:8b
[  ok  ] embeddings model — nomic-embed-text
```

**Smaller machine?** `ollama pull qwen2.5:3b` and set `llm.model: qwen2.5:3b` in
`config/settings.yaml`. Answers are weaker; everything else is identical.

---

## Block 9 — Re-index with real embeddings

The vectors from block 5 came from the offline hashing double. They are not
semantic and must be replaced.

```bash
python scripts/ingest.py --force
python scripts/check_environment.py
```

`--force` re-chunks and re-embeds even though the documents have not changed.
That flag exists for exactly this case: the documents are the same, the embedding
model is not.

**Success:** `chromadb index — N chunk(s)` and the embedding model recorded on the
chunks is now `nomic-embed-text`. This is the slowest block in the runbook —
embedding runs on your CPU or GPU.

---

## Block 10 — Start the backend

```bash
python run_backend.py --reload
```

**Success:** open http://localhost:8000/docs — the full OpenAPI page.

In another terminal:

```bash
curl -s localhost:8000/api/v1/health | python -m json.tool
```

`"status": "ok"` with an empty `warnings` list means everything is real: real
models, a populated index, tracing on if you set a key.

```bash
curl -s -X POST localhost:8000/api/v1/loan-assessment \
  -H 'content-type: application/json' \
  -d '{"application":{
        "applicant_name":"Asha Menon","age_years":34,"employment_type":"salaried",
        "employment_experience_months":72,"monthly_income_inr":150000,
        "credit_score":760,"existing_monthly_emi_inr":18000,
        "number_of_existing_loans":1,"loan_amount_required_inr":6000000,
        "property_value_inr":8000000,"loan_tenure_years":20}}' | python -m json.tool
```

**Success:** a decision, a list of rule checks with the arithmetic behind each,
an explanation, and citations that each carry a document version.

**If you get HTTP 503 `KnowledgeBaseEmptyError`:** you skipped block 5 or 9. That
is intentional — an un-ingested system refuses to answer rather than politely
saying "insufficient information" and hiding an operational fault.

---

## Block 11 — Start the frontend

Leave the backend running. In a new terminal:

```bash
source .venv/bin/activate
streamlit run frontend/streamlit_app.py
```

**Success:** http://localhost:8501 opens. The sidebar shows the backend health.
Three views:

- **Assess an application** — the loan form. Submit it and read the decision, the
  rule checks, the sources with their versions, and the retrieval diagnostics.
- **Policy corpus** — every registered source, its active version, and the full
  version history. The refresh button runs ingestion.
- **Ask a question** — the same retrieval path directly. Set "Pin Meridian policy
  to version" to `1` and ask *"What is the minimum CIBIL score?"* — you get 700.
  Set it back to `0` and you get 725. That is version filtering, visible.

**If the sidebar shows a connection error:** the backend is not running, or it is
on a different port. Set `HOME_LOAN_API_URL` in `.env` and restart Streamlit.

---

## Block 12 — Turn on LangSmith

```bash
# in .env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_your_key_here
LANGSMITH_PROJECT=home-loan-rag
```

Restart the backend, submit one application, then open
https://smith.langchain.com and select the `home-loan-rag` project.

**Success:** one trace per assessment, nested as

```
pipeline.assess_application
├── rag.retrieve          ← query, filter, chunks, similarities, what was dropped
└── rag.generate          ← prompt, answer, tokens, which citations were used
```

**What to look for first** — these are the four failures traces actually catch:

1. `rag.retrieve` returned chunks that have nothing to do with the question →
   chunking or embedding problem.
2. The right chunk is present but ranked 5th → a top-k or reranking problem.
3. `dropped_below_threshold` is large → `min_similarity` is too aggressive.
4. `invalid_markers` is non-empty on `rag.generate` → the model invented a
   citation; the system stripped it, but the prompt needs work.

**No key?** Everything still runs. Tracing degrades to a no-op.

---

## Block 13 — Run the golden dataset

The dataset grades against version 3 of the local policy, so make sure block 6
ran.

```bash
python evaluation/run_evaluation.py
```

**Success:**

```
correct answer rate : 90%
context precision   : 0.6xx
context recall      : 0.9xx
MRR                 : 0.9xx
version correctness : 100%
answer consistency  : 100%
```

Your numbers will differ — they depend on your model and machine. That is the
point: this is a measurement, not a claim.

```bash
python evaluation/run_evaluation.py --judge      # model-scored faithfulness
python evaluation/run_evaluation.py --repeat 5   # stronger consistency evidence
```

**If it refuses to start** with *"the corpus is not in the state this dataset
grades against"*, it is telling you which version is active and how to fix it.
Run block 6.

Results land in `evaluation/experiment_results/` as a JSON file and a Markdown
summary. Both record the exact configuration that produced them.

---

## Block 14 — Run the experiments

Every chunking strategy needs its own index first:

```bash
python scripts/ingest.py --all-strategies --force
```

This is slow. It embeds the corpus once per strategy, and the `semantic` strategy
additionally embeds every sentence to find its boundaries.

```bash
python evaluation/experiments.py chunking
python evaluation/experiments.py top-k --values 3 5 8 10
```

**Success:** a comparison table on stdout and in
`evaluation/experiment_results/comparison_chunking.md`.

**Read the result before believing it.** Two things regularly surprise people:

- With only the local policy enabled, `retrieval.max_chunks_per_source` (default
  3) binds before `top_k` does, so every top-k value returns three chunks and the
  sweep is flat. Enable more sources, or raise that cap, to make the sweep
  meaningful.
- The largest chunk size usually wins on recall and loses on precision and token
  cost. Which one matters is a decision, not a number — write it down in
  `docs/experiment_log.md`.

Then set the winner in `config/settings.yaml`:

```yaml
chunking:
  active_strategy: <winner>
retrieval:
  active_strategy: <winner>
```

and re-run block 13 to confirm the improvement held.

---

## Block 15 — Run the update watcher (optional)

```bash
python scripts/run_watcher.py --interval 120
```

**Success:** a cycle log every two minutes reporting no changes. Now, in another
terminal, run `python scripts/simulate_policy_update.py --version 2` and watch
the next cycle detect it and re-index. The running API picks up the new active
version without a restart.

Stop it with Ctrl-C — the current cycle finishes and saves before exiting.

---

## The five commands you will actually use daily

```bash
python -m pytest                          # is anything broken?
python scripts/ingest.py                  # refresh the corpus
python run_backend.py --reload            # API on :8000
streamlit run frontend/streamlit_app.py   # UI on :8501
python evaluation/run_evaluation.py       # did my change help or hurt?
```

`make help` lists the same things as Make targets.

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'app'` | Running from the wrong directory | `cd` to the repository root, or `pip install -e .` |
| `EmbeddingError: cannot reach Ollama` | `ollama serve` not running | Start it; confirm with `curl localhost:11434/api/tags` |
| `EmbeddingError: Ollama rejected model` | Model not pulled | `ollama pull nomic-embed-text` |
| HTTP 503 `KnowledgeBaseEmptyError` | Nothing ingested | `python scripts/ingest.py` |
| Every answer is "Insufficient information" | Retrieval returns nothing | Lower `retrieval.min_similarity`; check the index was built with the *same* embedding model you are querying with |
| Evaluation refuses to start | Corpus in the wrong version state | Follow the instructions in the error; run block 6 |
| Ingestion says `failed ... 403 Forbidden` | The site blocks automated clients | Expected; set `enabled: false` for that source |
| Chunk counts changed but answers did not | Wrong strategy is live | `retrieval.active_strategy` must match a strategy you actually ingested |
| First request after startup is slow | Model load | Normal; Ollama keeps the model warm afterwards |

---

## Resetting

```bash
python scripts/reset_data.py --dry-run   # see what would go
python scripts/reset_data.py --yes       # delete index + version history
python scripts/simulate_policy_update.py --version 1
python scripts/ingest.py
```

Deleting `data/metadata/version_store.json` discards the record of which policy
version answered which question. That is the audit trail — delete it knowingly.
