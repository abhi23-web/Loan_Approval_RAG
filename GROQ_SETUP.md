# Setup — Groq + Llama, no Ollama

This build uses **Llama 3.3 70B hosted on Groq** for generation and a **local
ONNX model** for embeddings. Nothing needs Ollama. Nothing needs a GPU. The only
paid-looking thing, the Groq key, is free and needs no card.

Total download: about 350 MB, versus ~5 GB for the Ollama build.

---

## Why embeddings are still local

Groq has no embedding endpoint, so embeddings have to run somewhere else. That
turns out to be the better arrangement regardless: **generation sees a handful of
retrieved chunks, but embedding sees every document in full.** Keeping the
embedding half local means your policy corpus is never uploaded anywhere, and it
costs nothing per query.

The default is `fastembed` (`BAAI/bge-small-en-v1.5`), which runs on ONNX Runtime
— tens of megabytes and no PyTorch. `sentence-transformers` is available as an
alternative if you prefer it, but it drags in PyTorch, which is why it isn't the
default.

---

## Windows (PowerShell)

```powershell
# 1. Virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Dependencies
pip install -r requirements.txt

# 3. Config
Copy-Item .env.example .env
```

If `Activate.ps1` is blocked:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

---

## Get your Groq key

1. Go to <https://console.groq.com/keys> and sign in.
2. Create an API key — it looks like `gsk_...`.
3. **Copy it immediately.** The console shows it once.
4. Open `.env` and set:

```
GROQ_API_KEY=gsk_your_actual_key_here
```

LangSmith is optional. If you have a key, add it too; if not, everything runs the
same, just untraced.

---

## Verify before you spend time

```powershell
python scripts/check_environment.py
```

You want:

```
[  ok  ] configuration — strategy='recursive_800_100', top_k=5
[  ok  ] llm — llama-3.3-70b-versatile via https://api.groq.com/openai/v1 (key gsk_abc…wxyz)
[  ok  ] embeddings — fastembed 'BAAI/bge-small-en-v1.5' (local)
[ warn ] langsmith — no LANGSMITH_API_KEY
[ warn ] chromadb index — empty
```

The two warnings are expected at this stage. The embeddings line takes a moment
the first time — that's the model downloading.

Note this check does **not** call the model. A pre-flight that costs a token and
a second of latency stops being run, and a check nobody runs is worse than none.

---

## Load the documents

```powershell
python scripts/simulate_policy_update.py --version 1
python scripts/ingest.py
python scripts/simulate_policy_update.py --version 2
python scripts/ingest.py
python scripts/simulate_policy_update.py --version 3
python scripts/ingest.py
```

Or on Windows with the helper script: `.\run.ps1 ingest -Real`

All three versions matter. The golden dataset grades against version 3 being
active while versions 1 and 2 remain queryable — that's the version-history
feature under test, so ingesting only version 3 is not equivalent.

---

## Run it

Three windows, each with the venv activated:

```powershell
python run_backend.py                              # :8000  API
python -m streamlit run frontend/streamlit_app.py  # :8501  UI
python scripts/run_watcher.py                      #        update watcher
```

Open <http://localhost:8501>, fill in the form, submit.

Run the golden dataset:

```powershell
python evaluation/run_evaluation.py
```

---

## Switching models and providers

Everything is in `config/settings.yaml`. No code changes.

**Faster and cheaper**, at some cost in reasoning quality:

```yaml
llm:
  model: llama-3.1-8b-instant
```

**A different provider** — the system speaks the OpenAI `/chat/completions`
format, so these all work by changing two lines:

| Provider | `base_url` | Key variable |
|---|---|---|
| Groq (default) | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` |
| OpenRouter | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| LM Studio (local) | `http://localhost:1234/v1` | none needed |
| Ollama | set `provider: ollama`, `base_url: http://localhost:11434` | none |

**Heavier but more standard embeddings:**

```yaml
embeddings:
  provider: huggingface
  model: sentence-transformers/all-MiniLM-L6-v2
```

Then: `pip install torch --index-url https://download.pytorch.org/whl/cpu` and
`pip install sentence-transformers`.

### The rule you cannot break

**Vectors written by one embedding model cannot be queried with another.**

Change `embeddings.model` or `embeddings.provider` and you must re-ingest:

```powershell
python scripts/ingest.py --force
```

ChromaDB will not stop you and will not warn you. It returns plausible-looking
results that are quietly wrong — much harder to notice than an error.

Changing the *LLM* needs no re-ingest. Only embeddings.

---

## When something breaks

**`GROQ_API_KEY is not set`** — you edited `.env.example` instead of `.env`, or
you're in a shell that hasn't picked up the file. `check_environment.py` will
confirm.

**`HTTP 401`** — the key is wrong, expired, or belongs to a different provider.
Groq keys start with `gsk_`.

**`HTTP 429 rate limit`** — Groq's free tier is metered per minute. Wait, or
switch to `llama-3.1-8b-instant`.

**`HTTP 404 model not found`** — Groq retires models. Check
<https://console.groq.com/docs/models> and update `llm.model`.

**`fastembed could not load ...`** — no internet, or a proxy blocking
huggingface.co on the first download. Set `embeddings.provider: deterministic` to
prove the rest of the system works, then come back to it.

**Answers are nonsense after changing the embedding model** — you skipped
`--force`.

**Testing without any of this:** the whole system runs offline with stub
providers. 86 tests use them and need no key, no network and no downloads:

```powershell
python -m pytest
$env:HLR__LLM__PROVIDER = "deterministic"
$env:HLR__EMBEDDINGS__PROVIDER = "deterministic"
```

---

## What is and isn't proven

Verified by running it: 86 tests pass, ingestion promotes v1 → v2 → v3 with
superseded versions still queryable, the API returns real rule-engine decisions
with citations, answer consistency and retrieval consistency are both 100%, and
`ruff check` is clean.

Not verified: **no real Groq call and no real embedding model run has happened**
— the build sandbox had no key and its proxy blocks huggingface.co. The model id
`llama-3.3-70b-versatile` comes from Groq's documentation, not from calling it.
Everything measured so far used the offline stubs, so `docs/chunking_strategy.md`
and `docs/experiment_log.md` still read "not yet measured" by design. Running
this on your machine with a real key is what fills them in.

One more thing worth deciding deliberately: with Groq, applicant details in the
prompt do leave your machine. The architecture keeps that contained — the rule
engine makes the decision locally and the model only explains it — but if that
matters for your use case, `provider: ollama` puts generation back on your own
hardware with a one-line change.
