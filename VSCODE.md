# Running this project in VS Code

Start here if you've never run it before. Everything in Part 1 works with **no
Ollama, no API key and no network** — you can have the full app running in your
browser in about ten minutes.

---

## Part 1 — Get it running (offline)

### Step 1: Open the folder

`File → Open Folder…` → select the `home_loan_rag` folder itself, not its parent.

VS Code will offer to install recommended extensions. Say yes. Only two matter:
**Python** and **Pylance**.

> **How do I know I opened the right folder?** The Explorer sidebar should show
> `app`, `config`, `scripts`, `tests` and `RUNBOOK.md` at the top level. If you
> see a single `home_loan_rag` folder you need to open, you opened the parent.

### Step 2: Create the virtual environment

Open a terminal inside VS Code: **`Ctrl+``** (backtick), or `Terminal → New Terminal`.

```bash
python3 -m venv .venv
```

Windows:

```powershell
py -m venv .venv
```

This creates a private Python just for this project. It takes a few seconds and
produces a `.venv` folder. Don't commit it — it's already gitignored.

### Step 3: Tell VS Code to use it

This is the step that, when skipped, causes almost every "nothing works"
problem. VS Code will not find the venv reliably on its own.

1. `Ctrl+Shift+P` → type **Python: Select Interpreter** → Enter
2. Choose the one whose path contains `.venv` — it's usually labelled
   **`('.venv': venv)`** and sits at the top.

Check the bottom-right status bar: it should now read something like
`Python 3.11.x ('.venv': venv)`.

**Now close your terminal and open a new one** (`Ctrl+`` twice). The new one
will have the venv active — your prompt starts with `(.venv)`.

### Step 4: Install the dependencies

In the terminal with `(.venv)` in the prompt:

```bash
pip install -r requirements.txt
```

Two to three minutes. Confirm it landed:

```bash
pip list | grep chromadb
```

Expected: `chromadb  1.5.9`. On Windows use `pip list | findstr chromadb`.

### Step 5: Create your `.env`

```bash
cp .env.example .env
```

Windows: `copy .env.example .env`

Leave the contents alone for now. Nothing in Part 1 needs a key.

### Step 6: Run the tests

This proves the install worked before you touch anything else.

Click the **flask icon** in the left sidebar (Testing), then the ▶▶ button at
the top to run everything. Or from the terminal:

```bash
python -m pytest
```

**Expected: `86 passed`.** These tests are fully offline — they use a hashing
embedder and a stub LLM, so they need no Ollama and no network.

If the Testing panel says "no tests found", you skipped Step 3.

### Step 7: Load some data

The repo ships a fictitious lender's policy in three versions. Indexing version 1
exercises the whole pipeline with no network.

Press **F5**, choose **"1. Ingest — offline (no Ollama)"**.

Expected output:

```
meridian_home_loan_policy    indexed    v1    recursive_800_100=6
1 indexed, 0 unchanged, 0 skipped, 0 failed
```

Six chunks are now in ChromaDB under `data/chroma/`.

### Step 8: Run the app

`Ctrl+Shift+D` to open Run and Debug, pick **"API + UI (offline)"** from the
dropdown, press the green ▶.

That starts two things: the API on `http://localhost:8000` and the Streamlit UI
on `http://localhost:8501`. Your browser should open the UI automatically; if
not, `Ctrl+Click` the localhost link in the terminal.

Fill in the loan form and submit. You'll get a decision, the individual rule
checks behind it, and a cited explanation.

**One thing to understand about what you're seeing:** the *decision* is real —
it comes from the deterministic rule engine, which is the same code that would
run in production. The *explanation text* is a stub, because there's no model
running. It will read like placeholder text, and it is. Part 2 fixes that.

To stop everything: the red ■ in the debug toolbar.

---

## Part 2 — Real answers (with Ollama)

Part 1 gives you the system with a fake brain. This gives it a real one, still
entirely on your machine — no API keys, no cost, no data leaving your laptop.

**You'll need ~6 GB of free disk.**

### Step 1: Install Ollama

Download from <https://ollama.com/download> and install. Then in a terminal
**outside VS Code** (this one stays running):

```bash
ollama serve
```

### Step 2: Pull the models

In a different terminal:

```bash
ollama pull llama3.1:8b        # ~4.7 GB — the language model
ollama pull nomic-embed-text   # ~275 MB — the embedding model
```

Coffee break. This is the long part.

### Step 3: Confirm

Back in VS Code: **F5 → "4. Environment check"**.

You want the Ollama line to read `ok`. If it still says unreachable, `ollama
serve` isn't running in that other terminal.

### Step 4: Re-index with real embeddings

**This step is mandatory and skipping it is the single most common way to break
this project.**

The six chunks in ChromaDB were embedded by the offline hashing stub. Real
queries will now be embedded by `nomic-embed-text`. Vectors from two different
models are not comparable — ChromaDB will not warn you, it will just return
quietly wrong results.

**F5 → "7. Ingest (real Ollama)"**. That runs with `--force`, which re-embeds
everything from scratch.

### Step 5: Run it for real

**F5 → "8. API backend (real Ollama)"**, then start the UI separately
(**F5 → "3. Streamlit UI"**).

The first question takes 10–30 seconds — the model is loading into memory.
Subsequent ones are faster.

---

## What each Run configuration does

Press F5 and pick from the dropdown:

| # | Configuration | Needs Ollama? |
|---|---|---|
| 1 | Ingest — offline | no |
| 2 | API backend — offline | no |
| 3 | Streamlit UI | no |
| 4 | Environment check | no |
| 5 | Evaluation — offline | no |
| 6 | Show policy version history | no |
| 7 | Ingest (real Ollama) | yes |
| 8 | API backend (real Ollama) | yes |
| — | **API + UI (offline)** — starts 2 and 3 together | no |

---

## Reading the code

If you want to understand it rather than just run it, this is the order that
makes sense. Set a breakpoint (click left of a line number) and run config 2,
then submit a question from the UI — execution will stop and you can inspect
every variable.

1. **`app/rules/eligibility.py`** — the decision authority. Plain deterministic
   Python, no model involved. Start here; everything else supports it.
2. **`app/rag/pipeline.py`** — the orchestration. Read `assess_application` top
   to bottom; it's the whole system in one function.
3. **`app/rag/retriever.py`** — how chunks are found and filtered.
4. **`app/rag/prompts.py`** — what the model is actually asked, and the rules it's
   told to follow.
5. **`app/ingestion/versioning.py`** — how policy versions are tracked and which
   one counts as current.

The architectural idea worth knowing: **the rule engine decides, the model only
explains.** `RuleAssessment` and `GroundedExplanation` stay separate types all
the way to the UI, so a model that hallucinates cannot change an outcome. It can
only produce a bad explanation of a correct decision — which is a bug you can
see, rather than one you can't.

---

## Calling the API directly

Install the **REST Client** extension, open `api.http`, and click "Send Request"
above any request. Working payloads for every endpoint are already in there.

Or use the auto-generated docs: start the API and open
<http://localhost:8000/docs> — you can fill in and send requests from the
browser.

Two things about the assessment endpoint that will otherwise cost you an hour:

- The application fields must be nested under an `"application"` key. Sending
  them at the top level returns a 422 listing every field as `extra_forbidden`,
  which looks like the schema is broken when it isn't.
- Fields are cross-validated. `existing_monthly_emi_inr` above zero with
  `number_of_existing_loans` at zero is rejected — an EMI with no loan behind it
  is a data error, and it's caught rather than quietly assessed.

---

## When something breaks

**Yellow squiggles under `from app.core.config import ...`** — VS Code is on the
wrong interpreter. Redo Step 3, then `Ctrl+Shift+P` → "Developer: Reload Window".

**`ModuleNotFoundError: No module named 'app'`** — you ran a script from inside a
subfolder. Run from the project root; the F5 configs already do.

**`ModuleNotFoundError: No module named 'chromadb'`** — the terminal you used
didn't have the venv active. Check for `(.venv)` in the prompt.

**Testing panel finds no tests** — Step 3, then reload the window.

**`Address already in use` on port 8000** — an earlier run is still going.
`pkill -f run_backend.py` on macOS/Linux; on Windows, Task Manager → end the
Python process. Or change the port: `python run_backend.py --port 8001`.

**Ollama "connection refused"** — `ollama serve` isn't running, or it's running
in a terminal you closed.

**Answers are nonsense after switching to Ollama** — you skipped Part 2 Step 4.
Run config 7.

**Starting completely over:**

```bash
python scripts/reset_data.py --yes
```

That deletes the index and version history. Your code and config are untouched.

---

## Reference

`RUNBOOK.md` in this folder is the full fourteen-block version — every command,
what success looks like, and what to do when it doesn't. This file is the VS
Code-shaped path through the same material.

`README.md` explains what the system does and why it's built this way.
`docs/architecture.md` goes deeper.

One honest caveat carried over from the runbook: **nothing here has been measured
for answer quality.** Everything was verified for correct wiring, which is a
different claim. `docs/chunking_strategy.md` and `docs/experiment_log.md` say
"not yet measured" on purpose — running blocks 8–14 of the runbook on your
machine is what fills them in.
