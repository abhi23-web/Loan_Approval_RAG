# Coding standards

The five that matter most for this codebase, each with a real example from it.
Enforced by `ruff check app/ evaluation/ tests/` where a linter can do it, and by
review where it can't.

---

## 1. Names describe the thing, not its type or position

`i`, `x`, `tmp`, `data`, `result` say nothing. A name should let you read a line
in isolation and know what it holds.

```python
# no
for i in range(len(c)):
    r = f(c[i])

# yes
for chunk_index, chunk in enumerate(retrieved_chunks):
    similarity = cosine_similarity(query_vector, chunk.embedding)
```

The convention used throughout:

- **Units in the name** when a number has one: `monthly_income_inr`,
  `request_timeout_seconds`, `max_download_bytes`, `latency_ms`. This is not
  verbosity — `timeout=300` is ambiguous between seconds and milliseconds, and
  the bug it causes takes an hour to find.
- **Booleans read as assertions**: `is_grounded`, `restrict_to_active_versions`,
  `redact_applicant_pii`.
- **Collections are plural**, scalars singular: `chunks` / `chunk`.
- **Functions are verbs**: `assemble_context`, `elect_active_version`,
  `build_embedding_provider`.
- **A leading underscore means private**: `_logger`, `_embed_batch`. It is a
  contract with the next reader — nothing outside this module depends on it.

The one exception worth naming: `for _ in range(3)` is correct when the variable
genuinely is not used. `_` means "deliberately discarded", not "I couldn't think
of a name".

---

## 2. Types at every boundary, validated at the edge

Every public function is annotated, and every external input is parsed into a
typed model before any logic touches it.

```python
def retrieve(self, request: RetrievalRequest) -> RetrievalOutcome:
    ...
```

Configuration is Pydantic (`app/core/config.py`), so a typo in `settings.yaml`
fails at startup naming the exact key — rather than silently defaulting and
producing quietly wrong retrieval three hours later. API payloads are Pydantic
(`app/api/schemas.py`), so a malformed request is a 422 with a field list, not a
`KeyError` in the middle of a pipeline.

The principle: **parse, don't validate.** Convert unstructured input into a type
that cannot represent an invalid state, once, at the boundary. Everything inside
then works with values it can trust and needs no defensive checks.

This is also why `RuleAssessment` and `GroundedExplanation` are separate types
rather than fields on one dict. The type system enforces that a model's output
cannot be mistaken for a decision.

---

## 3. Comments explain *why*, never *what*

The code already says what it does. A comment that restates it is noise that
goes stale.

```python
# no — restates the line
# increment the counter
processed_count += 1

# yes — explains a decision the code cannot express
# Over-fetch: the filters below discard chunks, so asking for exactly top_k
# would leave the prompt short whenever anything is filtered out.
fetch_count = max(settings.top_k * 4, 20)
```

Comments earn their place when they record a **trade-off**, a **non-obvious
constraint**, or a **bug that was already paid for**:

```python
# Chroma rejects None in metadata, so normalise anything the loaders left
# unset. Doing it here — once, at the boundary — beats defending against it
# at every call site.
```

Module docstrings carry the design rationale, so a reader opening a file cold
learns why it exists before reading how it works. `app/rag/llm.py` opens by
stating exactly what "deterministic" does and does not guarantee — because the
honest scope of that claim is not derivable from the code.

---

## 4. One function, one job — and functions short enough to hold in your head

If you cannot describe a function without "and", it is two functions.

`app/rag/pipeline.py` is the clearest case. `assess_application` reads as five
named steps, each delegating:

```python
rule_assessment = self._rule_engine.assess(application)
retrieval_outcome = self._retriever.retrieve(...)
context = assemble_context(retrieval_outcome.chunks, self._settings.retrieval)
explanation = self._generator.generate(...)
```

You can read the whole system's control flow in that one function and descend
only into the part you care about. The corollary is **dependency injection**:
the pipeline receives its retriever, generator and rule engine rather than
constructing them (`app/services/container.py` does the wiring). That is what
lets the test suite substitute a stub LLM and a hashing embedder, and is why 86
tests run offline with no model server.

---

## 5. Errors that tell you what to do about them

An exception is a message to a person who is stuck. `raise ValueError("invalid
config")` fails that person.

```python
raise LLMError(
    f"Ollama does not have model '{self._settings.model}'. "
    f"Run: ollama pull {self._settings.model}"
)
```

Three rules the codebase follows:

- **Name the fix, not just the fault.** The message above contains the command
  to type.
- **Fail at construction, not at first use.** `OpenAICompatibleLLMProvider`
  checks for its API key in `__init__`, so a missing key surfaces at startup
  rather than after a user has filled in a form.
- **Never catch broadly and continue silently.** Where a broad catch is correct —
  one source failing must not kill an eight-source ingest — the failure becomes a
  *value* that gets reported (`SourceResult.ok`), not a swallowed exception.

The narrow exception to this is observability. `add_trace_metadata` swallows
everything, because a tracing failure must never take down the request path it
is observing.

---

## Running the checks

```bash
ruff check app/ evaluation/ tests/     # lint
ruff format --check app/               # formatting
python -m pytest                       # 86 tests, fully offline
```

Ruff is configured in `pyproject.toml` with `E, F, I, N, UP, B, C4, SIM, ANN,
RUF` enabled — including `ANN`, which requires annotations, and `N`, which
enforces naming conventions. The `ignore` list is short and each entry has a
stated reason next to it, because an unexplained suppression is how a standard
quietly stops being one.
