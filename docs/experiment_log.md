# Experiment log

One row per controlled experiment. **One variable changes per row.** Every row
must reference the result files that justify it, so any number here can be traced
back to the executions that produced it.

Fill this in as you run experiments. Nothing is pre-filled, because nothing has
been measured on real models yet.

---

## How to add a row

```bash
# 1. Baseline, if you do not have one for this configuration
python evaluation/run_evaluation.py

# 2. Change exactly one variable — prefer an env override to editing YAML
HLR__RETRIEVAL__TOP_K=8 python evaluation/run_evaluation.py

# 3. Compare the two result files in evaluation/experiment_results/
# 4. Record the row below, then keep or revert the change
```

Sweeps that generate several rows at once:

```bash
python scripts/ingest.py --all-strategies --force
python evaluation/experiments.py chunking
python evaluation/experiments.py top-k --values 3 5 8 10
```

---

## Log

| ID | Date | Change | Reason | Precision | Recall | MRR | Faithfulness | Correct | Version ok | Latency | Tokens | Decision | Result file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-000 | — | Baseline: `recursive_800_100`, `top_k=5`, `min_similarity=0.25` | Starting point chosen from document structure, not from evidence | not measured | not measured | not measured | not measured | not measured | not measured | not measured | not measured | pending | — |

---

## Experiments worth running, in order of expected value

1. **Chunking strategy** (`experiments.py chunking`). The largest single lever.
   Expect the big-chunk strategies to win recall and lose precision and cost.
2. **`top_k`** (`experiments.py top-k`). Note the trap: with only the local
   policy enabled, `retrieval.max_chunks_per_source = 3` binds before `top_k`
   does and the sweep is flat. Enable more sources or raise the cap first.
3. **`min_similarity`**. `HLR__RETRIEVAL__MIN_SIMILARITY=0.15` and `0.35`. Watch
   `dropped_below_threshold_count` and the insufficient-information rate together
   — a threshold that is too high shows up as polite refusals, not as errors.
4. **Embedding model**. `nomic-embed-text` versus `mxbai-embed-large`. Requires a
   full re-index (`--force`), and the comparison is only valid if both indexes
   were built with their own model.
5. **Prompt**. Change `app/rag/prompts.py`, bump `PROMPT_VERSION`, re-run. The
   version string is recorded in every result file, so a metric shift can be
   attributed to it.
6. **`max_chunks_per_source`**. Only meaningful once several sources are ingested.
7. **Reranking**. Not implemented. Add a cross-encoder over the over-fetched
   candidates and measure MRR before adopting it.

## Rules for this log

- One variable per row. A row with two changes explains nothing.
- Record rejections. "Tried semantic chunking, +0.04 recall, 3× ingestion time,
  rejected" saves the next person a day.
- Record ties. If two configurations are within noise, say so, so nobody
  re-litigates it in a month.
- Never write a number here that is not in a file under
  `evaluation/experiment_results/`.
