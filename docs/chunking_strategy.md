# Chunking strategy

The live strategy is `recursive_800_100` — recursive splitting, 800-character
target, 100-character overlap — set in `config/settings.yaml`.

**It is a starting point chosen for reasons, not a conclusion drawn from
evidence.** The results table at the bottom of this document is empty until you
run the experiment. Filling it in and changing the default if the numbers say so
is step 14 of [`RUNBOOK.md`](../RUNBOOK.md).

---

## The ten questions

### 1. Why 800 characters?

A clause in these documents is one numbered statement plus its qualifiers —
roughly 300 to 700 characters. 800 fits a whole clause and usually its immediate
neighbour, which is what an LLM needs to answer "what is the maximum FOIR" without
having to stitch two chunks together. Smaller windows split clauses; much larger
ones start bundling unrelated clauses into a single retrievable unit.

### 2. Why 100 characters of overlap?

Roughly one sentence. Enough that a rule split across a boundary appears whole in
one of the two chunks, small enough that duplicated text is about 12% of the
index rather than 30%. Overlap is pure cost — duplicated storage, duplicated
tokens if both chunks retrieve — so it should be the smallest value that stops
boundary loss.

### 3. What goes wrong when chunks are too small?

The chunk stops being self-contained. *"3.2 Applications with a score between 675
and 724 may be considered for a conditional sanction"* retrieved without clause
3.1 leaves the model unable to say what the standard threshold is. Retrieval
precision looks excellent — the chunk is exactly on topic — while the answer is
incomplete. This is the failure mode that a precision metric alone will not catch.

### 4. What goes wrong when chunks are too large?

Three things. Relevance dilutes: the embedding averages several topics, so the
chunk matches everything weakly and nothing strongly. Precision drops: most of
what reaches the prompt is not about the question. Cost and latency rise linearly
with characters retrieved. In a policy corpus the large-chunk failure usually
shows up as an answer that is *correct but about the wrong clause*.

### 5. Why does overlap help?

Because boundaries are arbitrary and rules are not. A slab table split between
"Above INR 30,00,000 and up to INR 75,00,000" and "80 percent" produces two
chunks, neither of which answers the question. Overlap makes the odds of that
much lower without needing the splitter to be clever.

### 6. How does chunking affect retrieval precision?

Precision is the share of retrieved chunks that are genuinely about the question.
Smaller chunks raise it, because each unit is more topically pure. The limit is
that a very small chunk can be perfectly on-topic and still not carry the fact.

### 7. How does chunking affect context recall?

Recall is whether the retrieved text contained the facts needed. Larger chunks
raise it, because each retrieved unit carries more surrounding material. Recall
is the ceiling on correctness — no prompt can fix a fact that was never
retrieved — which is why a strategy that wins on precision but loses on recall is
usually the wrong choice for this domain.

### 8. How does chunking affect token cost?

Directly. Tokens per request ≈ `top_k × mean chunk size`. Moving from 800 to
1000 with `top_k = 5` adds roughly 1000 characters, about 250 tokens, to every
single request. The `max_context_characters` budget caps the damage, but the cap
works by *dropping* chunks, which trades cost against recall rather than fixing
anything.

### 9. How does chunking affect latency?

At ingestion: semantic chunking is far slower than the others because it embeds
every sentence group to find boundaries. At query time: retrieval latency barely
moves, but generation latency rises with prompt size, and on a local model that
is the dominant term. A larger chunk size therefore shows up mostly as slower
answers, not slower search.

### 10. Why is recursive splitting a good fit for policy documents specifically?

Because these documents are hierarchical and the separator ladder walks that
hierarchy. `config/chunking.yaml` tries paragraph breaks first, then line breaks,
then sentence ends, and only cuts inside a sentence when a single sentence is
genuinely longer than the chunk size. Fixed-width windows ignore all of that and
cut through the middle of numbered clauses and table rows. Semantic chunking also
respects meaning, but it pays an embedding call per sentence group and its
boundaries are less predictable — which matters when you need to explain to a
reviewer why a specific clause landed where it did.

---

## Strategies available

| Name | Type | Size | Overlap | Rationale |
| --- | --- | --- | --- | --- |
| `fixed_500_50` | fixed | 500 | 50 | Control. Ignores structure, so it shows what structure-awareness is worth |
| `recursive_500_50` | recursive | 500 | 50 | High precision per chunk; risk of orphaned qualifiers |
| `recursive_800_100` | recursive | 800 | 100 | Current default: about one clause plus context |
| `recursive_1000_150` | recursive | 1000 | 150 | Favours recall on multi-paragraph rules such as slab tables |
| `semantic` | semantic | — | — | Embedding-driven boundaries; most expensive to build |

Chunks may exceed the target size by up to the overlap: the overlap tail is
prepended before the next piece is measured. The size is a target, not a cap.

---

## Results

**Status: not yet measured.**

No numbers are written here by hand. To fill this table:

```bash
python scripts/ingest.py --all-strategies --force
python evaluation/experiments.py chunking
```

Then copy the generated table from
`evaluation/experiment_results/comparison_chunking.md`, which also records every
setting held constant across the runs.

| Strategy | Chunk size | Overlap | Context precision | Context recall | Faithfulness | MRR | Mean latency | Prompt tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fixed_500_50` | 500 | 50 | not measured | not measured | not measured | not measured | not measured | not measured |
| `recursive_500_50` | 500 | 50 | not measured | not measured | not measured | not measured | not measured | not measured |
| `recursive_800_100` | 800 | 100 | not measured | not measured | not measured | not measured | not measured | not measured |
| `recursive_1000_150` | 1000 | 150 | not measured | not measured | not measured | not measured | not measured | not measured |
| `semantic` | — | — | not measured | not measured | not measured | not measured | not measured | not measured |

### How to read the result when you have it

There will rarely be a strategy that wins everything. The usual shape is that the
largest chunks win recall and lose precision and cost. Decide with these
priorities, in this order, and record the decision in
[`experiment_log.md`](experiment_log.md):

1. **Correct answer rate and version correctness.** A wrong number told
   confidently to an applicant is the worst outcome available.
2. **Context recall.** It bounds correctness; precision problems can be mitigated
   with a reranker later, missing facts cannot.
3. **Faithfulness.** Prefer the strategy the model can stay inside.
4. **Latency and token cost.** Only as a tie-breaker, unless the gap is large.

If two strategies are within noise of each other, keep the cheaper one — and note
that they were within noise, so the next person does not re-litigate it.
