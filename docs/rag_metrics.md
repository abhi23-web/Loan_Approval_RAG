# RAG metrics — what each one answers and why it is here

Every metric below is computed in `evaluation/metrics.py`. Nothing is measured
because it is fashionable; each entry states the question it answers and the
decision it informs in a *lending* system.

---

## Retrieval metrics

### Context precision

> Of the chunks we put in the prompt, what share were actually about the question?

`relevant_retrieved / retrieved`. A chunk counts as relevant only if it comes from
the expected document, at the expected version, and contains one of the expected
facts. All three conditions are required: a chunk from the right document that
lacks the fact did not help, and a chunk with the fact taken from a superseded
version is actively harmful here.

**Why it matters:** low precision means paying tokens to read noise, and noise is
what a model drifts towards when the real answer is thin. It is also the first
thing to check when latency and cost creep up.

### Context recall

> Did the retrieved text actually contain the facts needed to answer?

Share of the case's expected keywords found anywhere in the retrieved context.
Digit-grouping commas are normalised, so `INR 15,000` matches an expectation of
`15000`.

**Why it matters:** this is the **ceiling on correctness**. No prompt, no model
and no reranker can recover a fact that was never retrieved. When correctness
drops, recall is the first place to look.

### MRR (mean reciprocal rank)

> How high up did the first genuinely relevant chunk appear?

`1 / rank` of the first relevant chunk, averaged over cases.

**Why it matters:** rank is not cosmetic. Models weight early context more
heavily, so the same chunk at position 1 and at position 8 produce different
answers. MRR is the metric that moves when you change `top_k`, reranking, or the
per-source cap.

### NDCG

Rank-sensitive like MRR, but credits every relevant chunk rather than only the
first, discounting by `log2(rank + 1)`. Reported for cases where more than one
chunk is genuinely relevant — a slab table split across two chunks, for example.

---

## Generation metrics

### Faithfulness

> Is every claim in the answer supported by the retrieved text?

Two implementations:

- **Lexical proxy** (always runs, deterministic, free): the share of answer
  sentences whose content words overlap the retrieved context above a threshold.
  Honest about what it is — it cannot detect a claim that reuses the context's
  vocabulary to say something the context does not. It is labelled a proxy in
  every report.
- **LLM judge** (`--judge`): the configured model is asked what share of the
  answer's factual claims are directly supported, and replies as JSON. Closer to
  the concept, costs one model call per execution. An unparsable judge response
  returns `null` rather than zero, because "the judge broke" and "the answer was
  unfaithful" must not look the same in a results file.

**Why it matters:** this is the hallucination metric. In a lending context an
unsupported claim about policy is a misrepresentation to an applicant, not a
stylistic flaw.

### Answer relevancy

> Does the answer address the question that was asked?

Same two implementations. Catches the failure where a perfectly faithful answer
quotes the wrong clause — it cites real policy, accurately, about something else.

---

## System metrics

| Metric | Why it is tracked |
| --- | --- |
| Retrieval latency | Isolates search cost from generation cost; usually small and stable, so a spike means an index problem |
| End-to-end latency (mean and p95) | p95 is what an applicant experiences on a bad request; the mean hides it |
| Context characters | The direct lever on prompt cost, and the thing `max_context_characters` caps |
| Prompt / completion tokens | Reported by the model where available, so cost comparisons across configurations are real rather than estimated |

---

## Business and quality metrics

### Correct answer rate

Ground-truth correctness against the golden dataset. Each case declares required
patterns and, crucially, **forbidden patterns** — an answer that quotes the
superseded figure alongside the current one is not correct, and without the
forbidden list it would score as if it were.

### Citation correctness

Did the answer point at the document the fact actually came from?

### Version correctness

Did every citation of that document name the **right version**? Strict on
purpose: one citation of a superseded version alongside a correct one still
misleads a reader about which rule applies.

**Why it matters:** this is the metric the entire versioning design exists to
move. If it is not 100% on the golden dataset, version filtering is not working,
no matter how good the other numbers look.

### Grounded rate and fabricated-citation rate

Share of answers where every citation marker the model wrote was one the system
supplied. The complement is the fabricated-citation rate. Because the marker set
is closed and validated by string comparison, this is a fact rather than an
estimate. A non-zero rate means the prompt or the model needs attention — the
invalid marker is stripped before the applicant sees it, but the attempt is a
signal.

### Insufficient-information rate

How often the system declines. It should not be zero — a system that always has
an answer is a system that fabricates one. It should also not be high, which
would mean retrieval is failing.

---

## Reproducibility metrics

Measured by executing every case `repeat_runs` times in a single evaluation:

| Measure | What identical means |
| --- | --- |
| `answer_consistency` | Byte-identical answer text |
| `retrieval_consistency` | Same chunk ids **in the same order** |
| `citation_consistency` | Same citation markers |
| `version_consistency` | Same cited document versions |

Retrieval consistency is compared on the ordered list, not the set: two runs that
retrieve the same chunks in a different order can still produce different
answers, so order is part of the guarantee.

**What consistency does and does not prove.** Retrieval consistency below 100% is
a bug in this codebase — the retrieval path is deterministic by construction.
Answer consistency below 100% with retrieval at 100% is the model, and is the
honest measurement of how far `temperature=0` plus a fixed seed actually gets you
on your hardware. That number belongs in the README, not an assurance that LLMs
are deterministic.

---

## Metrics deliberately not implemented

- **BLEU / ROUGE against a reference answer.** These reward surface wording. A
  correct answer phrased differently would score poorly and a wrong answer
  phrased similarly would score well.
- **Embedding similarity between answer and reference.** Same problem, less
  interpretable, and it would use the same embedding model being evaluated.
- **A single composite "RAG score".** It hides which dimension moved, which is
  the only thing an optimization loop needs to know.
