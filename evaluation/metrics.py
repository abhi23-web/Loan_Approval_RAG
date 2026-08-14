"""RAG evaluation metrics.

Each metric below is here because it answers a question this system has to be
able to answer about itself. Metrics that are merely popular are not included.

Retrieval
    context_precision  "Of the chunks we put in the prompt, what share were
                        actually about the question?" Low precision means the
                        model is paying tokens to read noise, and noise is what
                        it drifts towards when the real answer is thin.
    context_recall     "Did the retrieved text contain the facts needed to
                        answer?" This is the ceiling on correctness — no prompt
                        can fix a fact that was never retrieved.
    mrr                "How high up did the first genuinely relevant chunk
                        appear?" Rank matters because models weight early
                        context more heavily.
    ndcg               Rank-sensitive like MRR, but credits every relevant chunk
                        rather than only the first. Reported where more than one
                        chunk is relevant.

Generation
    faithfulness       "Is every claim in the answer supported by the retrieved
                        text?" For a lending system this is the hallucination
                        metric: an unsupported claim about policy is a
                        misrepresentation to an applicant.
    answer_relevancy   "Does the answer address the question that was asked?"
                        Catches the failure where a faithful answer quotes the
                        wrong clause perfectly.

Business / quality
    answer_correct     Exact-figure correctness against known ground truth.
    citation_correct   Did it cite the document the fact actually came from?
    version_correct    Did it cite the *right version* of that document? This is
                        the metric the whole versioning design exists to move.
    grounded_rate      Share of answers where every citation marker was one the
                        system supplied. Its complement is the fabricated-citation
                        rate.

Two faithfulness implementations are provided. The lexical one is a cheap,
deterministic proxy that always runs; it is explicitly a proxy and is labelled as
such in every report. The judge implementation uses the configured LLM and is
more faithful to the concept but costs a model call per claim, so it is opt-in.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from statistics import mean

from app.core.logging_config import get_logger
from app.models.assessment import Citation
from app.models.documents import RetrievedChunk
from app.rag.llm import LLMProvider

_logger = get_logger(__name__)

_WORD_PATTERN = re.compile(r"[a-z0-9][a-z0-9.\-]*")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Words carrying no discriminating power for overlap scoring.
_STOP_WORDS = frozenset(
    ["a", "an", "the", "and", "or", "of", "to", "in", "for", "on", "at", "by", "is", "are", "was", "were", "be", "been", "being", "with", "as", "that", "this", "these", "those", "it", "its", "from", "any", "all", "not", "no", "if", "then", "than", "which", "what", "when", "where", "who", "whom", "whose", "how", "must", "may", "can", "will", "shall", "should", "would", "could", "per", "percent", "under", "over", "above", "below", "into", "within", "upon", "such", "other", "same"]
)

# A claim sentence counts as supported when this share of its content words
# appears in the retrieved context. Chosen high enough that a paraphrase passes
# and a fabricated figure does not, and it is a threshold, not a truth.
_LEXICAL_SUPPORT_THRESHOLD = 0.65


def _normalise(text: str) -> str:
    """Lowercase, and strip digit-grouping commas so 15,000 matches 15000."""
    lowered = text.lower()
    return re.sub(r"(?<=\d),(?=\d)", "", lowered)


def content_words(text: str) -> set[str]:
    return {
        word
        for word in _WORD_PATTERN.findall(_normalise(text))
        if word not in _STOP_WORDS and len(word) > 1
    }


def keyword_present(keyword: str, haystack: str) -> bool:
    return _normalise(keyword) in _normalise(haystack)


# --------------------------------------------------------------------- retrieval


def is_chunk_relevant(
    chunk: RetrievedChunk,
    expected_source: str,
    expected_version: int | None,
    expected_keywords: list[str],
) -> bool:
    """Binary relevance: right document, right version, and carries a wanted fact.

    All three conditions are required on purpose. A chunk from the correct
    document that lacks the fact did not help, and a chunk with the fact taken
    from a superseded version is actively harmful in this domain.
    """
    if chunk.metadata.source_name != expected_source:
        return False
    if expected_version is not None and chunk.metadata.version_number != expected_version:
        return False
    return any(keyword_present(keyword, chunk.text) for keyword in expected_keywords)


def context_precision(relevance_flags: list[bool]) -> float:
    if not relevance_flags:
        return 0.0
    return sum(relevance_flags) / len(relevance_flags)


def context_recall(retrieved_chunks: list[RetrievedChunk], expected_keywords: list[str]) -> float:
    """Share of the expected facts that appear anywhere in the retrieved context."""
    if not expected_keywords:
        return 0.0
    combined_context = "\n".join(chunk.text for chunk in retrieved_chunks)
    found = sum(1 for keyword in expected_keywords if keyword_present(keyword, combined_context))
    return found / len(expected_keywords)


def reciprocal_rank(relevance_flags: list[bool]) -> float:
    for position, is_relevant in enumerate(relevance_flags, start=1):
        if is_relevant:
            return 1.0 / position
    return 0.0


def normalised_discounted_cumulative_gain(relevance_flags: list[bool]) -> float:
    """Binary-relevance NDCG@k.

    Discounts by log2 of rank, so a relevant chunk at position 1 is worth more
    than the same chunk at position 5 — which matches how a model actually reads
    a prompt.
    """
    if not relevance_flags or not any(relevance_flags):
        return 0.0
    discounted_gain = sum(
        1.0 / math.log2(position + 1)
        for position, is_relevant in enumerate(relevance_flags, start=1)
        if is_relevant
    )
    ideal_gain = sum(
        1.0 / math.log2(position + 1) for position in range(1, sum(relevance_flags) + 1)
    )
    return discounted_gain / ideal_gain


# -------------------------------------------------------------------- generation


def answer_matches_expectation(
    answer_text: str,
    expected_patterns: list[str],
    forbidden_patterns: list[str],
    acceptable_variations: list[str],
) -> bool:
    """Correctness against known ground truth.

    Every expected pattern must appear and no forbidden pattern may — the
    forbidden list is what catches an answer that quotes a superseded figure
    alongside the current one and would otherwise score as correct.
    """
    normalised_answer = _normalise(answer_text)
    for forbidden_pattern in forbidden_patterns:
        if re.search(forbidden_pattern, normalised_answer):
            return False
    if expected_patterns:
        return all(
            re.search(pattern, normalised_answer) for pattern in expected_patterns
        )
    return any(_normalise(variation) in normalised_answer for variation in acceptable_variations)


def lexical_faithfulness(answer_text: str, retrieved_chunks: list[RetrievedChunk]) -> float:
    """Proxy faithfulness: share of answer sentences supported by the context.

    A proxy, and labelled as one wherever it is reported. It cannot detect a
    claim that reuses the context's words to say something the context does not,
    which is why the judge implementation exists.
    """
    if not retrieved_chunks:
        return 0.0
    context_vocabulary = content_words("\n".join(chunk.text for chunk in retrieved_chunks))
    sentences = [
        sentence for sentence in _SENTENCE_SPLIT.split(answer_text.strip()) if sentence.strip()
    ]
    if not sentences:
        return 0.0

    support_scores: list[float] = []
    for sentence in sentences:
        sentence_words = content_words(sentence)
        if not sentence_words:
            continue
        overlap = len(sentence_words & context_vocabulary) / len(sentence_words)
        support_scores.append(1.0 if overlap >= _LEXICAL_SUPPORT_THRESHOLD else 0.0)
    return mean(support_scores) if support_scores else 0.0


def lexical_answer_relevancy(answer_text: str, question: str) -> float:
    """Proxy relevancy: share of the question's content words the answer engages.

    Deliberately simple. Relevancy is the metric least well served by a lexical
    proxy, so the number is reported next to the judge score rather than instead
    of it whenever the judge is enabled.
    """
    question_words = content_words(question)
    if not question_words:
        return 0.0
    return len(question_words & content_words(answer_text)) / len(question_words)


_JUDGE_SYSTEM_PROMPT = """\
You are a strict evaluator. You answer only with JSON. You never explain.
"""

_FAITHFULNESS_JUDGE_TEMPLATE = """\
CONTEXT
{context}

ANSWER
{answer}

Decide what share of the factual claims in ANSWER are directly supported by
CONTEXT. A claim that is plausible but absent from CONTEXT is NOT supported.

Reply with exactly this JSON and nothing else:
{{"supported_claims": <integer>, "total_claims": <integer>}}
"""

_RELEVANCY_JUDGE_TEMPLATE = """\
QUESTION
{question}

ANSWER
{answer}

Score how well ANSWER addresses QUESTION, from 0.0 (unrelated) to 1.0 (fully
answers it). Ignore whether the answer is factually correct.

Reply with exactly this JSON and nothing else:
{{"relevancy": <float between 0 and 1>}}
"""


def _parse_judge_json(raw_text: str) -> dict | None:
    """Pull the JSON object out of a judge response.

    Small local models often wrap JSON in prose or a code fence, so the first
    balanced object is extracted rather than assuming a clean response. A judge
    that cannot be parsed returns None and the caller falls back to the lexical
    proxy instead of recording a zero, which would be indistinguishable from a
    genuinely unfaithful answer.
    """
    match = re.search(r"\{.*?\}", raw_text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def judge_faithfulness(
    answer_text: str, retrieved_chunks: list[RetrievedChunk], judge: LLMProvider
) -> float | None:
    if not retrieved_chunks or not answer_text.strip():
        return None
    context = "\n\n".join(chunk.text for chunk in retrieved_chunks)
    response = judge.complete(
        _JUDGE_SYSTEM_PROMPT,
        _FAITHFULNESS_JUDGE_TEMPLATE.format(context=context, answer=answer_text),
    )
    parsed = _parse_judge_json(response.text)
    if not parsed:
        _logger.warning("faithfulness judge returned unparsable output; falling back to lexical")
        return None
    total_claims = parsed.get("total_claims") or 0
    supported_claims = parsed.get("supported_claims") or 0
    if not isinstance(total_claims, int) or total_claims <= 0:
        return None
    return max(0.0, min(1.0, supported_claims / total_claims))


def judge_answer_relevancy(answer_text: str, question: str, judge: LLMProvider) -> float | None:
    if not answer_text.strip():
        return None
    response = judge.complete(
        _JUDGE_SYSTEM_PROMPT,
        _RELEVANCY_JUDGE_TEMPLATE.format(question=question, answer=answer_text),
    )
    parsed = _parse_judge_json(response.text)
    if not parsed or "relevancy" not in parsed:
        _logger.warning("relevancy judge returned unparsable output; falling back to lexical")
        return None
    try:
        return max(0.0, min(1.0, float(parsed["relevancy"])))
    except (TypeError, ValueError):
        return None


# -------------------------------------------------------------------- citations


def citation_correct(citations: list[Citation], expected_source: str) -> bool:
    """Did the answer point at the document the fact actually came from?"""
    return any(citation.source_name == expected_source for citation in citations)


def version_correct(
    citations: list[Citation], expected_source: str, expected_version: int
) -> bool:
    """Did every citation of the expected document name the expected version?

    Strict on purpose. One citation of a superseded version alongside a correct
    one still misleads a reader about which rule applies.
    """
    relevant_citations = [
        citation for citation in citations if citation.source_name == expected_source
    ]
    if not relevant_citations:
        return False
    return all(citation.version_number == expected_version for citation in relevant_citations)


# ------------------------------------------------------------------ aggregation


@dataclass
class CaseMetrics:
    """Metrics for one execution of one golden-dataset case."""

    case_id: str
    answer_correct: bool
    context_precision: float
    context_recall: float
    reciprocal_rank: float
    ndcg: float
    faithfulness_lexical: float
    faithfulness_judge: float | None
    answer_relevancy_lexical: float
    answer_relevancy_judge: float | None
    citation_correct: bool
    version_correct: bool
    is_grounded: bool
    insufficient_information: bool
    retrieval_latency_ms: float
    total_latency_ms: float
    retrieved_chunk_count: int
    context_characters: int
    prompt_tokens: int | None
    completion_tokens: int | None


@dataclass
class ConsistencyMetrics:
    """Reproducibility across repeated executions of the same question."""

    answer_consistency: float
    retrieval_consistency: float
    citation_consistency: float
    version_consistency: float


@dataclass
class AggregateMetrics:
    """Run-level summary. Every field is an average over executed cases."""

    case_count: int
    execution_count: int
    correct_answer_rate: float
    context_precision: float
    context_recall: float
    mrr: float
    ndcg: float
    faithfulness_lexical: float
    faithfulness_judge: float | None
    answer_relevancy_lexical: float
    answer_relevancy_judge: float | None
    citation_correct_rate: float
    version_correct_rate: float
    grounded_rate: float
    hallucination_rate: float
    insufficient_information_rate: float
    mean_retrieval_latency_ms: float
    mean_total_latency_ms: float
    p95_total_latency_ms: float
    mean_retrieved_chunks: float
    mean_context_characters: float
    mean_prompt_tokens: float | None
    mean_completion_tokens: float | None
    consistency: ConsistencyMetrics | None = None
    notes: list[str] = field(default_factory=list)


def _mean_or_zero(values: list[float]) -> float:
    return round(mean(values), 4) if values else 0.0


def _mean_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return round(mean(present), 4) if present else None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((percentile / 100.0) * (len(ordered) - 1)))
    return round(ordered[index], 2)


def aggregate(
    case_metrics: list[CaseMetrics], consistency: ConsistencyMetrics | None = None
) -> AggregateMetrics:
    if not case_metrics:
        raise ValueError("cannot aggregate an empty metrics list")

    unique_case_ids = {metrics.case_id for metrics in case_metrics}
    total_latencies = [metrics.total_latency_ms for metrics in case_metrics]

    return AggregateMetrics(
        case_count=len(unique_case_ids),
        execution_count=len(case_metrics),
        correct_answer_rate=_mean_or_zero([float(m.answer_correct) for m in case_metrics]),
        context_precision=_mean_or_zero([m.context_precision for m in case_metrics]),
        context_recall=_mean_or_zero([m.context_recall for m in case_metrics]),
        mrr=_mean_or_zero([m.reciprocal_rank for m in case_metrics]),
        ndcg=_mean_or_zero([m.ndcg for m in case_metrics]),
        faithfulness_lexical=_mean_or_zero([m.faithfulness_lexical for m in case_metrics]),
        faithfulness_judge=_mean_optional([m.faithfulness_judge for m in case_metrics]),
        answer_relevancy_lexical=_mean_or_zero(
            [m.answer_relevancy_lexical for m in case_metrics]
        ),
        answer_relevancy_judge=_mean_optional([m.answer_relevancy_judge for m in case_metrics]),
        citation_correct_rate=_mean_or_zero([float(m.citation_correct) for m in case_metrics]),
        version_correct_rate=_mean_or_zero([float(m.version_correct) for m in case_metrics]),
        grounded_rate=_mean_or_zero([float(m.is_grounded) for m in case_metrics]),
        hallucination_rate=round(
            1.0 - _mean_or_zero([float(m.is_grounded) for m in case_metrics]), 4
        ),
        insufficient_information_rate=_mean_or_zero(
            [float(m.insufficient_information) for m in case_metrics]
        ),
        mean_retrieval_latency_ms=_mean_or_zero([m.retrieval_latency_ms for m in case_metrics]),
        mean_total_latency_ms=_mean_or_zero(total_latencies),
        p95_total_latency_ms=_percentile(total_latencies, 95),
        mean_retrieved_chunks=_mean_or_zero(
            [float(m.retrieved_chunk_count) for m in case_metrics]
        ),
        mean_context_characters=_mean_or_zero(
            [float(m.context_characters) for m in case_metrics]
        ),
        mean_prompt_tokens=_mean_optional(
            [float(m.prompt_tokens) if m.prompt_tokens is not None else None for m in case_metrics]
        ),
        mean_completion_tokens=_mean_optional(
            [
                float(m.completion_tokens) if m.completion_tokens is not None else None
                for m in case_metrics
            ]
        ),
        consistency=consistency,
    )
