"""Citation validation and the insufficient-information guard.

These tests encode the anti-hallucination contract: a marker the system never
supplied cannot survive into a response, and an answer with no valid citation is
replaced rather than shown.
"""

from __future__ import annotations

from app.core.config import get_settings
from app.models.assessment import Citation
from app.models.documents import ChunkMetadata, RetrievedChunk
from app.rag.context_builder import assemble_context
from app.rag.generator import (
    GroundedAnswerGenerator,
    extract_cited_markers,
    strip_unknown_markers,
)
from app.rag.llm import LLMProvider, LLMResponse
from app.rag.prompts import INSUFFICIENT_INFORMATION_SENTENCE


class ScriptedLLM(LLMProvider):
    """Returns a fixed answer so the validator, not the model, is under test."""

    def __init__(self, scripted_answer: str) -> None:
        super().__init__("scripted")
        self._scripted_answer = scripted_answer
        self.was_called = False

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        self.was_called = True
        return LLMResponse(
            text=self._scripted_answer,
            model="scripted",
            prompt_tokens=100,
            completion_tokens=20,
            latency_ms=1.0,
        )


def _chunk(chunk_index: int, text: str, version_number: int = 3) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        similarity=0.9 - chunk_index * 0.01,
        rank=chunk_index + 1,
        metadata=ChunkMetadata(
            chunk_id=f"meridian_home_loan_policy::v{version_number}::recursive_800_100::{chunk_index:05d}",
            chunk_index=chunk_index,
            source_name="meridian_home_loan_policy",
            version_id=f"meridian_home_loan_policy::v{version_number}",
            version_number=version_number,
            institution="Meridian Housing Finance Limited (illustrative)",
            document_title="Meridian Retail Home Loan Credit Policy",
            document_type="credit_policy",
            authority="primary",
            url="file://documents/local_policies/current/meridian_home_loan_policy.md",
            effective_date="2026-01-01",
            page_number=None,
            chunking_strategy="recursive_800_100",
            embedding_model="deterministic-hashing-256",
            ingested_at="2026-08-14T00:00:00+00:00",
            character_count=len(text),
        ),
    )


def _context(chunk_count: int = 2):
    chunks = [
        _chunk(0, "3.1 The minimum acceptable CIBIL score for a standard sanction is 725."),
        _chunk(1, "5.2 The maximum permitted FOIR is 50 percent."),
    ][:chunk_count]
    return assemble_context(chunks, get_settings().retrieval)


def test_context_is_numbered_and_carries_version_metadata() -> None:
    context = _context()
    assert "[S1]" in context.sources_block
    assert "[S2]" in context.sources_block
    assert "Version: 3" in context.sources_block
    assert [citation.marker for citation in context.citations] == ["S1", "S2"]


def test_valid_markers_survive_and_become_citations() -> None:
    generator = GroundedAnswerGenerator(
        ScriptedLLM("The minimum score is 725 [S1]. The FOIR ceiling is 50 percent [S2].")
    )
    explanation = generator.generate("prompt", _context())

    assert explanation.is_grounded
    assert not explanation.insufficient_information
    assert [citation.marker for citation in explanation.citations] == ["S1", "S2"]


def test_a_fabricated_marker_is_stripped_and_flagged() -> None:
    generator = GroundedAnswerGenerator(
        ScriptedLLM("The minimum score is 725 [S1], and clause 12 applies [S9].")
    )
    explanation = generator.generate("prompt", _context())

    assert "[S9]" not in explanation.explanation
    assert explanation.is_grounded is False, "an invented citation must not read as grounded"
    assert [citation.marker for citation in explanation.citations] == ["S1"]


def test_an_answer_with_no_citation_becomes_insufficient_information() -> None:
    generator = GroundedAnswerGenerator(ScriptedLLM("Home loans usually require a good score."))
    explanation = generator.generate("prompt", _context())

    assert explanation.insufficient_information
    assert explanation.explanation == INSUFFICIENT_INFORMATION_SENTENCE
    assert explanation.citations == []


def test_the_model_is_not_called_when_nothing_was_retrieved() -> None:
    """No context means no question worth asking a model."""
    scripted_llm = ScriptedLLM("should never be produced")
    generator = GroundedAnswerGenerator(scripted_llm)
    explanation = generator.generate("prompt", _context(chunk_count=0))

    assert scripted_llm.was_called is False
    assert explanation.insufficient_information
    assert explanation.model_name == "not_invoked"


def test_the_models_own_refusal_is_honoured() -> None:
    generator = GroundedAnswerGenerator(ScriptedLLM(INSUFFICIENT_INFORMATION_SENTENCE))
    explanation = generator.generate("prompt", _context())
    assert explanation.insufficient_information


def test_marker_helpers() -> None:
    assert extract_cited_markers("a [S2] b [S1] c [S2]") == ["S2", "S1"]
    cleaned, invalid = strip_unknown_markers("a [S1] b [S7]", {"S1"})
    assert cleaned == "a [S1] b"
    assert invalid == ["S7"]


def test_context_budget_drops_chunks_rather_than_overflowing() -> None:
    """Raising top_k must not be able to blow the prompt budget."""
    retrieval_settings = get_settings().retrieval.model_copy(
        update={"max_context_characters": 400}
    )
    long_chunks = [_chunk(index, "policy text " * 40) for index in range(5)]
    context = assemble_context(long_chunks, retrieval_settings)

    assert context.dropped_for_budget_count > 0
    assert len(context.included_chunks) < len(long_chunks)


def test_citations_expose_everything_a_reviewer_needs() -> None:
    citation: Citation = _context().citations[0]
    assert citation.document_title
    assert citation.version_number == 3
    assert citation.url
    assert citation.excerpt
