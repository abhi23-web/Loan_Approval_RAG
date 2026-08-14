"""Grounded generation and citation validation.

The model's output is not trusted as written. Three checks run on every response:

1. **Empty context.** If nothing was retrieved, the model is never called at all.
   Asking a model to explain policy with no policy in front of it is how
   confident fabrication happens.
2. **Citation validity.** Markers written by the model are matched against the
   closed set the prompt supplied. An unknown marker is stripped and the response
   is flagged ungrounded — a fabricated citation is caught by string comparison,
   not by hoping the prompt held.
3. **At least one citation.** A policy explanation with no citation is an opinion.
   When none survives validation, the fixed insufficient-information sentence is
   returned instead of the model's text.
"""

from __future__ import annotations

import re

from app.core.logging_config import get_logger
from app.core.tracing import add_trace_metadata, traced
from app.models.assessment import Citation, GroundedExplanation
from app.rag.context_builder import AssembledContext
from app.rag.llm import LLMProvider
from app.rag.prompts import INSUFFICIENT_INFORMATION_SENTENCE, SYSTEM_PROMPT

_logger = get_logger(__name__)

_CITATION_MARKER_PATTERN = re.compile(r"\[(S\d{1,2})\]")


def extract_cited_markers(answer_text: str) -> list[str]:
    """Markers the model actually wrote, in order of first appearance."""
    seen: set[str] = set()
    ordered_markers: list[str] = []
    for marker in _CITATION_MARKER_PATTERN.findall(answer_text):
        if marker not in seen:
            seen.add(marker)
            ordered_markers.append(marker)
    return ordered_markers


def strip_unknown_markers(answer_text: str, allowed_markers: set[str]) -> tuple[str, list[str]]:
    """Remove markers that were never offered, returning the cleaned text.

    The claim itself is left in place rather than deleted: silently removing a
    sentence would hide the failure, whereas an uncited sentence is visible to
    the reviewer and pushes the response's grounded flag to false.
    """
    invalid_markers: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        marker = match.group(1)
        if marker in allowed_markers:
            return match.group(0)
        invalid_markers.append(marker)
        return ""

    return _CITATION_MARKER_PATTERN.sub(_replace, answer_text).strip(), invalid_markers


class GroundedAnswerGenerator:
    """Calls the model and enforces that its answer stays inside the evidence."""

    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm_provider = llm_provider

    @traced("rag.generate", run_type="llm")
    def generate(self, user_prompt: str, context: AssembledContext) -> GroundedExplanation:
        if context.is_empty:
            _logger.warning("no context retrieved; refusing to call the model")
            return self._insufficient_information(reason="empty_context")

        llm_response = self._llm_provider.complete(SYSTEM_PROMPT, user_prompt)
        allowed_markers = {citation.marker for citation in context.citations}
        cleaned_text, invalid_markers = strip_unknown_markers(llm_response.text, allowed_markers)

        if invalid_markers:
            # Worth a warning, not a silent fix: repeated fabricated markers mean
            # the prompt or the model needs changing.
            _logger.warning(
                "model produced %d citation marker(s) that were never supplied: %s",
                len(invalid_markers),
                sorted(set(invalid_markers)),
            )

        if cleaned_text.strip() == INSUFFICIENT_INFORMATION_SENTENCE:
            return self._insufficient_information(
                reason="model_declined", model_name=llm_response.model
            )

        used_markers = set(extract_cited_markers(cleaned_text))
        used_citations = [
            citation for citation in context.citations if citation.marker in used_markers
        ]

        if not used_citations:
            _logger.warning("model answer cited no valid source; returning insufficient information")
            return self._insufficient_information(
                reason="no_valid_citation", model_name=llm_response.model
            )

        add_trace_metadata(
            citations_used=sorted(used_markers),
            citations_offered=sorted(allowed_markers),
            invalid_markers=sorted(set(invalid_markers)),
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
        )
        return GroundedExplanation(
            explanation=cleaned_text,
            citations=used_citations,
            is_grounded=not invalid_markers,
            insufficient_information=False,
            model_name=llm_response.model,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
        )

    @staticmethod
    def _insufficient_information(
        *, reason: str, model_name: str = "not_invoked"
    ) -> GroundedExplanation:
        add_trace_metadata(insufficient_information_reason=reason)
        return GroundedExplanation(
            explanation=INSUFFICIENT_INFORMATION_SENTENCE,
            citations=[],
            is_grounded=False,
            insufficient_information=True,
            model_name=model_name,
        )


def citations_for_markers(
    citations: list[Citation], markers: list[str]
) -> list[Citation]:
    """Look up citation objects for a list of markers, preserving marker order."""
    citation_by_marker = {citation.marker: citation for citation in citations}
    return [citation_by_marker[marker] for marker in markers if marker in citation_by_marker]
