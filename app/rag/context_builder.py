"""Context assembly.

Two jobs, both of which affect answer quality more than they look like they
should:

1. **Number the sources.** Each chunk gets a marker (``S1``, ``S2``, …) that the
   model must quote when it makes a claim. Citations therefore come from a closed
   set that the system controls, so an invented citation is detectable by string
   comparison rather than by trusting the model.
2. **Budget the context.** Chunks are added in rank order until the character
   budget runs out. Cutting at the budget rather than at ``top_k`` means raising
   ``top_k`` during an experiment cannot quietly blow up latency and token cost.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import RetrievalSection
from app.core.logging_config import get_logger
from app.models.assessment import Citation
from app.models.documents import RetrievedChunk
from app.utils.text import truncate_for_display

_logger = get_logger(__name__)

CITATION_MARKER_PREFIX = "S"


@dataclass(frozen=True)
class AssembledContext:
    """The numbered sources block, plus the citations the model may choose from."""

    sources_block: str
    citations: list[Citation]
    included_chunks: list[RetrievedChunk]
    dropped_for_budget_count: int

    @property
    def character_count(self) -> int:
        return len(self.sources_block)

    @property
    def is_empty(self) -> bool:
        return not self.included_chunks


def build_citation_marker(position: int) -> str:
    return f"{CITATION_MARKER_PREFIX}{position}"


def _format_source_entry(marker: str, chunk: RetrievedChunk) -> str:
    """One numbered extract, with the provenance the model is allowed to repeat.

    Version and page are included in the block itself so the model never has to
    infer them — inference is where fabricated page numbers come from.
    """
    metadata = chunk.metadata
    page_fragment = f" | Page: {metadata.page_number}" if metadata.page_number else ""
    effective_fragment = (
        f" | Effective: {metadata.effective_date}" if metadata.effective_date else ""
    )
    return (
        f"[{marker}] {metadata.document_title} — {metadata.institution}\n"
        f"Version: {metadata.version_number}{page_fragment}{effective_fragment}\n"
        f"Extract: {chunk.text}"
    )


def assemble_context(
    chunks: list[RetrievedChunk], settings: RetrievalSection
) -> AssembledContext:
    """Number, budget and format retrieved chunks into a prompt-ready block."""
    included_chunks: list[RetrievedChunk] = []
    formatted_entries: list[str] = []
    citations: list[Citation] = []
    used_characters = 0
    dropped_for_budget = 0

    for position, chunk in enumerate(chunks, start=1):
        marker = build_citation_marker(position)
        entry = _format_source_entry(marker, chunk)
        # +2 for the blank line that will separate entries in the joined block.
        if used_characters + len(entry) + 2 > settings.max_context_characters and included_chunks:
            dropped_for_budget += 1
            continue

        used_characters += len(entry) + 2
        included_chunks.append(chunk)
        formatted_entries.append(entry)
        citations.append(
            Citation(
                marker=marker,
                source_name=chunk.metadata.source_name,
                institution=chunk.metadata.institution,
                document_title=chunk.metadata.document_title,
                version_number=chunk.metadata.version_number,
                version_label=f"Version {chunk.metadata.version_number}",
                url=chunk.metadata.url,
                page_number=chunk.metadata.page_number,
                effective_date=chunk.metadata.effective_date,
                excerpt=truncate_for_display(chunk.text),
                similarity=chunk.similarity,
            )
        )

    if dropped_for_budget:
        _logger.info(
            "context budget dropped %d chunk(s) beyond %d characters",
            dropped_for_budget,
            settings.max_context_characters,
        )

    return AssembledContext(
        sources_block="\n\n".join(formatted_entries),
        citations=citations,
        included_chunks=included_chunks,
        dropped_for_budget_count=dropped_for_budget,
    )
