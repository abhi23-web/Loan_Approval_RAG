"""Chunking strategies.

Three strategies are implemented so they can be compared, not because three are
needed. They are implemented here rather than pulled from a framework for two
reasons: the behaviour is 150 lines and fully inspectable, and an experiment that
compares chunkers should not also be comparing framework versions.

Chunking happens **per page**. A chunk therefore never spans a page boundary,
which costs a little recall on rules that straddle a page break but buys an
unambiguous "page N" in every citation. For a lending system that trade is worth
making: a reviewer must be able to open the PDF and see the clause.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime

import numpy as np

from app.core.config import ChunkingStrategyConfig
from app.core.exceptions import ConfigurationError
from app.core.logging_config import get_logger
from app.models.documents import ChunkMetadata, DocumentVersion, ExtractedDocument, TextChunk
from app.utils.hashing import build_chunk_id
from app.utils.text import split_into_sentences

_logger = get_logger(__name__)


class Chunker(ABC):
    """Splits one page of text into chunk-sized strings."""

    @abstractmethod
    def split(self, text: str) -> list[str]: ...


class FixedSizeChunker(Chunker):
    """Blind fixed-width windows with a fixed stride.

    The control condition. It respects nothing about the document, so any gain a
    structure-aware splitter shows over it is attributable to structure awareness.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self._chunk_size = chunk_size
        self._stride = chunk_size - chunk_overlap

    def split(self, text: str) -> list[str]:
        if len(text) <= self._chunk_size:
            return [text] if text.strip() else []
        chunks = [
            text[start : start + self._chunk_size]
            for start in range(0, len(text), self._stride)
        ]
        return [chunk.strip() for chunk in chunks if chunk.strip()]


class RecursiveChunker(Chunker):
    """Split on the most meaningful separator that keeps pieces under the size.

    Policy documents are hierarchical — clause, sub-clause, sentence — so the
    separator ladder walks that hierarchy and only falls back to cutting inside a
    sentence when a single sentence is genuinely longer than the chunk size.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int, separators: list[str]) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators

    def split(self, text: str) -> list[str]:
        if not text.strip():
            return []
        pieces = self._split_to_pieces(text, self._separators)
        return self._merge_with_overlap(pieces)

    def _split_to_pieces(self, text: str, separators: list[str]) -> list[str]:
        """Recursively break ``text`` down until pieces fit, or separators run out."""
        if len(text) <= self._chunk_size or not separators:
            return [text] if text else []

        separator, remaining_separators = separators[0], separators[1:]
        if separator == "":
            # Last resort: a single unbroken run longer than chunk_size. Cut it
            # into stride-sized pieces rather than chunk_size ones, so that the
            # overlap tail added during merging cannot push a chunk over budget.
            stride = max(1, self._chunk_size - self._chunk_overlap)
            return [text[start : start + stride] for start in range(0, len(text), stride)]
        if separator not in text:
            return self._split_to_pieces(text, remaining_separators)

        pieces: list[str] = []
        for part in text.split(separator):
            if not part:
                continue
            # Re-attach the separator so joined chunks read as the original text.
            part_with_separator = part + separator
            if len(part_with_separator) > self._chunk_size:
                pieces.extend(self._split_to_pieces(part_with_separator, remaining_separators))
            else:
                pieces.append(part_with_separator)
        return pieces

    def _merge_with_overlap(self, pieces: list[str]) -> list[str]:
        """Greedily pack pieces up to chunk_size, carrying an overlap tail forward.

        The overlap is taken from the end of the emitted chunk so that a rule
        split across a boundary still appears whole in one of the two chunks.

        A chunk can therefore exceed ``chunk_size`` by up to ``chunk_overlap``:
        the tail is prepended before the next piece is measured. That is the
        conventional behaviour and it is bounded, but it is why the size shown in
        the experiment table is a target rather than a hard cap.
        """
        chunks: list[str] = []
        buffer = ""
        for piece in pieces:
            if buffer and len(buffer) + len(piece) > self._chunk_size:
                chunks.append(buffer.strip())
                buffer = buffer[-self._chunk_overlap :] if self._chunk_overlap else ""
            buffer += piece
        if buffer.strip():
            chunks.append(buffer.strip())
        return [chunk for chunk in chunks if chunk]


class SemanticChunker(Chunker):
    """Cut where consecutive sentence groups stop talking about the same thing.

    Boundaries come from embedding distance rather than character counts, so a
    short definition clause and a long slab table each end up as one chunk. The
    cost is one embedding call per sentence group at ingestion time, which is why
    this strategy must earn its place in the evaluation before being adopted.
    """

    def __init__(
        self,
        embedder: SupportsEmbedDocuments,
        breakpoint_percentile: float,
        buffer_sentences: int,
        min_chunk_characters: int,
        max_chunk_characters: int,
    ) -> None:
        self._embedder = embedder
        self._breakpoint_percentile = breakpoint_percentile
        self._buffer_sentences = max(0, buffer_sentences)
        self._min_chunk_characters = min_chunk_characters
        self._max_chunk_characters = max_chunk_characters

    def split(self, text: str) -> list[str]:
        sentences = split_into_sentences(text)
        if len(sentences) < 3:
            return [text.strip()] if text.strip() else []

        windows = self._build_windows(sentences)
        window_vectors = np.asarray(self._embedder.embed_documents(windows), dtype=np.float32)
        distances = self._consecutive_cosine_distances(window_vectors)
        if distances.size == 0:
            return [text.strip()]

        breakpoint_threshold = float(np.percentile(distances, self._breakpoint_percentile))
        boundary_indices = [
            index + 1
            for index, distance in enumerate(distances)
            if distance >= breakpoint_threshold
        ]
        return self._assemble(sentences, boundary_indices)

    def _build_windows(self, sentences: list[str]) -> list[str]:
        """Embed each sentence together with its neighbours.

        A bare sentence like "The same applies to clause 4.2." has almost no
        standalone meaning; buffering neighbours makes the distance signal
        reflect topic shift rather than sentence-length noise.
        """
        if self._buffer_sentences == 0:
            return sentences
        windows: list[str] = []
        for sentence_index in range(len(sentences)):
            window_start = max(0, sentence_index - self._buffer_sentences)
            window_end = min(len(sentences), sentence_index + self._buffer_sentences + 1)
            windows.append(" ".join(sentences[window_start:window_end]))
        return windows

    @staticmethod
    def _consecutive_cosine_distances(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # A zero vector would divide by zero; treat it as maximally distant later.
        normalised = vectors / np.clip(norms, 1e-12, None)
        similarities = np.sum(normalised[:-1] * normalised[1:], axis=1)
        return 1.0 - similarities

    def _assemble(self, sentences: list[str], boundary_indices: list[int]) -> list[str]:
        """Turn boundary positions into chunks, honouring the size guard rails."""
        chunks: list[str] = []
        current_sentences: list[str] = []
        boundaries = set(boundary_indices)

        for sentence_index, sentence in enumerate(sentences):
            current_sentences.append(sentence)
            current_length = sum(len(part) + 1 for part in current_sentences)
            is_boundary = (sentence_index + 1) in boundaries
            # Emit at a semantic boundary only once the chunk is worth retrieving,
            # and force an emit before it grows past the context-cost ceiling.
            if (is_boundary and current_length >= self._min_chunk_characters) or (
                current_length >= self._max_chunk_characters
            ):
                chunks.append(" ".join(current_sentences).strip())
                current_sentences = []

        if current_sentences:
            trailing_chunk = " ".join(current_sentences).strip()
            # A short trailing fragment belongs with its predecessor, not alone.
            if chunks and len(trailing_chunk) < self._min_chunk_characters:
                chunks[-1] = f"{chunks[-1]} {trailing_chunk}".strip()
            else:
                chunks.append(trailing_chunk)
        return [chunk for chunk in chunks if chunk]


class SupportsEmbedDocuments:  # pragma: no cover - structural typing helper
    """Minimal interface the semantic chunker needs from an embedding provider."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


def build_chunker(
    strategy: ChunkingStrategyConfig,
    recursive_separators: list[str],
    embedder: SupportsEmbedDocuments | None = None,
) -> Chunker:
    """Instantiate the chunker named by ``strategy``."""
    if strategy.type == "fixed":
        return FixedSizeChunker(strategy.chunk_size or 500, strategy.chunk_overlap or 50)
    if strategy.type == "recursive":
        return RecursiveChunker(
            strategy.chunk_size or 800, strategy.chunk_overlap or 100, recursive_separators
        )
    if strategy.type == "semantic":
        if embedder is None:
            raise ConfigurationError(
                "the semantic strategy needs an embedding provider to find boundaries"
            )
        return SemanticChunker(
            embedder=embedder,
            breakpoint_percentile=strategy.breakpoint_percentile or 90.0,
            buffer_sentences=strategy.buffer_sentences or 1,
            min_chunk_characters=strategy.min_chunk_characters or 300,
            max_chunk_characters=strategy.max_chunk_characters or 1600,
        )
    raise ConfigurationError(f"unsupported chunking strategy type '{strategy.type}'")


def chunk_document(
    extracted_document: ExtractedDocument,
    document_version: DocumentVersion,
    chunker: Chunker,
    *,
    strategy_name: str,
    embedding_model: str,
) -> list[TextChunk]:
    """Chunk every page and attach the metadata that makes a citation possible."""
    ingested_at = datetime.now(UTC).isoformat()
    chunks: list[TextChunk] = []

    for page in extracted_document.pages:
        for page_chunk_text in chunker.split(page.text):
            chunk_index = len(chunks)
            chunks.append(
                TextChunk(
                    text=page_chunk_text,
                    metadata=ChunkMetadata(
                        chunk_id=build_chunk_id(
                            document_version.version_id, strategy_name, chunk_index
                        ),
                        chunk_index=chunk_index,
                        source_name=document_version.source_name,
                        version_id=document_version.version_id,
                        version_number=document_version.version_number,
                        institution=document_version.institution,
                        document_title=document_version.document_title,
                        document_type=document_version.document_type,
                        authority=document_version.authority,
                        url=document_version.url,
                        effective_date=(
                            document_version.effective_date.isoformat()
                            if document_version.effective_date
                            else None
                        ),
                        page_number=page.page_number,
                        chunking_strategy=strategy_name,
                        embedding_model=embedding_model,
                        ingested_at=ingested_at,
                        character_count=len(page_chunk_text),
                    ),
                )
            )

    _logger.info(
        "chunked '%s' %s with '%s': %d chunk(s), mean %d characters",
        document_version.source_name,
        document_version.version_label,
        strategy_name,
        len(chunks),
        int(np.mean([len(chunk.text) for chunk in chunks])) if chunks else 0,
    )
    return chunks
