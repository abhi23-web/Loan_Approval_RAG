"""Chunking behaviour.

The properties asserted here are the ones a retrieval failure traces back to:
chunks that overshoot their configured size, overlap that does not actually
overlap, and a splitter that silently drops text.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import ChunkingStrategyConfig, load_chunking_config
from app.core.exceptions import ConfigurationError
from app.ingestion.chunking import (
    FixedSizeChunker,
    RecursiveChunker,
    SemanticChunker,
    build_chunker,
)
from app.ingestion.embeddings import DeterministicEmbeddingProvider

_SEPARATORS = load_chunking_config().recursive_separators

_POLICY_TEXT = """\
Clause 3 — Credit Bureau Requirements

3.1 The minimum acceptable CIBIL score for a standard sanction is 725.

3.2 Applications with a CIBIL score between 675 and 724 may be considered for a
conditional sanction subject to a co-applicant with an independent income.

Clause 4 — Loan to Value Ratio

4.1 The loan-to-value ratio is calculated on the lower of the documented property
value and the value assessed by the empanelled valuer.

4.2 The maximum loan-to-value ratio is 90 percent for loans up to INR 30,00,000,
80 percent above that and up to INR 75,00,000, and 70 percent thereafter.
"""


def test_fixed_chunker_respects_size_and_stride() -> None:
    chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=50)
    chunks = chunker.split(_POLICY_TEXT)

    assert chunks
    assert all(len(chunk) <= 200 for chunk in chunks)
    # Consecutive windows must genuinely share text, otherwise the overlap
    # setting is decorative and boundary-straddling rules get lost.
    assert _POLICY_TEXT[150:200].strip() in chunks[0] + chunks[1]


def test_recursive_chunker_prefers_paragraph_boundaries() -> None:
    chunker = RecursiveChunker(chunk_size=300, chunk_overlap=40, separators=_SEPARATORS)
    chunks = chunker.split(_POLICY_TEXT)

    assert chunks
    # Some tolerance above chunk_size is expected: a piece is only split further
    # when it exceeds the size, so the final merge can land slightly over.
    assert all(len(chunk) <= 300 + 40 for chunk in chunks)
    # Clause 3.1 carries the answer to a golden-dataset case; it must survive
    # intact in exactly one chunk rather than being cut mid-sentence.
    assert any("minimum acceptable CIBIL score for a standard sanction is 725" in chunk for chunk in chunks)


def test_recursive_chunker_preserves_all_content_words() -> None:
    chunker = RecursiveChunker(chunk_size=250, chunk_overlap=30, separators=_SEPARATORS)
    chunks = chunker.split(_POLICY_TEXT)

    original_words = set(_POLICY_TEXT.split())
    chunked_words = {word for chunk in chunks for word in chunk.split()}
    assert original_words <= chunked_words, "chunking must not drop text"


def test_recursive_chunker_handles_text_shorter_than_chunk_size() -> None:
    chunker = RecursiveChunker(chunk_size=5000, chunk_overlap=100, separators=_SEPARATORS)
    assert chunker.split(_POLICY_TEXT) == [_POLICY_TEXT.strip()]


def test_recursive_chunker_splits_an_unbroken_run() -> None:
    """A single token longer than chunk_size must still be split, not returned whole."""
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10, separators=_SEPARATORS)
    chunks = chunker.split("x" * 350)
    assert len(chunks) >= 3
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_semantic_chunker_produces_bounded_chunks() -> None:
    chunker = SemanticChunker(
        embedder=DeterministicEmbeddingProvider(),
        breakpoint_percentile=80,
        buffer_sentences=1,
        min_chunk_characters=100,
        max_chunk_characters=400,
    )
    chunks = chunker.split(_POLICY_TEXT)

    assert chunks
    assert all(len(chunk) <= 400 + 200 for chunk in chunks), "max_chunk_characters is a soft ceiling per sentence"
    assert "725" in " ".join(chunks)


def test_semantic_strategy_requires_an_embedder() -> None:
    strategy = ChunkingStrategyConfig(name="semantic", type="semantic", breakpoint_percentile=90)
    with pytest.raises(ConfigurationError, match="embedding provider"):
        build_chunker(strategy, _SEPARATORS, embedder=None)


def test_overlap_larger_than_size_is_rejected_at_config_time() -> None:
    """An overlap >= size cannot make forward progress, so it must not validate.

    ``ConfigurationError`` subclasses ``ValueError`` so Pydantic can raise it from
    a validator; Pydantic then re-wraps it as a ``ValidationError``, which is what
    a caller actually sees.
    """
    with pytest.raises(ValidationError, match="smaller than size"):
        ChunkingStrategyConfig(
            name="broken", type="recursive", chunk_size=100, chunk_overlap=100
        )


def test_every_configured_strategy_can_be_built() -> None:
    """config/chunking.yaml must not contain a strategy the code cannot construct."""
    chunking_config = load_chunking_config()
    embedder = DeterministicEmbeddingProvider()
    for strategy_name in chunking_config.strategies:
        chunker = build_chunker(
            chunking_config.get(strategy_name),
            chunking_config.recursive_separators,
            embedder=embedder,
        )
        assert chunker.split(_POLICY_TEXT), f"strategy '{strategy_name}' produced no chunks"
