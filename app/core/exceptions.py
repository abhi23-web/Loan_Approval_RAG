"""Domain exceptions.

Every failure mode in this system has a name. A bare ``except Exception: pass``
hides the two failures that actually matter here — a policy document that could
not be fetched, and a retrieval that returned nothing — and both of those must
surface to the caller rather than quietly degrade into a confident wrong answer.
"""

from __future__ import annotations


class HomeLoanRagError(Exception):
    """Base class for every error raised by this application."""


class ConfigurationError(HomeLoanRagError, ValueError):
    """Configuration is missing, malformed or internally inconsistent.

    Subclasses ``ValueError`` so Pydantic validators can raise it directly and
    still produce a normal validation error.
    """


class DocumentFetchError(HomeLoanRagError):
    """A registered source could not be downloaded."""

    def __init__(self, source_name: str, url: str, reason: str) -> None:
        super().__init__(f"failed to fetch '{source_name}' from {url}: {reason}")
        self.source_name = source_name
        self.url = url
        self.reason = reason


class UnsupportedDocumentError(HomeLoanRagError):
    """The document was fetched but its content type cannot be parsed."""


class DocumentExtractionError(HomeLoanRagError):
    """The document was fetched and recognised but yielded no usable text."""


class EmbeddingError(HomeLoanRagError):
    """The embedding backend failed or returned a malformed response."""


class LLMError(HomeLoanRagError):
    """The generation backend failed or returned a malformed response."""


class VectorStoreError(HomeLoanRagError):
    """ChromaDB rejected an operation or is in an unusable state."""


class KnowledgeBaseEmptyError(HomeLoanRagError):
    """Retrieval was attempted before any document was ingested.

    Kept distinct from "retrieved nothing relevant": an empty index is an
    operator error, whereas no relevant chunk is a legitimate answer.
    """


class EvaluationError(HomeLoanRagError):
    """The evaluation harness could not run or score a case."""
