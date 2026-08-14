"""Document, version and chunk contracts.

These types are the seam between ingestion and retrieval. Everything the answer
needs in order to cite itself — institution, title, version, page, URL — is
carried on the chunk, because a citation reconstructed later from a chunk id is
a citation that can drift from what was actually retrieved.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DocumentType = Literal["regulation", "credit_policy", "product_page"]
SourceAuthority = Literal["primary", "secondary"]

# Chroma stores only flat scalar metadata; anything else must be serialised.
ChromaScalar = str | int | float | bool


def build_version_id(source_name: str, version_number: int) -> str:
    """Stable identifier for one version of one source.

    A single metadata key holds it so that retrieval can filter to the active
    versions with one ``$in`` clause instead of a nested ``$or`` over pairs.
    """
    return f"{source_name}::v{version_number}"


class DocumentSource(BaseModel):
    """One entry from ``documents/source_registry.yaml``."""

    model_config = ConfigDict(extra="forbid")

    source_name: str = Field(min_length=1)
    url: str
    institution: str
    document_type: DocumentType
    document_title: str
    version: str | None = None
    effective_date: date | None = None
    last_checked: datetime | None = None
    enabled: bool = True
    authority: SourceAuthority = "secondary"

    @field_validator("source_name")
    @classmethod
    def _validate_source_name(cls, source_name: str) -> str:
        # The name becomes part of ids and metadata filters, so keep it boring.
        if not all(character.isalnum() or character in {"_", "-"} for character in source_name):
            raise ValueError(
                "source_name may contain only letters, digits, underscore and hyphen"
            )
        return source_name


class FetchedDocument(BaseModel):
    """A successfully downloaded source, already written to ``data/raw``."""

    model_config = ConfigDict(extra="forbid")

    source: DocumentSource
    raw_path: Path
    content_type: str
    content_sha256: str
    byte_size: int
    fetched_at: datetime
    # HTTP validators returned by the origin, stored so the next poll can send a
    # conditional request and skip the download entirely when nothing changed.
    etag: str | None = None
    last_modified: str | None = None


class ExtractedPage(BaseModel):
    """Extracted text for one page. ``page_number`` is None for HTML sources."""

    model_config = ConfigDict(extra="forbid")

    page_number: int | None
    text: str


class ExtractedDocument(BaseModel):
    """Cleaned text plus whatever provenance the document declared about itself."""

    model_config = ConfigDict(extra="forbid")

    source_name: str
    pages: list[ExtractedPage]
    text_sha256: str
    # Parsed out of the document body when present; more trustworthy than the
    # registry's hand-maintained fields, so it takes precedence.
    declared_version: str | None = None
    declared_effective_date: date | None = None

    @property
    def full_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text)

    @property
    def character_count(self) -> int:
        return sum(len(page.text) for page in self.pages)


class DocumentVersion(BaseModel):
    """An immutable record of one observed version of one source.

    Superseded versions are retained rather than overwritten: a loan decision
    made last year was made under the policy in force last year, and an audit
    has to be able to see it.
    """

    model_config = ConfigDict(extra="forbid")

    source_name: str
    version_number: int = Field(ge=1)
    version_id: str
    content_sha256: str
    text_sha256: str
    url: str
    institution: str
    document_title: str
    document_type: DocumentType
    authority: SourceAuthority
    declared_version: str | None = None
    effective_date: date | None = None
    first_seen_at: datetime
    last_checked_at: datetime
    is_active: bool
    raw_path: str
    character_count: int
    # chunk counts keyed by chunking strategy name, so one version can be
    # indexed under several strategies at once during experiments.
    chunk_counts: dict[str, int] = Field(default_factory=dict)

    @property
    def version_label(self) -> str:
        return f"Version {self.version_number}"


class ChunkMetadata(BaseModel):
    """Everything stored alongside a chunk vector in ChromaDB.

    Kept deliberately flat: Chroma's ``where`` filters only work on scalars, and
    a metadata field that cannot be filtered on is a field that cannot be used
    for version-aware retrieval.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    chunk_index: int
    source_name: str
    version_id: str
    version_number: int
    institution: str
    document_title: str
    document_type: str
    authority: str
    url: str
    effective_date: str | None = None
    page_number: int | None = None
    chunking_strategy: str
    embedding_model: str
    ingested_at: str
    character_count: int

    def to_chroma_metadata(self) -> dict[str, ChromaScalar]:
        """Flatten for Chroma, dropping None (Chroma rejects null metadata)."""
        return {
            key: value
            for key, value in self.model_dump().items()
            if value is not None
        }

    @classmethod
    def from_chroma_metadata(cls, raw_metadata: dict[str, Any]) -> ChunkMetadata:
        return cls.model_validate(raw_metadata)


class TextChunk(BaseModel):
    """A chunk of policy text ready to be embedded and stored."""

    model_config = ConfigDict(extra="forbid")

    text: str
    metadata: ChunkMetadata


class RetrievedChunk(BaseModel):
    """A chunk returned by the retriever, with its score and prompt position."""

    model_config = ConfigDict(extra="forbid")

    text: str
    metadata: ChunkMetadata
    similarity: float = Field(ge=-1.0, le=1.0)
    rank: int = Field(ge=1)

    @property
    def citation_label(self) -> str:
        """Short human-readable provenance string, e.g. for the UI and the prompt."""
        page_fragment = f", page {self.metadata.page_number}" if self.metadata.page_number else ""
        return (
            f"{self.metadata.document_title} "
            f"(Version {self.metadata.version_number}{page_fragment})"
        )
