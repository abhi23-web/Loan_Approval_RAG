"""The ingestion pipeline.

fetch -> extract -> clean -> detect change -> version -> chunk -> embed -> store

Every stage is a separate, individually testable function elsewhere in this
package; this module only sequences them and decides what work can be skipped.
That decision is the point of the module: on a normal poll, almost every source
should reach the "unchanged" branch and cost nothing but one conditional request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from app.core.config import ChunkingConfig, Settings
from app.core.exceptions import (
    DocumentExtractionError,
    DocumentFetchError,
    HomeLoanRagError,
    UnsupportedDocumentError,
)
from app.core.logging_config import get_logger
from app.core.tracing import add_trace_metadata, traced
from app.ingestion.chunking import build_chunker, chunk_document
from app.ingestion.cleaner import clean_document
from app.ingestion.embeddings import EmbeddingProvider
from app.ingestion.extractor import extract_document
from app.ingestion.fetcher import DocumentFetcher
from app.ingestion.registry import SourceRegistry
from app.ingestion.vector_store import ChromaVectorStore
from app.ingestion.versioning import VersionStore
from app.models.documents import DocumentSource, DocumentVersion

_logger = get_logger(__name__)

SourceOutcome = Literal["indexed", "unchanged", "failed", "skipped"]


@dataclass
class SourceIngestionResult:
    source_name: str
    outcome: SourceOutcome
    detail: str
    version_number: int | None = None
    chunks_written: dict[str, int] = field(default_factory=dict)


@dataclass
class IngestionReport:
    started_at: datetime
    finished_at: datetime
    results: list[SourceIngestionResult]

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()

    def count_with_outcome(self, outcome: SourceOutcome) -> int:
        return sum(1 for result in self.results if result.outcome == outcome)

    def summary(self) -> str:
        return (
            f"{self.count_with_outcome('indexed')} indexed, "
            f"{self.count_with_outcome('unchanged')} unchanged, "
            f"{self.count_with_outcome('skipped')} skipped, "
            f"{self.count_with_outcome('failed')} failed "
            f"in {self.duration_seconds:.1f}s"
        )

    @property
    def has_changes(self) -> bool:
        return self.count_with_outcome("indexed") > 0


class IngestionPipeline:
    """Sequences the ingestion stages and skips work that is provably unnecessary."""

    def __init__(
        self,
        settings: Settings,
        chunking_config: ChunkingConfig,
        registry: SourceRegistry,
        version_store: VersionStore,
        vector_store: ChromaVectorStore,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._settings = settings
        self._chunking_config = chunking_config
        self._registry = registry
        self._version_store = version_store
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider
        self._fetcher = DocumentFetcher(settings.ingestion, settings.paths.raw_dir)

    @traced("ingestion.run", run_type="chain")
    def run(
        self,
        *,
        source_names: list[str] | None = None,
        strategy_names: list[str] | None = None,
        force_reindex: bool = False,
    ) -> IngestionReport:
        """Ingest the selected sources into the selected chunking strategies.

        ``force_reindex`` bypasses change detection. It exists for one legitimate
        case — the chunker, the embedding model or the cleaner changed, so the
        stored vectors are stale even though the documents are not.
        """
        started_at = datetime.now(UTC)
        strategies = strategy_names or [self._settings.chunking.active_strategy]
        for strategy_name in strategies:
            self._chunking_config.get(strategy_name)  # fail fast on a typo

        selected_sources = (
            [self._registry.get(name) for name in source_names]
            if source_names
            else self._registry.enabled_sources()
        )

        results: list[SourceIngestionResult] = []
        for source in selected_sources:
            if not source.enabled and source_names is None:
                results.append(
                    SourceIngestionResult(source.source_name, "skipped", "disabled in registry")
                )
                continue
            results.append(self._ingest_one_source(source, strategies, force_reindex))

        self._version_store.save()
        report = IngestionReport(started_at, datetime.now(UTC), results)
        add_trace_metadata(
            strategies=strategies,
            sources=len(selected_sources),
            summary=report.summary(),
        )
        _logger.info("ingestion finished: %s", report.summary())
        return report

    @traced("ingestion.source", run_type="chain")
    def _ingest_one_source(
        self, source: DocumentSource, strategy_names: list[str], force_reindex: bool
    ) -> SourceIngestionResult:
        history = self._version_store.history_for(source.source_name)
        latest_version = self._version_store.latest_version(source.source_name)

        try:
            fetch_result = self._fetcher.fetch(
                source,
                previous_content_sha256=(
                    None if force_reindex else (latest_version.content_sha256 if latest_version else None)
                ),
                etag=None if force_reindex else history.etag,
                last_modified=None if force_reindex else history.last_modified,
            )
        except (DocumentFetchError, HomeLoanRagError) as fetch_error:
            # One unreachable source must not abort the whole run: the other
            # policies are still worth having.
            _logger.error("fetch failed for '%s': %s", source.source_name, fetch_error)
            self._version_store.record_check(source.source_name)
            return SourceIngestionResult(source.source_name, "failed", str(fetch_error))

        if not fetch_result.changed or fetch_result.document is None:
            self._version_store.record_check(source.source_name)
            return SourceIngestionResult(
                source.source_name, "unchanged", fetch_result.reason,
                version_number=latest_version.version_number if latest_version else None,
            )

        fetched_document = fetch_result.document
        self._version_store.record_check(
            source.source_name,
            checked_at=fetched_document.fetched_at,
            etag=fetched_document.etag,
            last_modified=fetched_document.last_modified,
        )

        try:
            extracted_document = clean_document(extract_document(fetched_document))
        except (DocumentExtractionError, UnsupportedDocumentError) as extraction_error:
            _logger.error("extraction failed for '%s': %s", source.source_name, extraction_error)
            return SourceIngestionResult(source.source_name, "failed", str(extraction_error))

        # Third and final change gate: the bytes moved, but did the policy text?
        # Marketing pages regenerate constantly with identical wording.
        existing_version = self._version_store.find_by_text_hash(
            source.source_name, extracted_document.text_sha256
        )
        if existing_version is not None and not force_reindex:
            _logger.info(
                "'%s' has new bytes but identical text; keeping %s",
                source.source_name,
                existing_version.version_label,
            )
            return SourceIngestionResult(
                source.source_name, "unchanged", "identical_text",
                version_number=existing_version.version_number,
            )

        document_version = existing_version or self._version_store.register_version(
            source,
            content_sha256=fetched_document.content_sha256,
            text_sha256=extracted_document.text_sha256,
            raw_path=fetched_document.raw_path,
            character_count=extracted_document.character_count,
            declared_version=extracted_document.declared_version,
            effective_date=extracted_document.declared_effective_date or source.effective_date,
            observed_at=fetched_document.fetched_at,
        )

        chunks_written = self._index_version(extracted_document, document_version, strategy_names)
        return SourceIngestionResult(
            source_name=source.source_name,
            outcome="indexed",
            detail=f"indexed {document_version.version_label}",
            version_number=document_version.version_number,
            chunks_written=chunks_written,
        )

    @traced("ingestion.index_version", run_type="chain")
    def _index_version(
        self,
        extracted_document,  # noqa: ANN001 - ExtractedDocument, kept untyped to avoid a cycle in docs
        document_version: DocumentVersion,
        strategy_names: list[str],
    ) -> dict[str, int]:
        """Chunk, embed and store one version under each requested strategy."""
        chunks_written: dict[str, int] = {}

        for strategy_name in strategy_names:
            strategy = self._chunking_config.get(strategy_name)
            chunker = build_chunker(
                strategy,
                self._chunking_config.recursive_separators,
                embedder=self._embedding_provider if strategy.type == "semantic" else None,
            )
            chunks = chunk_document(
                extracted_document,
                document_version,
                chunker,
                strategy_name=strategy_name,
                embedding_model=self._embedding_provider.model_name,
            )
            if not chunks:
                _logger.warning(
                    "strategy '%s' produced no chunks for '%s'",
                    strategy_name,
                    document_version.source_name,
                )
                continue

            embeddings = self._embedding_provider.embed_documents([chunk.text for chunk in chunks])
            written = self._vector_store.upsert_chunks(strategy_name, chunks, embeddings)
            self._version_store.record_chunk_count(document_version.version_id, strategy_name, written)
            chunks_written[strategy_name] = written

        return chunks_written
