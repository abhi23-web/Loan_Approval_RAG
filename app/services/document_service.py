"""Document service: ingestion triggers and corpus status."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from app.core.logging_config import get_logger
from app.ingestion.pipeline import IngestionReport
from app.models.documents import DocumentVersion
from app.services.container import ApplicationContainer

_logger = get_logger(__name__)


@dataclass
class SourceStatus:
    """What the system currently knows about one registered source."""

    source_name: str
    institution: str
    document_title: str
    url: str
    enabled: bool
    document_type: str
    authority: str
    last_checked_at: datetime | None
    active_version_number: int | None
    active_effective_date: str | None
    total_versions: int
    chunk_counts_by_strategy: dict[str, int] = field(default_factory=dict)
    version_history: list[dict[str, object]] = field(default_factory=list)


@dataclass
class CorpusStatus:
    active_chunking_strategy: str
    embedding_model: str
    indexed_strategies: list[str]
    chunk_count_active_strategy: int
    sources: list[SourceStatus]


class DocumentService:
    """Read and refresh the policy corpus."""

    def __init__(self, container: ApplicationContainer) -> None:
        self._container = container

    def refresh(
        self,
        *,
        source_names: list[str] | None = None,
        strategy_names: list[str] | None = None,
        force_reindex: bool = False,
    ) -> IngestionReport:
        return self._container.ingestion_pipeline.run(
            source_names=source_names,
            strategy_names=strategy_names,
            force_reindex=force_reindex,
        )

    def status(self) -> CorpusStatus:
        version_store = self._container.fresh_version_store()
        settings = self._container.settings
        active_strategy = settings.retrieval.active_strategy

        source_statuses: list[SourceStatus] = []
        for source in self._container.registry.all_sources():
            history = version_store.history_for(source.source_name)
            active_version = version_store.active_version(source.source_name)
            source_statuses.append(
                SourceStatus(
                    source_name=source.source_name,
                    institution=source.institution,
                    document_title=source.document_title,
                    url=source.url,
                    enabled=source.enabled,
                    document_type=source.document_type,
                    authority=source.authority,
                    last_checked_at=history.last_checked_at,
                    active_version_number=(
                        active_version.version_number if active_version else None
                    ),
                    active_effective_date=(
                        active_version.effective_date.isoformat()
                        if active_version and active_version.effective_date
                        else None
                    ),
                    total_versions=len(history.versions),
                    chunk_counts_by_strategy=(
                        dict(active_version.chunk_counts) if active_version else {}
                    ),
                    version_history=[
                        self._describe_version(version) for version in history.versions
                    ],
                )
            )

        return CorpusStatus(
            active_chunking_strategy=active_strategy,
            embedding_model=self._container.embedding_provider.model_name,
            indexed_strategies=self._container.vector_store.indexed_strategies(),
            chunk_count_active_strategy=self._container.vector_store.count(active_strategy),
            sources=source_statuses,
        )

    @staticmethod
    def _describe_version(version: DocumentVersion) -> dict[str, object]:
        return {
            "version_number": version.version_number,
            "version_id": version.version_id,
            "is_active": version.is_active,
            "effective_date": (
                version.effective_date.isoformat() if version.effective_date else None
            ),
            "declared_version": version.declared_version,
            "first_seen_at": version.first_seen_at.isoformat(),
            "character_count": version.character_count,
            "chunk_counts": dict(version.chunk_counts),
        }
