"""Composition root.

Every collaborator is built exactly once, here, and handed to whoever needs it.
Nothing else in the application constructs a ChromaDB client, an embedding
provider or a rule engine, which is what keeps expensive objects from being
rebuilt per request and keeps the modules import-cycle free.

Construction is lazy. Importing this module must not open a database or contact
a model server — the test suite, the CLI scripts and the API all import it, and
only some of them need all of it.
"""

from __future__ import annotations

from functools import cached_property

from app.core.config import (
    ChunkingConfig,
    Settings,
    get_chunking_config,
    get_settings,
)
from app.core.logging_config import get_logger
from app.ingestion.embeddings import EmbeddingProvider, build_embedding_provider
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.registry import SourceRegistry
from app.ingestion.vector_store import ChromaVectorStore
from app.ingestion.versioning import VersionStore
from app.rag.generator import GroundedAnswerGenerator
from app.rag.llm import LLMProvider, build_llm_provider
from app.rag.pipeline import HomeLoanRagPipeline
from app.rag.retriever import PolicyRetriever
from app.rules.eligibility import EligibilityRuleEngine

_logger = get_logger(__name__)


class ApplicationContainer:
    """Lazily builds and holds the application's long-lived collaborators."""

    def __init__(
        self,
        settings: Settings | None = None,
        chunking_config: ChunkingConfig | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.chunking_config = chunking_config or get_chunking_config()
        self.settings.paths.create_all()

    # ------------------------------------------------------------ ingestion

    @cached_property
    def registry(self) -> SourceRegistry:
        return SourceRegistry.load()

    @cached_property
    def version_store(self) -> VersionStore:
        return VersionStore(self.settings.paths.metadata_dir)

    def fresh_version_store(self) -> VersionStore:
        """The version store, re-read if another process has updated it.

        Call this on the request path rather than touching ``version_store``
        directly, so a running API picks up a newly ingested policy version
        without a restart.
        """
        self.version_store.reload_if_changed()
        return self.version_store

    @cached_property
    def vector_store(self) -> ChromaVectorStore:
        return ChromaVectorStore(
            persist_directory=self.settings.paths.chroma_dir,
            collection_prefix=self.settings.vector_store.collection_prefix,
            distance=self.settings.vector_store.distance,
        )

    @cached_property
    def embedding_provider(self) -> EmbeddingProvider:
        return build_embedding_provider(self.settings.embeddings)

    @cached_property
    def ingestion_pipeline(self) -> IngestionPipeline:
        return IngestionPipeline(
            settings=self.settings,
            chunking_config=self.chunking_config,
            registry=self.registry,
            version_store=self.version_store,
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider,
        )

    # ------------------------------------------------------------------ rag

    @cached_property
    def llm_provider(self) -> LLMProvider:
        return build_llm_provider(self.settings.llm)

    @cached_property
    def retriever(self) -> PolicyRetriever:
        return PolicyRetriever(
            settings=self.settings.retrieval,
            vector_store=self.vector_store,
            version_store=self.version_store,
            embedding_provider=self.embedding_provider,
        )

    @cached_property
    def generator(self) -> GroundedAnswerGenerator:
        return GroundedAnswerGenerator(self.llm_provider)

    @cached_property
    def rule_engine(self) -> EligibilityRuleEngine:
        return EligibilityRuleEngine(self.settings.rules)

    @cached_property
    def rag_pipeline(self) -> HomeLoanRagPipeline:
        return HomeLoanRagPipeline(
            settings=self.settings,
            retriever=self.retriever,
            generator=self.generator,
            rule_engine=self.rule_engine,
        )


_container: ApplicationContainer | None = None


def get_container() -> ApplicationContainer:
    """Process-wide container.

    A module-level singleton rather than a FastAPI dependency so that the CLI
    scripts, the watcher and the evaluation harness all share the same wiring as
    the API without importing FastAPI.
    """
    global _container
    if _container is None:
        _container = ApplicationContainer()
        _logger.info("application container initialised")
    return _container


def reset_container() -> None:
    """Drop the singleton. Used by tests that need a different configuration."""
    global _container
    _container = None
