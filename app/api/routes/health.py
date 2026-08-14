"""Health endpoint.

Reports what the process is actually configured with, not just "ok". A green
health check that hides "the index is empty" or "the offline stub provider is
active" is worse than no health check, so both appear as explicit warnings.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends

from app.api.deps import get_application_container
from app.api.schemas import HealthResponse
from app.core.tracing import tracing_enabled
from app.services.container import ApplicationContainer

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def read_health(
    container: ApplicationContainer = Depends(get_application_container),
) -> HealthResponse:
    settings = container.settings
    active_strategy = settings.retrieval.active_strategy

    warnings: list[str] = []
    try:
        indexed_chunk_count = container.vector_store.count(active_strategy)
    except Exception as vector_store_error:
        indexed_chunk_count = -1
        warnings.append(f"vector store unavailable: {vector_store_error}")

    if indexed_chunk_count == 0:
        warnings.append(
            f"no chunks indexed for strategy '{active_strategy}'; run scripts/ingest.py"
        )
    if settings.llm.provider == "deterministic":
        warnings.append("LLM provider is the offline stub; answers are not real")
    if settings.embeddings.provider == "deterministic":
        warnings.append("embedding provider is the offline test double; retrieval is lexical only")
    if not tracing_enabled():
        warnings.append("LangSmith tracing is off (no LANGSMITH_API_KEY)")

    return HealthResponse(
        status="degraded" if warnings else "ok",
        checked_at=datetime.now(UTC),
        application=settings.app.name,
        environment=settings.app.environment,
        active_chunking_strategy=active_strategy,
        embedding_provider=settings.embeddings.provider,
        # The provider's own model name, not the configured one: when the offline
        # double is active they differ, and reporting the configured name would
        # make a stubbed process look like a real one.
        embedding_model=container.embedding_provider.model_name,
        llm_provider=settings.llm.provider,
        llm_model=container.llm_provider.model_name,
        indexed_chunk_count=indexed_chunk_count,
        active_source_count=len(container.registry.enabled_sources()),
        langsmith_tracing=tracing_enabled(),
        warnings=warnings,
    )
