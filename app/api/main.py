"""FastAPI application factory.

Routes handle HTTP and nothing else; every domain exception is translated to a
status code in one place here, so a new route cannot accidentally leak a stack
trace or return a 500 for a user error.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import assessment, documents, health
from app.core.config import get_settings
from app.core.exceptions import (
    ConfigurationError,
    DocumentFetchError,
    EmbeddingError,
    HomeLoanRagError,
    KnowledgeBaseEmptyError,
    LLMError,
    VectorStoreError,
)
from app.core.logging_config import configure_logging, get_logger
from app.core.tracing import configure_tracing
from app.services.container import get_container

_logger = get_logger(__name__)

API_PREFIX = "/api/v1"

# Domain exception -> HTTP status. 503 for "a dependency is not ready", 500 only
# for genuine internal faults.
_STATUS_BY_EXCEPTION: list[tuple[type[Exception], int]] = [
    (ConfigurationError, 500),
    (KnowledgeBaseEmptyError, 503),
    (EmbeddingError, 503),
    (LLMError, 503),
    (DocumentFetchError, 502),
    (VectorStoreError, 500),
]


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Warm the container once at startup rather than on the first request.

    ChromaDB opens its on-disk index during construction. Doing that inside the
    first request would make one unlucky applicant pay for it and would make the
    first latency measurement meaningless.
    """
    load_dotenv()
    settings = get_settings()
    configure_logging(settings.app.log_level)
    configure_tracing(settings.observability.langsmith_project)

    container = get_container()
    active_strategy = settings.retrieval.active_strategy
    try:
        indexed_chunks = container.vector_store.count(active_strategy)
        _logger.info(
            "vector store ready: strategy='%s', %d chunk(s) indexed",
            active_strategy,
            indexed_chunks,
        )
        if indexed_chunks == 0:
            _logger.warning(
                "the index is empty; run 'python scripts/ingest.py' before assessing"
            )
    except Exception as startup_error:
        # A broken index must not stop the process: /health has to stay
        # reachable to report why.
        _logger.error("vector store unavailable at startup: %s", startup_error)

    yield
    _logger.info("shutting down")


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title="Home Loan Approval Automation API",
        description=(
            "Deterministic eligibility rules with retrieval-augmented, "
            "version-aware policy grounding."
        ),
        version="1.0.0",
        lifespan=_lifespan,
    )

    # The Streamlit frontend is a separate origin in local development.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    application.include_router(health.router, prefix=API_PREFIX)
    application.include_router(documents.router, prefix=API_PREFIX)
    application.include_router(assessment.router, prefix=API_PREFIX)

    @application.exception_handler(HomeLoanRagError)
    async def handle_domain_error(_: Request, error: HomeLoanRagError) -> JSONResponse:
        status_code = next(
            (
                status
                for exception_type, status in _STATUS_BY_EXCEPTION
                if isinstance(error, exception_type)
            ),
            500,
        )
        _logger.error("%s -> HTTP %d: %s", type(error).__name__, status_code, error)
        return JSONResponse(
            status_code=status_code,
            content={"error": type(error).__name__, "detail": str(error)},
        )

    @application.get("/", include_in_schema=False)
    async def read_root() -> dict[str, str]:
        return {
            "service": settings.app.name,
            "docs": "/docs",
            "health": f"{API_PREFIX}/health",
        }

    return application


app = create_app()
