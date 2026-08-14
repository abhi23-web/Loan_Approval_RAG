"""Document corpus endpoints: status and refresh."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends

from app.api.deps import get_document_service
from app.api.schemas import DocumentRefreshRequest, DocumentRefreshResponse
from app.core.logging_config import get_logger
from app.services.document_service import DocumentService

_logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("/status")
def read_document_status(
    document_service: DocumentService = Depends(get_document_service),
) -> dict:
    """Which sources are known, which version of each is active, how many chunks."""
    return asdict(document_service.status())


@router.post("/refresh", response_model=DocumentRefreshResponse)
def refresh_documents(
    request: DocumentRefreshRequest,
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentRefreshResponse:
    """Run ingestion now.

    Deliberately synchronous. Ingestion is idempotent and change-gated, so the
    common case returns in well under a second, and a caller who triggered a
    refresh should be told what happened rather than handed a job id. Continuous
    background refresh is the watcher process's job, not this endpoint's.
    """
    report = document_service.refresh(
        source_names=request.source_names,
        strategy_names=request.strategy_names,
        force_reindex=request.force_reindex,
    )
    return DocumentRefreshResponse(
        summary=report.summary(),
        duration_seconds=round(report.duration_seconds, 2),
        results=[asdict(result) for result in report.results],
    )
