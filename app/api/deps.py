"""FastAPI dependencies.

Thin adapters over the composition root. Keeping them here means the routes
depend on services, and the services depend on nothing from FastAPI.
"""

from __future__ import annotations

from app.services.assessment_service import AssessmentService
from app.services.container import ApplicationContainer, get_container
from app.services.document_service import DocumentService


def get_application_container() -> ApplicationContainer:
    return get_container()


def get_assessment_service() -> AssessmentService:
    return AssessmentService(get_container())


def get_document_service() -> DocumentService:
    return DocumentService(get_container())
