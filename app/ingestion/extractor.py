"""Text extraction from PDF, HTML and plain-text sources.

Page numbers are preserved for PDFs because a citation that says "page 14" is
verifiable by a human in seconds, and one that says "somewhere in this document"
is not. HTML has no pages, so those chunks cite the URL instead.
"""

from __future__ import annotations

from typing import Final

from bs4 import BeautifulSoup
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.exceptions import DocumentExtractionError, UnsupportedDocumentError
from app.core.logging_config import get_logger
from app.models.documents import ExtractedDocument, ExtractedPage, FetchedDocument
from app.utils.hashing import sha256_of_text
from app.utils.text import (
    extract_declared_effective_date,
    extract_declared_version,
    normalise_whitespace,
)

_logger = get_logger(__name__)

# Page furniture and interactive chrome carry no policy content but do add
# tokens, and worse, they retrieve well because they repeat on every page.
_NON_CONTENT_TAGS: Final = (
    "script", "style", "noscript", "nav", "header", "footer",
    "form", "iframe", "svg", "button", "aside",
)

_CONTENT_CONTAINER_SELECTORS: Final = (
    "main", "article", "[role=main]", "#main-content", ".main-content", ".content",
)


def _extract_pdf(fetched_document: FetchedDocument) -> list[ExtractedPage]:
    try:
        reader = PdfReader(str(fetched_document.raw_path))
    except (PdfReadError, OSError, ValueError) as pdf_error:
        raise DocumentExtractionError(
            f"could not read PDF for '{fetched_document.source.source_name}': {pdf_error}"
        ) from pdf_error

    pages: list[ExtractedPage] = []
    for page_index, pdf_page in enumerate(reader.pages, start=1):
        try:
            page_text = pdf_page.extract_text() or ""
        except Exception as page_error:
            # One malformed page must not discard an otherwise usable 80-page
            # circular; record the gap and keep going.
            _logger.warning(
                "page %d of '%s' could not be extracted: %s",
                page_index,
                fetched_document.source.source_name,
                page_error,
            )
            continue
        if page_text.strip():
            pages.append(ExtractedPage(page_number=page_index, text=page_text))
    return pages


def _extract_html(fetched_document: FetchedDocument) -> list[ExtractedPage]:
    markup = fetched_document.raw_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(markup, "lxml")

    for tag_name in _NON_CONTENT_TAGS:
        for element in soup.find_all(tag_name):
            element.decompose()

    # Prefer the semantic content container when the page offers one; falling
    # back to <body> keeps badly structured pages usable.
    content_root = None
    for selector in _CONTENT_CONTAINER_SELECTORS:
        content_root = soup.select_one(selector)
        if content_root is not None:
            break
    content_root = content_root or soup.body or soup

    page_text = content_root.get_text(separator="\n")
    return [ExtractedPage(page_number=None, text=page_text)] if page_text.strip() else []


def _extract_plain_text(fetched_document: FetchedDocument) -> list[ExtractedPage]:
    page_text = fetched_document.raw_path.read_text(encoding="utf-8", errors="replace")
    return [ExtractedPage(page_number=None, text=page_text)] if page_text.strip() else []


def extract_document(fetched_document: FetchedDocument) -> ExtractedDocument:
    """Turn a downloaded file into pages of normalised text plus its provenance."""
    content_type = fetched_document.content_type.split(";")[0].strip().lower()
    file_suffix = fetched_document.raw_path.suffix.lower()

    if content_type == "application/pdf" or file_suffix == ".pdf":
        pages = _extract_pdf(fetched_document)
    elif content_type in {"text/html", "application/xhtml+xml"} or file_suffix in {".html", ".htm"}:
        pages = _extract_html(fetched_document)
    elif content_type in {"text/plain", "text/markdown"} or file_suffix in {".txt", ".md"}:
        pages = _extract_plain_text(fetched_document)
    else:
        raise UnsupportedDocumentError(
            f"'{fetched_document.source.source_name}' has unsupported content type "
            f"'{fetched_document.content_type}'"
        )

    normalised_pages = [
        ExtractedPage(page_number=page.page_number, text=normalise_whitespace(page.text))
        for page in pages
    ]
    normalised_pages = [page for page in normalised_pages if page.text]

    if not normalised_pages:
        raise DocumentExtractionError(
            f"'{fetched_document.source.source_name}' produced no extractable text; "
            "it may be a scanned PDF requiring OCR"
        )

    combined_text = "\n\n".join(page.text for page in normalised_pages)
    extracted = ExtractedDocument(
        source_name=fetched_document.source.source_name,
        pages=normalised_pages,
        text_sha256=sha256_of_text(combined_text),
        declared_version=extract_declared_version(combined_text),
        declared_effective_date=extract_declared_effective_date(combined_text),
    )
    _logger.info(
        "extracted '%s': %d page(s), %d characters, declared version=%s, effective=%s",
        extracted.source_name,
        len(extracted.pages),
        extracted.character_count,
        extracted.declared_version,
        extracted.declared_effective_date,
    )
    return extracted
