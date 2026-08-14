"""Post-extraction cleaning.

The single highest-value cleaning step for this corpus is removing repeated
running headers and footers from PDFs. They appear on every page, so they are
the most frequent text in the document, they embed well, and they crowd out real
clauses in the top-k. Removing them is a retrieval-quality change, not cosmetics.
"""

from __future__ import annotations

import re
from collections import Counter

from app.core.logging_config import get_logger
from app.models.documents import ExtractedDocument, ExtractedPage
from app.utils.hashing import sha256_of_text

_logger = get_logger(__name__)

# A line has to appear on this share of pages before it counts as furniture.
_REPEATED_LINE_PAGE_SHARE = 0.6
# Below this many pages the statistic is meaningless and would delete content.
_MIN_PAGES_FOR_REPEAT_DETECTION = 4
# Long lines are prose, not headers, however often they repeat.
_MAX_FURNITURE_LINE_LENGTH = 90

_PAGE_NUMBER_LINE = re.compile(r"^(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?$", re.IGNORECASE)

_BOILERPLATE_MARKERS = (
    "cookie", "javascript is disabled", "skip to main content",
    "all rights reserved", "terms and conditions apply", "©",
    "sign in", "log in", "download the app",
)


def _find_repeated_furniture(pages: list[ExtractedPage]) -> set[str]:
    if len(pages) < _MIN_PAGES_FOR_REPEAT_DETECTION:
        return set()

    pages_containing_line: Counter[str] = Counter()
    for page in pages:
        # Count each distinct line once per page: a line repeated within one page
        # is emphasis, not furniture.
        for line in {stripped for stripped in (raw.strip() for raw in page.text.split("\n")) if stripped}:
            pages_containing_line[line] += 1

    threshold = max(2, int(len(pages) * _REPEATED_LINE_PAGE_SHARE))
    return {
        line
        for line, page_count in pages_containing_line.items()
        if page_count >= threshold and len(line) <= _MAX_FURNITURE_LINE_LENGTH
    }


def _is_noise_line(line: str, furniture_lines: set[str]) -> bool:
    if not line:
        return True
    if line in furniture_lines:
        return True
    if _PAGE_NUMBER_LINE.match(line):
        return True
    lowered = line.lower()
    return any(marker in lowered for marker in _BOILERPLATE_MARKERS) and len(line) < 160


def clean_document(extracted_document: ExtractedDocument) -> ExtractedDocument:
    """Strip running headers, page numbers and web boilerplate.

    Returns a new document; the input is left untouched so the raw extraction
    stays available for debugging a cleaning bug.
    """
    furniture_lines = _find_repeated_furniture(extracted_document.pages)

    cleaned_pages: list[ExtractedPage] = []
    removed_line_count = 0
    for page in extracted_document.pages:
        kept_lines: list[str] = []
        for raw_line in page.text.split("\n"):
            line = raw_line.strip()
            if _is_noise_line(line, furniture_lines):
                removed_line_count += 1 if line else 0
                continue
            kept_lines.append(line)
        cleaned_text = "\n".join(kept_lines).strip()
        if cleaned_text:
            cleaned_pages.append(ExtractedPage(page_number=page.page_number, text=cleaned_text))

    if not cleaned_pages:
        # Cleaning that empties a document is a cleaning bug, not a clean document.
        _logger.warning(
            "cleaning removed all content from '%s'; keeping the uncleaned text",
            extracted_document.source_name,
        )
        return extracted_document

    combined_text = "\n\n".join(page.text for page in cleaned_pages)
    _logger.info(
        "cleaned '%s': removed %d noise line(s), %d furniture pattern(s)",
        extracted_document.source_name,
        removed_line_count,
        len(furniture_lines),
    )
    return ExtractedDocument(
        source_name=extracted_document.source_name,
        pages=cleaned_pages,
        text_sha256=sha256_of_text(combined_text),
        declared_version=extracted_document.declared_version,
        declared_effective_date=extracted_document.declared_effective_date,
    )
