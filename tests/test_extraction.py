"""Fetching, extraction and cleaning."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.core.exceptions import DocumentFetchError, UnsupportedDocumentError
from app.ingestion.cleaner import clean_document
from app.ingestion.extractor import extract_document
from app.ingestion.fetcher import DocumentFetcher
from app.models.documents import DocumentSource, ExtractedDocument, ExtractedPage, FetchedDocument
from app.utils.hashing import sha256_of_text
from app.utils.text import extract_declared_effective_date, extract_declared_version

_HTML_PAGE = """
<html><head><title>Home Loan</title><style>.a{color:red}</style></head>
<body>
  <nav>Skip to main content</nav>
  <main>
    <h1>Home Loan Eligibility</h1>
    <p>The minimum credit score is 725.</p>
  </main>
  <footer>All rights reserved</footer>
  <script>console.log('tracking')</script>
</body></html>
"""


def _fetched(path: Path, content_type: str) -> FetchedDocument:
    return FetchedDocument(
        source=DocumentSource(
            source_name="fixture",
            url=f"file://{path}",
            institution="Fixture",
            document_type="credit_policy",
            document_title="Fixture",
        ),
        raw_path=path,
        content_type=content_type,
        content_sha256="x" * 64,
        byte_size=path.stat().st_size,
        fetched_at=datetime.now(UTC),
    )


def test_html_extraction_drops_chrome_and_keeps_content(tmp_path: Path) -> None:
    html_path = tmp_path / "page.html"
    html_path.write_text(_HTML_PAGE, encoding="utf-8")

    extracted = extract_document(_fetched(html_path, "text/html"))
    combined = extracted.full_text

    assert "minimum credit score is 725" in combined
    assert "tracking" not in combined, "script content must be removed"
    assert "All rights reserved" not in combined, "footer must be removed"
    assert extracted.pages[0].page_number is None, "HTML has no page numbers"


def test_markdown_extraction_reads_declared_provenance(tmp_path: Path) -> None:
    markdown_path = tmp_path / "policy.md"
    markdown_path.write_text(
        "# Policy\n\nDocument Version: 3\nEffective Date: 2026-01-01\n\n"
        "3.1 The minimum acceptable CIBIL score is 725.\n",
        encoding="utf-8",
    )

    extracted = extract_document(_fetched(markdown_path, "text/markdown"))
    assert extracted.declared_version == "3"
    assert extracted.declared_effective_date is not None
    assert extracted.declared_effective_date.year == 2026


def test_unsupported_content_type_is_rejected(tmp_path: Path) -> None:
    binary_path = tmp_path / "thing.bin"
    binary_path.write_bytes(b"\x00\x01\x02")
    with pytest.raises(UnsupportedDocumentError):
        extract_document(_fetched(binary_path, "application/octet-stream"))


def test_cleaner_removes_repeated_running_headers() -> None:
    """Furniture repeated on most pages must go; unique content must stay."""
    pages = [
        ExtractedPage(
            page_number=page_number,
            text=(
                "Meridian Housing Finance Limited\n"
                f"Clause {page_number} states a unique requirement.\n"
                f"{page_number}"
            ),
        )
        for page_number in range(1, 7)
    ]
    document = ExtractedDocument(
        source_name="fixture", pages=pages, text_sha256=sha256_of_text("x")
    )

    cleaned = clean_document(document)
    combined = cleaned.full_text

    assert "Meridian Housing Finance Limited" not in combined
    assert "Clause 3 states a unique requirement." in combined


def test_cleaner_leaves_short_documents_alone() -> None:
    """With too few pages the repetition statistic would delete real content."""
    pages = [ExtractedPage(page_number=1, text="Meridian\nClause 1 applies.")]
    document = ExtractedDocument(
        source_name="fixture", pages=pages, text_sha256=sha256_of_text("y")
    )
    assert "Meridian" in clean_document(document).full_text


def test_fetcher_rejects_a_disallowed_scheme(tmp_path: Path) -> None:
    """The registry is configuration, not a trusted source of URLs."""
    from app.core.config import get_settings

    fetcher = DocumentFetcher(get_settings().ingestion, tmp_path)
    source = DocumentSource(
        source_name="bad",
        url="ftp://example.invalid/policy.pdf",
        institution="Nowhere",
        document_type="credit_policy",
        document_title="Bad",
    )
    with pytest.raises(DocumentFetchError, match="allowed_schemes"):
        fetcher.fetch(source)


def test_fetcher_rejects_a_file_url_outside_the_repository(tmp_path: Path) -> None:
    from app.core.config import get_settings

    fetcher = DocumentFetcher(get_settings().ingestion, tmp_path)
    source = DocumentSource(
        source_name="escape",
        url="file:///etc/passwd",
        institution="Nowhere",
        document_type="credit_policy",
        document_title="Escape",
    )
    with pytest.raises(DocumentFetchError, match="inside the repository"):
        fetcher.fetch(source)


def test_fetcher_reports_unchanged_content_by_hash(tmp_path: Path) -> None:
    from app.core.config import PROJECT_ROOT, get_settings

    fetcher = DocumentFetcher(get_settings().ingestion, tmp_path)
    source = DocumentSource(
        source_name="meridian_home_loan_policy",
        url="file://documents/local_policies/current/meridian_home_loan_policy.md",
        institution="Meridian",
        document_type="credit_policy",
        document_title="Meridian",
    )
    assert (PROJECT_ROOT / "documents/local_policies/current/meridian_home_loan_policy.md").exists()

    first = fetcher.fetch(source)
    assert first.changed and first.document is not None

    second = fetcher.fetch(source, previous_content_sha256=first.document.content_sha256)
    assert second.changed is False
    assert second.reason == "identical_bytes"


def test_declared_version_patterns() -> None:
    assert extract_declared_version("Document Version: 2\n") == "2"
    assert extract_declared_version("no version statement here") is None
    assert extract_declared_effective_date("Effective Date: 2024-07-01") is not None
