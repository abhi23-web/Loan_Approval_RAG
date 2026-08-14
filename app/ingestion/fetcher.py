"""Document fetching with change detection.

Three layers of "do not do unnecessary work", cheapest first:

1. A conditional HTTP request using the stored ETag / Last-Modified. A 304 costs
   one round trip and no download at all.
2. A byte hash comparison against the newest stored version, which catches
   servers that ignore conditional requests.
3. A text hash comparison, done later in the pipeline after extraction, which
   catches pages whose bytes churn (build ids, CSRF tokens) but whose policy text
   is identical.

Only when all three say "changed" does anything get embedded.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import PROJECT_ROOT, IngestionSection
from app.core.exceptions import DocumentFetchError
from app.core.logging_config import get_logger
from app.models.documents import DocumentSource, FetchedDocument
from app.utils.hashing import sha256_of_bytes

_logger = get_logger(__name__)

_EXTENSION_BY_CONTENT_TYPE: Final[dict[str, str]] = {
    "application/pdf": ".pdf",
    "text/html": ".html",
    "application/xhtml+xml": ".html",
    "text/markdown": ".md",
    "text/plain": ".txt",
}

_CONTENT_TYPE_BY_EXTENSION: Final[dict[str, str]] = {
    ".pdf": "application/pdf",
    ".html": "text/html",
    ".htm": "text/html",
    ".md": "text/markdown",
    ".txt": "text/plain",
}


@dataclass(frozen=True)
class FetchResult:
    """Outcome of one fetch attempt.

    ``document is None`` means the source is confirmed unchanged and nothing was
    written to disk; ``reason`` says which of the checks decided that.
    """

    source_name: str
    changed: bool
    reason: str
    document: FetchedDocument | None = None


def _guess_extension(content_type: str, url: str) -> str:
    normalised_type = content_type.split(";")[0].strip().lower()
    if normalised_type in _EXTENSION_BY_CONTENT_TYPE:
        return _EXTENSION_BY_CONTENT_TYPE[normalised_type]
    url_suffix = Path(urlparse(url).path).suffix.lower()
    return url_suffix if url_suffix in _CONTENT_TYPE_BY_EXTENSION else ".bin"


def _validate_url(source: DocumentSource, allowed_schemes: list[str]) -> str:
    """Reject anything not explicitly allowed before a request is made.

    The registry is a config file; treating its URLs as trusted would make a
    config edit enough to read arbitrary local files over ``file://`` on an
    unexpected host, so the scheme allow-list is enforced here rather than assumed.
    """
    parsed_url = urlparse(source.url)
    if parsed_url.scheme not in allowed_schemes:
        raise DocumentFetchError(
            source.source_name,
            source.url,
            f"scheme '{parsed_url.scheme}' is not in allowed_schemes {allowed_schemes}",
        )
    return parsed_url.scheme


def _read_local_file(source: DocumentSource) -> tuple[bytes, str]:
    """Resolve a ``file://`` source against the repository root.

    Repository-relative paths keep the registry portable between machines, and
    the containment check stops ``file://../../etc/passwd`` from being reachable
    through a config edit.
    """
    raw_path = source.url[len("file://") :]
    candidate_path = Path(raw_path)
    resolved_path = (
        candidate_path if candidate_path.is_absolute() else PROJECT_ROOT / candidate_path
    ).resolve()

    if not str(resolved_path).startswith(str(PROJECT_ROOT.resolve())):
        raise DocumentFetchError(
            source.source_name, source.url, "file:// sources must live inside the repository"
        )
    if not resolved_path.exists():
        raise DocumentFetchError(source.source_name, source.url, "local file does not exist")

    content_type = _CONTENT_TYPE_BY_EXTENSION.get(resolved_path.suffix.lower(), "text/plain")
    return resolved_path.read_bytes(), content_type


def _download(
    source: DocumentSource,
    settings: IngestionSection,
    etag: str | None,
    last_modified: str | None,
) -> tuple[bytes, str, str | None, str | None] | None:
    """Perform a conditional GET. Returns None on HTTP 304 (not modified)."""
    request_headers = {"User-Agent": settings.user_agent, "Accept": "*/*"}
    if etag:
        request_headers["If-None-Match"] = etag
    if last_modified:
        request_headers["If-Modified-Since"] = last_modified

    downloaded_chunks: list[bytes] = []
    downloaded_bytes = 0

    with httpx.Client(
        timeout=settings.request_timeout_seconds,
        follow_redirects=True,
        headers=request_headers,
    ) as client, client.stream("GET", source.url) as response:
        if response.status_code == httpx.codes.NOT_MODIFIED:
            return None
        response.raise_for_status()

        # Stream and cap: a mis-registered URL pointing at a huge file must
        # not be able to exhaust local disk or memory.
        for block in response.iter_bytes():
            downloaded_bytes += len(block)
            if downloaded_bytes > settings.max_download_bytes:
                raise DocumentFetchError(
                    source.source_name,
                    source.url,
                    f"download exceeded max_download_bytes ({settings.max_download_bytes})",
                )
            downloaded_chunks.append(block)

        return (
            b"".join(downloaded_chunks),
            response.headers.get("content-type", "application/octet-stream"),
            response.headers.get("etag"),
            response.headers.get("last-modified"),
        )


class DocumentFetcher:
    """Fetches registry sources into ``data/raw`` and reports whether they changed."""

    def __init__(self, ingestion_settings: IngestionSection, raw_dir: Path) -> None:
        self._settings = ingestion_settings
        self._raw_dir = raw_dir

    def fetch(
        self,
        source: DocumentSource,
        *,
        previous_content_sha256: str | None = None,
        etag: str | None = None,
        last_modified: str | None = None,
    ) -> FetchResult:
        """Fetch one source, short-circuiting when it is provably unchanged."""
        scheme = _validate_url(source, self._settings.allowed_schemes)

        if scheme == "file":
            payload, content_type = _read_local_file(source)
            response_etag, response_last_modified = None, None
        else:
            download = self._download_with_retries(source, etag, last_modified)
            if download is None:
                _logger.info("source '%s' unchanged (HTTP 304)", source.source_name)
                return FetchResult(source.source_name, changed=False, reason="http_304")
            payload, content_type, response_etag, response_last_modified = download

        if not payload:
            raise DocumentFetchError(source.source_name, source.url, "empty response body")

        content_sha256 = sha256_of_bytes(payload)
        if previous_content_sha256 and content_sha256 == previous_content_sha256:
            _logger.info("source '%s' unchanged (identical bytes)", source.source_name)
            return FetchResult(source.source_name, changed=False, reason="identical_bytes")

        raw_path = self._write_raw_copy(source, payload, content_type, content_sha256)
        _logger.info(
            "fetched '%s' (%d bytes, %s) -> %s",
            source.source_name,
            len(payload),
            content_type.split(";")[0],
            raw_path.name,
        )
        return FetchResult(
            source_name=source.source_name,
            changed=True,
            reason="new_content",
            document=FetchedDocument(
                source=source,
                raw_path=raw_path,
                content_type=content_type,
                content_sha256=content_sha256,
                byte_size=len(payload),
                fetched_at=datetime.now(UTC),
                etag=response_etag,
                last_modified=response_last_modified,
            ),
        )

    def _download_with_retries(
        self, source: DocumentSource, etag: str | None, last_modified: str | None
    ) -> tuple[bytes, str, str | None, str | None] | None:
        """Retry transient network faults; surface everything else immediately.

        A 404 or a TLS failure will not fix itself, so retrying it only delays the
        operator's feedback. Timeouts and 5xx genuinely do resolve on retry.
        """

        @retry(
            retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
            stop=stop_after_attempt(self._settings.max_retries),
            wait=wait_exponential(multiplier=self._settings.retry_backoff_seconds),
            reraise=True,
        )
        def _attempt() -> tuple[bytes, str, str | None, str | None] | None:
            try:
                return _download(source, self._settings, etag, last_modified)
            except httpx.HTTPStatusError as status_error:
                if status_error.response.status_code < 500:
                    raise DocumentFetchError(
                        source.source_name,
                        source.url,
                        f"HTTP {status_error.response.status_code}",
                    ) from status_error
                raise

        try:
            return _attempt()
        except DocumentFetchError:
            raise
        except httpx.HTTPError as transport_error:
            raise DocumentFetchError(
                source.source_name, source.url, str(transport_error)
            ) from transport_error

    def _write_raw_copy(
        self, source: DocumentSource, payload: bytes, content_type: str, content_sha256: str
    ) -> Path:
        """Archive the exact bytes that produced a version.

        Keeping the raw copy is what makes a past decision reproducible: the text
        can be re-extracted with a fixed parser without re-downloading a URL whose
        content has since moved on.
        """
        source_directory = self._raw_dir / source.source_name
        source_directory.mkdir(parents=True, exist_ok=True)
        extension = _guess_extension(content_type, source.url)
        raw_path = source_directory / f"{content_sha256[:16]}{extension}"
        if not raw_path.exists():
            raw_path.write_bytes(payload)
        return raw_path
