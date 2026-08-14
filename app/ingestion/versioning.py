"""Document version history.

Design points that matter for auditability:

* A new version is **appended**, never written over the previous one. A decision
  made in March under version 1 must still be explainable in September under
  version 3.
* "Active" is derived, not stored as a hand-set flag: the active version is the
  newest version whose effective date has arrived. A policy published today with
  an effective date next quarter is stored, retrievable by explicit request, and
  correctly *not* used to answer "what is the current rule".
* Version identity is a text hash. Re-fetching the same policy a hundred times
  produces one version, because nothing about the policy changed.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.exceptions import ConfigurationError
from app.core.logging_config import get_logger
from app.models.documents import DocumentSource, DocumentVersion, build_version_id
from app.utils.json_store import read_json, write_json_atomic

_logger = get_logger(__name__)

VERSION_STORE_FILENAME = "version_store.json"
_SCHEMA_VERSION = 1


class SourceVersionHistory(BaseModel):
    """Everything known about one registered source."""

    model_config = ConfigDict(extra="forbid")

    last_checked_at: datetime | None = None
    etag: str | None = None
    last_modified: str | None = None
    versions: list[DocumentVersion] = Field(default_factory=list)


class VersionStoreDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = _SCHEMA_VERSION
    sources: dict[str, SourceVersionHistory] = Field(default_factory=dict)


class VersionStore:
    """Append-only version history persisted as one atomic JSON file."""

    def __init__(self, metadata_dir: Path) -> None:
        self._path = metadata_dir / VERSION_STORE_FILENAME
        self._document = self._load()
        self._loaded_mtime_ns = self._current_mtime_ns()

    # ------------------------------------------------------------------ io

    def _current_mtime_ns(self) -> int | None:
        return self._path.stat().st_mtime_ns if self._path.exists() else None

    def reload_if_changed(self) -> bool:
        """Re-read the store when another process has written it.

        The update watcher runs as its own process, so the API's in-memory copy
        goes stale the moment a new policy version is ingested. Without this the
        API would keep filtering retrieval to a superseded version until it was
        restarted. An mtime check is a stat() call — cheap enough to do per
        request, and it is what makes "ingest in one terminal, see the new
        version in the running API" work.
        """
        current_mtime_ns = self._current_mtime_ns()
        if current_mtime_ns == self._loaded_mtime_ns:
            return False
        self._document = self._load()
        self._loaded_mtime_ns = current_mtime_ns
        _logger.info("version store reloaded from disk (changed by another process)")
        return True

    def _load(self) -> VersionStoreDocument:
        raw_document: Any = read_json(self._path, default=None)
        if raw_document is None:
            return VersionStoreDocument()
        if raw_document.get("schema_version") != _SCHEMA_VERSION:
            raise ConfigurationError(
                f"{self._path} has schema_version "
                f"{raw_document.get('schema_version')}, expected {_SCHEMA_VERSION}"
            )
        return VersionStoreDocument.model_validate(raw_document)

    def save(self) -> None:
        write_json_atomic(self._path, self._document.model_dump(mode="json"))
        self._loaded_mtime_ns = self._current_mtime_ns()

    @property
    def path(self) -> Path:
        return self._path

    # -------------------------------------------------------------- queries

    def history_for(self, source_name: str) -> SourceVersionHistory:
        return self._document.sources.setdefault(source_name, SourceVersionHistory())

    def versions_for(self, source_name: str) -> list[DocumentVersion]:
        return list(self.history_for(source_name).versions)

    def latest_version(self, source_name: str) -> DocumentVersion | None:
        versions = self.history_for(source_name).versions
        return max(versions, key=lambda version: version.version_number, default=None)

    def active_version(self, source_name: str) -> DocumentVersion | None:
        for version in self.history_for(source_name).versions:
            if version.is_active:
                return version
        return None

    def find_by_text_hash(self, source_name: str, text_sha256: str) -> DocumentVersion | None:
        for version in self.history_for(source_name).versions:
            if version.text_sha256 == text_sha256:
                return version
        return None

    def find_version(self, source_name: str, version_number: int) -> DocumentVersion | None:
        for version in self.history_for(source_name).versions:
            if version.version_number == version_number:
                return version
        return None

    def active_version_ids(self, source_names: list[str] | None = None) -> list[str]:
        """Version ids used to answer 'current policy' questions.

        Returned sorted so that the same knowledge state always produces the same
        filter, which is one of the ingredients of a reproducible answer.
        """
        selected_names = source_names or list(self._document.sources)
        active_ids = [
            active.version_id
            for source_name in selected_names
            if (active := self.active_version(source_name)) is not None
        ]
        return sorted(active_ids)

    def all_versions(self) -> list[DocumentVersion]:
        return [
            version
            for history in self._document.sources.values()
            for version in history.versions
        ]

    # -------------------------------------------------------------- updates

    def record_check(
        self,
        source_name: str,
        *,
        checked_at: datetime | None = None,
        etag: str | None = None,
        last_modified: str | None = None,
    ) -> None:
        """Record that the source was polled, with any HTTP validators seen.

        Validators are kept even when the content was unchanged: that is exactly
        the case where the next poll can be answered with a cheap 304.
        """
        history = self.history_for(source_name)
        history.last_checked_at = checked_at or datetime.now(UTC)
        if etag is not None:
            history.etag = etag
        if last_modified is not None:
            history.last_modified = last_modified
        latest = self.latest_version(source_name)
        if latest is not None:
            latest.last_checked_at = history.last_checked_at

    def register_version(
        self,
        source: DocumentSource,
        *,
        content_sha256: str,
        text_sha256: str,
        raw_path: Path,
        character_count: int,
        declared_version: str | None,
        effective_date: date | None,
        observed_at: datetime | None = None,
    ) -> DocumentVersion:
        """Append a new version for ``source`` and recompute which one is active."""
        history = self.history_for(source.source_name)
        observed_at = observed_at or datetime.now(UTC)
        next_version_number = (
            max((version.version_number for version in history.versions), default=0) + 1
        )

        new_version = DocumentVersion(
            source_name=source.source_name,
            version_number=next_version_number,
            version_id=build_version_id(source.source_name, next_version_number),
            content_sha256=content_sha256,
            text_sha256=text_sha256,
            url=source.url,
            institution=source.institution,
            document_title=source.document_title,
            document_type=source.document_type,
            authority=source.authority,
            declared_version=declared_version,
            effective_date=effective_date,
            first_seen_at=observed_at,
            last_checked_at=observed_at,
            is_active=False,  # set by _recompute_active_version below
            raw_path=str(raw_path),
            character_count=character_count,
        )
        history.versions.append(new_version)
        self._recompute_active_version(source.source_name, as_of=observed_at.date())

        _logger.info(
            "registered %s for '%s' (declared=%s, effective=%s, active=%s)",
            new_version.version_label,
            source.source_name,
            declared_version,
            effective_date,
            new_version.is_active,
        )
        return new_version

    def record_chunk_count(self, version_id: str, strategy_name: str, chunk_count: int) -> None:
        for history in self._document.sources.values():
            for version in history.versions:
                if version.version_id == version_id:
                    version.chunk_counts[strategy_name] = chunk_count
                    return
        raise ConfigurationError(f"unknown version_id '{version_id}'")

    def _recompute_active_version(self, source_name: str, *, as_of: date) -> None:
        """Elect the active version: newest effective version that has commenced.

        Versions with no parsable effective date fall back to their observation
        order, so a source that never states a date still behaves sensibly.
        """
        history = self.history_for(source_name)
        if not history.versions:
            return

        commenced_versions = [
            version
            for version in history.versions
            if version.effective_date is None or version.effective_date <= as_of
        ]
        # A corpus consisting only of future-dated policies still needs an answer;
        # the earliest one is the least wrong choice and is logged as unusual.
        if not commenced_versions:
            _logger.warning(
                "every version of '%s' is future-dated; activating the earliest",
                source_name,
            )
            commenced_versions = [min(history.versions, key=lambda version: version.version_number)]

        elected = max(
            commenced_versions,
            key=lambda version: (
                version.effective_date or date.min,
                version.version_number,
            ),
        )
        for version in history.versions:
            version.is_active = version.version_id == elected.version_id
