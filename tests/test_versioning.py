"""Document versioning.

The behaviours asserted here are the ones an auditor would ask about: nothing is
overwritten, the active version is the one in force today, and re-reading the
same document does not manufacture a new version.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from app.ingestion.versioning import VersionStore
from app.models.documents import DocumentSource

_SOURCE = DocumentSource(
    source_name="meridian_home_loan_policy",
    url="file://documents/local_policies/current/meridian_home_loan_policy.md",
    institution="Meridian Housing Finance Limited (illustrative)",
    document_type="credit_policy",
    document_title="Meridian Retail Home Loan Credit Policy",
    authority="primary",
)


def _register(
    store: VersionStore,
    text_hash: str,
    effective: date | None,
    observed: datetime | None = None,
):
    return store.register_version(
        _SOURCE,
        content_sha256=f"content-{text_hash}",
        text_sha256=text_hash,
        raw_path=Path("data/raw/meridian/x.md"),
        character_count=1000,
        declared_version=None,
        effective_date=effective,
        observed_at=observed or datetime.now(UTC),
    )


def test_versions_are_appended_not_replaced(tmp_path: Path) -> None:
    store = VersionStore(tmp_path)
    _register(store, "hash-v1", date(2023, 4, 1))
    _register(store, "hash-v2", date(2024, 7, 1))

    versions = store.versions_for(_SOURCE.source_name)
    assert [version.version_number for version in versions] == [1, 2]
    assert versions[0].text_sha256 == "hash-v1", "the superseded version must survive"


def test_newest_commenced_version_is_active(tmp_path: Path) -> None:
    store = VersionStore(tmp_path)
    _register(store, "hash-v1", date(2023, 4, 1))
    _register(store, "hash-v2", date(2024, 7, 1))

    active = store.active_version(_SOURCE.source_name)
    assert active is not None
    assert active.version_number == 2
    assert store.active_version_ids() == [active.version_id]


def test_future_dated_version_is_stored_but_not_active(tmp_path: Path) -> None:
    """A policy published early must not answer 'what is the rule today'."""
    store = VersionStore(tmp_path)
    _register(store, "hash-v1", date(2023, 4, 1))
    _register(store, "hash-v2", date(2999, 1, 1))

    active = store.active_version(_SOURCE.source_name)
    assert active is not None
    assert active.version_number == 1
    assert store.find_version(_SOURCE.source_name, 2) is not None


def test_identical_text_is_recognised_and_not_re_registered(tmp_path: Path) -> None:
    store = VersionStore(tmp_path)
    _register(store, "hash-v1", date(2023, 4, 1))

    assert store.find_by_text_hash(_SOURCE.source_name, "hash-v1") is not None
    assert store.find_by_text_hash(_SOURCE.source_name, "hash-other") is None


def test_store_round_trips_through_disk(tmp_path: Path) -> None:
    store = VersionStore(tmp_path)
    _register(store, "hash-v1", date(2023, 4, 1))
    store.record_chunk_count("meridian_home_loan_policy::v1", "recursive_800_100", 42)
    store.save()

    reloaded = VersionStore(tmp_path)
    version = reloaded.find_version(_SOURCE.source_name, 1)
    assert version is not None
    assert version.chunk_counts["recursive_800_100"] == 42
    assert version.is_active


def test_reload_if_changed_picks_up_another_process_write(tmp_path: Path) -> None:
    """The API must see a version ingested by the watcher without a restart."""
    reader = VersionStore(tmp_path)
    assert reader.versions_for(_SOURCE.source_name) == []

    writer = VersionStore(tmp_path)
    _register(writer, "hash-v1", date(2023, 4, 1))
    writer.save()

    assert reader.reload_if_changed() is True
    assert len(reader.versions_for(_SOURCE.source_name)) == 1
    assert reader.reload_if_changed() is False


def test_recording_a_chunk_count_for_an_unknown_version_fails(tmp_path: Path) -> None:
    store = VersionStore(tmp_path)
    with pytest.raises(Exception, match="unknown version_id"):
        store.record_chunk_count("nope::v9", "recursive_800_100", 1)
