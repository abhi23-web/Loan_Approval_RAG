"""Content hashing — the basis of change detection and of stable chunk ids.

Change detection uses a hash of the *extracted text*, not of the raw bytes.
A bank product page ships a new build id or CSRF token on every request, so byte
equality is almost never true, while the policy text underneath is unchanged.
Hashing the text is what prevents a nightly re-embed of documents that did not
actually change.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_HASH_READ_BLOCK_BYTES = 1024 * 1024


def sha256_of_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_of_file(path: Path) -> str:
    """Stream the file so a large PDF never has to sit in memory twice."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_HASH_READ_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def build_chunk_id(version_id: str, chunking_strategy: str, chunk_index: int) -> str:
    """Deterministic chunk id.

    Deterministic on purpose: re-running ingestion for the same version and
    strategy overwrites the same ids in Chroma instead of duplicating them, so an
    interrupted run is safe to repeat.
    """
    return f"{version_id}::{chunking_strategy}::{chunk_index:05d}"
