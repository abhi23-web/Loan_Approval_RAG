"""ChromaDB persistence.

Why ChromaDB, in one line each — the full argument and the trade-offs live in
docs/architecture.md:

* it persists to a local directory, so the index survives a restart and the
  server does not re-embed anything on boot;
* its ``where`` filters operate on chunk metadata, which is what makes
  version-aware retrieval a filter rather than a second index;
* it is an embedded library, so there is no service to run for a reviewer to
  clone this repository and get an answer.

One collection per chunking strategy. That is what lets five strategies coexist
on disk and be evaluated against the same golden dataset without a rebuild
between runs, and it makes switching the live strategy a configuration change.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaClientSettings

from app.core.exceptions import VectorStoreError
from app.core.logging_config import get_logger
from app.models.documents import ChunkMetadata, TextChunk

_logger = get_logger(__name__)

# Chroma requires 3-512 characters of [a-zA-Z0-9._-], starting and ending
# alphanumeric. Strategy names contain digits and underscores, so only the
# length and edge rules need enforcing.
_MIN_COLLECTION_NAME_LENGTH = 3


@dataclass(frozen=True)
class VectorMatch:
    """One nearest-neighbour hit, with distance already converted to similarity."""

    text: str
    metadata: ChunkMetadata
    similarity: float


@lru_cache(maxsize=4)
def _get_chroma_client(persist_directory: str) -> chromadb.ClientAPI:
    """One client per directory per process.

    Constructing a ``PersistentClient`` opens the on-disk index; doing that per
    request would make every API call pay the open cost and would hold several
    handles to the same database.
    """
    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=persist_directory,
        settings=ChromaClientSettings(anonymized_telemetry=False, allow_reset=True),
    )


class ChromaVectorStore:
    """Thin, typed wrapper over the Chroma collections used by this project."""

    def __init__(
        self, persist_directory: Path, collection_prefix: str, distance: str = "cosine"
    ) -> None:
        self._persist_directory = str(persist_directory)
        self._collection_prefix = collection_prefix
        self._distance = distance
        self._client = _get_chroma_client(self._persist_directory)

    # ------------------------------------------------------------ plumbing

    def collection_name_for(self, strategy_name: str) -> str:
        name = f"{self._collection_prefix}_{strategy_name}".replace(" ", "_")
        if len(name) < _MIN_COLLECTION_NAME_LENGTH:
            raise VectorStoreError(f"collection name '{name}' is too short for ChromaDB")
        return name

    def collection_for(self, strategy_name: str) -> Collection:
        """Get or create the collection backing one chunking strategy.

        ``embedding_function=None`` is deliberate: this project supplies its own
        vectors, and leaving Chroma's default in place would silently download and
        run a second, different embedding model.
        """
        try:
            return self._client.get_or_create_collection(
                name=self.collection_name_for(strategy_name),
                metadata={"hnsw:space": self._distance},
                embedding_function=None,
            )
        except Exception as chroma_error:
            raise VectorStoreError(
                f"could not open collection for strategy '{strategy_name}': {chroma_error}"
            ) from chroma_error

    # ------------------------------------------------------------- writing

    def upsert_chunks(
        self, strategy_name: str, chunks: list[TextChunk], embeddings: list[list[float]]
    ) -> int:
        """Write chunks and their vectors, replacing any with the same ids.

        Chunk ids are deterministic, so re-running ingestion for a version that is
        already indexed overwrites in place instead of duplicating. An interrupted
        run is therefore safe to simply repeat.
        """
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"{len(chunks)} chunks but {len(embeddings)} embeddings; refusing to write"
            )
        if not chunks:
            return 0

        collection = self.collection_for(strategy_name)
        try:
            collection.upsert(
                ids=[chunk.metadata.chunk_id for chunk in chunks],
                embeddings=embeddings,
                documents=[chunk.text for chunk in chunks],
                metadatas=[chunk.metadata.to_chroma_metadata() for chunk in chunks],
            )
        except Exception as chroma_error:
            raise VectorStoreError(f"upsert failed for '{strategy_name}': {chroma_error}") from chroma_error

        _logger.info(
            "upserted %d chunk(s) into '%s'", len(chunks), self.collection_name_for(strategy_name)
        )
        return len(chunks)

    def delete_version(self, strategy_name: str, version_id: str) -> None:
        """Remove one version's chunks. Used only for rebuilds, never for updates."""
        self.collection_for(strategy_name).delete(where={"version_id": version_id})
        _logger.info("deleted chunks for version '%s' from '%s'", version_id, strategy_name)

    # ------------------------------------------------------------- reading

    def query(
        self,
        strategy_name: str,
        query_embedding: list[float],
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[VectorMatch]:
        """Nearest neighbours, optionally restricted by chunk metadata."""
        collection = self.collection_for(strategy_name)
        try:
            raw_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=metadata_filter or None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as chroma_error:
            raise VectorStoreError(f"query failed for '{strategy_name}': {chroma_error}") from chroma_error

        documents = (raw_results.get("documents") or [[]])[0]
        metadatas = (raw_results.get("metadatas") or [[]])[0]
        distances = (raw_results.get("distances") or [[]])[0]

        return [
            VectorMatch(
                text=document_text,
                metadata=ChunkMetadata.from_chroma_metadata(dict(raw_metadata)),
                similarity=self._distance_to_similarity(float(distance)),
            )
            for document_text, raw_metadata, distance in zip(
                documents, metadatas, distances, strict=False
            )
        ]

    def _distance_to_similarity(self, distance: float) -> float:
        """Convert Chroma's distance to a similarity in [0, 1].

        Kept in one place because a threshold in settings.yaml is meaningless
        unless everyone agrees what the number on the left of it means.
        """
        if self._distance == "cosine":
            # Chroma's cosine distance is 1 - cosine_similarity, in [0, 2].
            return max(0.0, min(1.0, 1.0 - distance))
        if self._distance == "ip":
            return max(0.0, min(1.0, -distance))
        # L2: map to a bounded, monotonically decreasing score.
        return 1.0 / (1.0 + max(0.0, distance))

    # --------------------------------------------------------- diagnostics

    def count(self, strategy_name: str) -> int:
        return self.collection_for(strategy_name).count()

    def indexed_strategies(self) -> list[str]:
        """Strategy names that currently have a non-empty collection on disk."""
        prefix = f"{self._collection_prefix}_"
        strategies: list[str] = []
        for collection in self._client.list_collections():
            collection_name = (
                collection if isinstance(collection, str) else collection.name
            )
            if collection_name.startswith(prefix):
                strategies.append(collection_name[len(prefix) :])
        return sorted(strategies)

    def reset_strategy(self, strategy_name: str) -> None:
        """Drop a whole collection. Only used by scripts/reset_data.py."""
        self._client.delete_collection(self.collection_name_for(strategy_name))
        _logger.warning("deleted collection for strategy '%s'", strategy_name)
