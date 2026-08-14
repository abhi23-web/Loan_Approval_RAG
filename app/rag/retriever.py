"""Version-aware retrieval over ChromaDB.

Three behaviours here are worth more than the vector search itself:

* **Version filtering.** "What is the current rule" and "what was the rule in
  2023" are different questions over the same corpus. The active-version filter
  is what keeps a superseded clause from answering the first one.
* **Deterministic ordering.** Ties in similarity are broken by chunk id, so the
  prompt sees the same context in the same order on every run. Without this,
  reproducibility fails for reasons that have nothing to do with the model.
* **Over-fetch then filter.** Post-filters (threshold, per-source cap) can only
  remove results, so the store is asked for more than ``top_k`` and the surplus
  is discarded. Asking for exactly ``top_k`` and then filtering would silently
  return fewer chunks than configured.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from app.core.config import RetrievalSection
from app.core.logging_config import get_logger
from app.core.tracing import add_trace_metadata, traced
from app.ingestion.embeddings import EmbeddingProvider
from app.ingestion.vector_store import ChromaVectorStore, VectorMatch
from app.ingestion.versioning import VersionStore
from app.models.assessment import RetrievalDiagnostics
from app.models.documents import RetrievedChunk
from app.utils.timing import measure_latency

_logger = get_logger(__name__)

# How much to over-fetch before post-filtering. Three times top_k has been enough
# to survive the per-source cap in every configuration exercised so far; it is a
# multiplier rather than a constant so it scales with top_k.
_OVERFETCH_MULTIPLIER = 3


@dataclass(frozen=True)
class RetrievalRequest:
    """One retrieval, with every knob that can change its result made explicit."""

    query: str
    top_k: int | None = None
    min_similarity: float | None = None
    restrict_to_active_versions: bool | None = None
    # Explicit historical lookup, e.g. {"meridian_home_loan_policy": 2}.
    version_numbers_by_source: dict[str, int] | None = None
    source_names: list[str] | None = None
    strategy_name: str | None = None


@dataclass
class RetrievalOutcome:
    chunks: list[RetrievedChunk]
    diagnostics: RetrievalDiagnostics


class PolicyRetriever:
    """Embeds a query, searches one strategy's collection, and filters the hits."""

    def __init__(
        self,
        settings: RetrievalSection,
        vector_store: ChromaVectorStore,
        version_store: VersionStore,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._settings = settings
        self._vector_store = vector_store
        self._version_store = version_store
        self._embedding_provider = embedding_provider

    @traced("rag.retrieve", run_type="retriever")
    def retrieve(self, request: RetrievalRequest) -> RetrievalOutcome:
        top_k = request.top_k or self._settings.top_k
        min_similarity = (
            request.min_similarity
            if request.min_similarity is not None
            else self._settings.min_similarity
        )
        restrict_to_active = (
            request.restrict_to_active_versions
            if request.restrict_to_active_versions is not None
            else self._settings.restrict_to_active_versions
        )
        strategy_name = request.strategy_name or self._settings.active_strategy

        version_ids = self._resolve_version_ids(request, restrict_to_active)
        metadata_filter = self._build_metadata_filter(version_ids, request.source_names)

        with measure_latency() as stopwatch:
            query_embedding = self._embedding_provider.embed_query(request.query)
            matches = self._vector_store.query(
                strategy_name=strategy_name,
                query_embedding=query_embedding,
                top_k=top_k * _OVERFETCH_MULTIPLIER,
                metadata_filter=metadata_filter,
            )

        above_threshold = [match for match in matches if match.similarity >= min_similarity]
        selected = self._apply_per_source_cap(self._sort_deterministically(above_threshold))[:top_k]

        chunks = [
            RetrievedChunk(
                text=match.text,
                metadata=match.metadata,
                similarity=round(match.similarity, 6),
                rank=rank,
            )
            for rank, match in enumerate(selected, start=1)
        ]

        diagnostics = RetrievalDiagnostics(
            query=request.query,
            chunking_strategy=strategy_name,
            top_k=top_k,
            min_similarity=min_similarity,
            restricted_to_active_versions=restrict_to_active,
            active_version_ids=version_ids or [],
            retrieved_count=len(chunks),
            dropped_below_threshold_count=len(matches) - len(above_threshold),
            retrieval_latency_ms=round(stopwatch.elapsed_ms, 2),
            context_characters=sum(len(chunk.text) for chunk in chunks),
        )
        add_trace_metadata(
            strategy=strategy_name,
            top_k=top_k,
            min_similarity=min_similarity,
            version_filter=version_ids,
            retrieved=len(chunks),
            dropped_below_threshold=diagnostics.dropped_below_threshold_count,
        )
        _logger.info(
            "retrieved %d/%d chunk(s) from '%s' in %.0fms (dropped %d below %.2f)",
            len(chunks),
            len(matches),
            strategy_name,
            stopwatch.elapsed_ms,
            diagnostics.dropped_below_threshold_count,
            min_similarity,
        )
        return RetrievalOutcome(chunks=chunks, diagnostics=diagnostics)

    # ------------------------------------------------------------- filtering

    def _resolve_version_ids(
        self, request: RetrievalRequest, restrict_to_active: bool
    ) -> list[str] | None:
        """Decide which version ids retrieval is allowed to see.

        An explicit historical request wins over the active-version restriction;
        that is the whole point of asking for a specific version.
        """
        if request.version_numbers_by_source:
            explicit_ids = [
                version.version_id
                for source_name, version_number in sorted(
                    request.version_numbers_by_source.items()
                )
                if (version := self._version_store.find_version(source_name, version_number))
            ]
            return sorted(explicit_ids)
        if restrict_to_active:
            return self._version_store.active_version_ids(request.source_names)
        return None

    @staticmethod
    def _build_metadata_filter(
        version_ids: list[str] | None, source_names: list[str] | None
    ) -> dict[str, Any] | None:
        """Assemble a Chroma ``where`` clause from the resolved constraints."""
        conditions: list[dict[str, Any]] = []
        if version_ids:
            conditions.append({"version_id": {"$in": version_ids}})
        if source_names:
            conditions.append({"source_name": {"$in": sorted(source_names)}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    @staticmethod
    def _sort_deterministically(matches: list[VectorMatch]) -> list[VectorMatch]:
        """Rank by similarity, breaking ties on chunk id rather than on luck."""
        return sorted(
            matches,
            key=lambda match: (-round(match.similarity, 6), match.metadata.chunk_id),
        )

    def _apply_per_source_cap(self, matches: list[VectorMatch]) -> list[VectorMatch]:
        """Stop one verbose document from occupying every slot in the prompt.

        A 200-page circular will always have more near-duplicate paragraphs than
        a two-page policy, and without a cap it wins top-k on volume rather than
        relevance — which is exactly the failure LangSmith traces surface as
        "every retrieved chunk says the same thing".
        """
        cap = self._settings.max_chunks_per_source
        kept: list[VectorMatch] = []
        count_by_source: defaultdict[str, int] = defaultdict(int)
        for match in matches:
            source_name = match.metadata.source_name
            if count_by_source[source_name] >= cap:
                continue
            count_by_source[source_name] += 1
            kept.append(match)
        return kept
