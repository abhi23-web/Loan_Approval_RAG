"""Embedding providers.

Groq serves generation only — it has no embedding endpoint — so embeddings are
always computed locally. That is the right split anyway: generation sees a
handful of retrieved chunks, but embedding sees every document in full, so this
is the half worth keeping on your own machine regardless of which LLM is in use.

``fastembed`` (default)
    ONNX Runtime. Tens of megabytes, no PyTorch, fast on CPU for models this
    small. Downloads ~130 MB of weights once, then works offline.

``huggingface``
    sentence-transformers. Functionally interchangeable; offered because it is
    the better-known ecosystem and some models are published only for it. Costs
    a PyTorch install.

``ollama``
    A local Ollama server, if you happen to run one.

``deterministic``
    A hashing vectoriser used by the test suite. It produces real lexical
    similarity — shared words really do move two texts closer — so retrieval
    ordering can be asserted in CI with no model server and no download. It is
    not a semantic model and must never back a real answer; every evaluation
    artefact records which provider ran.

All providers cache query embeddings. Evaluating a 10-question golden dataset
across 5 chunking strategies and 4 top-k values embeds the same 10 questions 200
times otherwise, which is pure waste.

One rule holds across all of them: **vectors written by one model cannot be
queried with another.** ChromaDB will not warn you; it returns plausible-looking
results that are quietly wrong. After changing the model or provider, re-ingest
with ``--force``.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections import OrderedDict

import httpx
import numpy as np
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import EmbeddingsSection
from app.core.exceptions import EmbeddingError
from app.core.logging_config import get_logger

_logger = get_logger(__name__)

_QUERY_CACHE_MAX_ENTRIES = 512
_DETERMINISTIC_DIMENSIONS = 256


class EmbeddingProvider(ABC):
    """Turns text into vectors. One instance per process, reused everywhere."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()

    @abstractmethod
    def _embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a query, serving repeats from an in-process LRU cache."""
        cached_vector = self._query_cache.get(text)
        if cached_vector is not None:
            self._query_cache.move_to_end(text)
            return cached_vector

        vector = self._embed([text])[0]
        self._query_cache[text] = vector
        if len(self._query_cache) > _QUERY_CACHE_MAX_ENTRIES:
            self._query_cache.popitem(last=False)
        return vector

    @property
    def cache_size(self) -> int:
        return len(self._query_cache)


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Local embeddings via the Ollama HTTP API."""

    def __init__(self, settings: EmbeddingsSection) -> None:
        super().__init__(settings.model)
        self._settings = settings
        self._endpoint = f"{settings.base_url.rstrip('/')}/api/embed"

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        # Batching keeps a 400-chunk document to 25 requests instead of 400.
        for batch_start in range(0, len(texts), self._settings.batch_size):
            batch = texts[batch_start : batch_start + self._settings.batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        @retry(
            retry=retry_if_exception_type(httpx.TransportError),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1.5),
            reraise=True,
        )
        def _post() -> httpx.Response:
            with httpx.Client(timeout=self._settings.request_timeout_seconds) as client:
                return client.post(
                    self._endpoint, json={"model": self._settings.model, "input": batch}
                )

        try:
            response = _post()
        except httpx.HTTPError as transport_error:
            raise EmbeddingError(
                f"cannot reach Ollama at {self._endpoint}. Is 'ollama serve' running? "
                f"({transport_error})"
            ) from transport_error

        if response.status_code == httpx.codes.NOT_FOUND:
            raise EmbeddingError(
                f"Ollama rejected model '{self._settings.model}'. "
                f"Run: ollama pull {self._settings.model}"
            )
        if response.status_code >= 400:
            raise EmbeddingError(
                f"Ollama embedding request failed with HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )

        payload = response.json()
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(batch):
            raise EmbeddingError(
                f"Ollama returned {len(embeddings) if isinstance(embeddings, list) else 'no'} "
                f"embeddings for {len(batch)} input(s)"
            )
        return [[float(component) for component in vector] for vector in embeddings]


class DeterministicEmbeddingProvider(EmbeddingProvider):
    """Hashing vectoriser: reproducible, offline, and lexically meaningful.

    Words are hashed into a fixed number of buckets and the resulting count
    vector is L2-normalised, so cosine similarity behaves like a bag-of-words
    overlap score. That is enough to assert "the LTV chunk ranks above the fees
    chunk for an LTV query" in a test, without pretending to be a semantic model.
    """

    def __init__(self, model_name: str = "deterministic-hashing-256") -> None:
        super().__init__(model_name)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    @staticmethod
    def _embed_one(text: str) -> list[float]:
        vector = np.zeros(_DETERMINISTIC_DIMENSIONS, dtype=np.float32)
        for token in text.lower().split():
            cleaned_token = token.strip(".,;:()[]{}\"'`—–")
            if not cleaned_token:
                continue
            digest = hashlib.blake2b(cleaned_token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, "big") % _DETERMINISTIC_DIMENSIONS
            vector[bucket] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            # An empty or punctuation-only string still needs a valid unit vector.
            vector[0] = 1.0
            norm = 1.0
        return (vector / norm).tolist()


class FastEmbedEmbeddingProvider(EmbeddingProvider):
    """Local embeddings through ONNX Runtime, via the ``fastembed`` library.

    The default, and the reason is install weight rather than quality. Groq has
    no embedding endpoint, so embeddings must be computed somewhere local; the
    obvious candidate, ``sentence-transformers``, pulls PyTorch, whose default
    wheel bundles CUDA libraries and runs to several gigabytes. fastembed uses
    ONNX Runtime instead — tens of megabytes, no CUDA, and on CPU it is the
    faster of the two for models this small.

    ``BAAI/bge-small-en-v1.5`` produces 384-dimension vectors and is a genuine
    semantic model, unlike the deterministic stub below. The weights (~130 MB)
    download once on first use and then work offline.
    """

    def __init__(self, settings: EmbeddingsSection) -> None:
        super().__init__(settings.model)
        try:
            from fastembed import TextEmbedding
        except ImportError as missing_dependency:  # pragma: no cover - install-time
            raise EmbeddingError(
                "the 'fastembed' embedding provider needs the fastembed package.\n"
                "  pip install fastembed\n"
                "Alternatives: set embeddings.provider to 'huggingface' (needs PyTorch), "
                "or to 'deterministic' to run offline with no download at all."
            ) from missing_dependency

        self._settings = settings
        try:
            self._model = TextEmbedding(model_name=settings.model)
        except Exception as model_failure:
            raise EmbeddingError(
                f"fastembed could not load '{settings.model}': {model_failure}\n"
                "The first run downloads the model, so this usually means no internet "
                "or a blocked proxy. Known-good names include 'BAAI/bge-small-en-v1.5' "
                "and 'sentence-transformers/all-MiniLM-L6-v2'."
            ) from model_failure

    def _embed(self, texts: list[str]) -> list[list[float]]:
        # fastembed returns a generator of numpy arrays and batches internally.
        return [
            vector.tolist()
            for vector in self._model.embed(texts, batch_size=self._settings.batch_size)
        ]


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """Local embeddings through sentence-transformers.

    Functionally interchangeable with the fastembed provider; offered because
    sentence-transformers is the better-known ecosystem and some models are only
    published for it. The cost is PyTorch. On a machine without an NVIDIA GPU,
    install the CPU build first or pip will fetch several gigabytes of CUDA
    libraries that will never be used:

        pip install torch --index-url https://download.pytorch.org/whl/cpu
    """

    def __init__(self, settings: EmbeddingsSection) -> None:
        super().__init__(settings.model)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as missing_dependency:  # pragma: no cover - install-time
            raise EmbeddingError(
                "the 'huggingface' embedding provider needs sentence-transformers.\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                "  pip install sentence-transformers\n"
                "Or switch to embeddings.provider: fastembed, which needs neither."
            ) from missing_dependency

        self._settings = settings
        try:
            self._model = SentenceTransformer(settings.model, device="cpu")
        except Exception as model_failure:
            raise EmbeddingError(
                f"sentence-transformers could not load '{settings.model}': {model_failure}\n"
                "The first run downloads the model from huggingface.co, so this usually "
                "means no internet or a blocked proxy."
            ) from model_failure

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self._settings.batch_size,
            normalize_embeddings=self._settings.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [[float(component) for component in vector] for vector in vectors]


def build_embedding_provider(settings: EmbeddingsSection) -> EmbeddingProvider:
    """Instantiate the configured provider.

    Whichever is chosen, one rule holds absolutely: **vectors written by one
    model cannot be queried with another.** ChromaDB will not warn you — it
    returns plausible-looking results that are quietly wrong, which is far harder
    to notice than an error. After any change here, re-ingest with ``--force``.
    """
    if settings.provider == "fastembed":
        _logger.info("using fastembed embeddings: model=%s (local ONNX)", settings.model)
        return FastEmbedEmbeddingProvider(settings)

    if settings.provider == "huggingface":
        _logger.info(
            "using sentence-transformers embeddings: model=%s (local, CPU)", settings.model
        )
        return HuggingFaceEmbeddingProvider(settings)

    if settings.provider == "ollama":
        _logger.info(
            "using Ollama embeddings: model=%s endpoint=%s", settings.model, settings.base_url
        )
        return OllamaEmbeddingProvider(settings)

    _logger.warning(
        "using the DETERMINISTIC embedding provider — offline test double, "
        "not a semantic model; results are not valid retrieval quality evidence"
    )
    return DeterministicEmbeddingProvider()
