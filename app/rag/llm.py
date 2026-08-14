"""Generation providers.

The real provider is Ollama over HTTP. The determinism controls the system needs
— ``temperature``, ``top_p`` and ``seed`` — are all passed on every call, because
"the same question gives the same answer" is a requirement here, not a nicety.

That said: fixing a seed does not make a language model deterministic in the
mathematical sense. Different hardware, a different quantisation of the same
model, or a different context window can still change the output. What the system
actually guarantees is narrower and honest — *given the same model build, the
same retrieved context in the same order, and the same prompt, the output is
reproducible* — and the golden dataset measures whether that holds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import LLMSection
from app.core.exceptions import LLMError
from app.core.logging_config import get_logger
from app.utils.timing import measure_latency

_logger = get_logger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float


class LLMProvider(ABC):
    """Single-turn completion with a system prompt."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse: ...


class OllamaLLMProvider(LLMProvider):
    """Chat completion against a local Ollama server."""

    def __init__(self, settings: LLMSection) -> None:
        super().__init__(settings.model)
        self._settings = settings
        self._endpoint = f"{settings.base_url.rstrip('/')}/api/chat"

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        request_body = {
            "model": self._settings.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": self._settings.temperature,
                "top_p": self._settings.top_p,
                "seed": self._settings.seed,
                "num_ctx": self._settings.num_ctx,
                "num_predict": self._settings.num_predict,
            },
        }

        @retry(
            retry=retry_if_exception_type(httpx.TransportError),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=2),
            reraise=True,
        )
        def _post() -> httpx.Response:
            with httpx.Client(timeout=self._settings.request_timeout_seconds) as client:
                return client.post(self._endpoint, json=request_body)

        with measure_latency() as stopwatch:
            try:
                response = _post()
            except httpx.HTTPError as transport_error:
                raise LLMError(
                    f"cannot reach Ollama at {self._endpoint}. Is 'ollama serve' running? "
                    f"({transport_error})"
                ) from transport_error

        if response.status_code == httpx.codes.NOT_FOUND:
            raise LLMError(
                f"Ollama does not have model '{self._settings.model}'. "
                f"Run: ollama pull {self._settings.model}"
            )
        if response.status_code >= 400:
            raise LLMError(
                f"Ollama chat request failed with HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )

        payload = response.json()
        message_content = (payload.get("message") or {}).get("content")
        if not isinstance(message_content, str) or not message_content.strip():
            raise LLMError("Ollama returned an empty message")

        return LLMResponse(
            text=message_content.strip(),
            model=payload.get("model", self._settings.model),
            prompt_tokens=payload.get("prompt_eval_count"),
            completion_tokens=payload.get("eval_count"),
            latency_ms=stopwatch.elapsed_ms,
        )


class DeterministicLLMProvider(LLMProvider):
    """Offline test double.

    Returns a fixed, citation-shaped answer built from the prompt so that the
    citation validator, the API contract and the evaluation harness can all be
    tested in CI without a model server. It performs no reasoning and must never
    be used to produce a real recommendation.
    """

    def __init__(self, model_name: str = "deterministic-stub") -> None:
        super().__init__(model_name)

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        with measure_latency() as stopwatch:
            first_marker = "S1" if "[S1]" in user_prompt else ""
            citation_suffix = f" [{first_marker}]" if first_marker else ""
            answer = (
                "This is a deterministic stub response used for offline testing. "
                "The decision shown was produced by the rule engine, and the "
                "supporting policy text is quoted from the retrieved sources"
                f"{citation_suffix}."
            )
        return LLMResponse(
            text=answer,
            model=self.model_name,
            prompt_tokens=len(user_prompt) // 4,
            completion_tokens=len(answer) // 4,
            latency_ms=stopwatch.elapsed_ms,
        )


def build_llm_provider(settings: LLMSection) -> LLMProvider:
    if settings.provider == "ollama":
        _logger.info(
            "using Ollama generation: model=%s temperature=%s seed=%s",
            settings.model,
            settings.temperature,
            settings.seed,
        )
        return OllamaLLMProvider(settings)
    _logger.warning(
        "using the DETERMINISTIC LLM provider — offline stub, not a real model; "
        "generation-quality metrics from this provider are not meaningful"
    )
    return DeterministicLLMProvider()
