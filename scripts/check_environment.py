#!/usr/bin/env python3
"""Pre-flight check.

Run this first. It tells you exactly which of the four things this project needs
is not ready — configuration, Ollama, the models, LangSmith — instead of letting
you discover it three commands later inside a stack trace.

    python scripts/check_environment.py
"""

from __future__ import annotations

import os
import sys

# Running this file directly puts its own directory on sys.path rather than the
# repository root, which would hide the "app" package. `pip install -e .` makes
# this redundant but never harmful; keeping it means a fresh clone runs with no
# install step at all.
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from app.core.config import get_chunking_config, get_settings
from app.core.logging_config import configure_logging
from app.core.tracing import tracing_enabled
from app.ingestion.embeddings import build_embedding_provider

_PASS = "  ok  "
_WARN = " warn "
_FAIL = " FAIL "


def _line(status: str, label: str, detail: str = "") -> None:
    print(f"[{status}] {label}" + (f" — {detail}" if detail else ""))


def _check_configuration() -> bool:
    try:
        settings = get_settings()
        chunking = get_chunking_config()
    except Exception as configuration_error:
        _line(_FAIL, "configuration", str(configuration_error))
        return False
    chunking.get(settings.retrieval.active_strategy)
    _line(
        _PASS,
        "configuration",
        f"strategy='{settings.retrieval.active_strategy}', top_k={settings.retrieval.top_k}",
    )
    return True


def _check_llm() -> bool:
    """Confirm the configured generation provider can actually be used.

    Deliberately does NOT call the model. A pre-flight check that costs a token
    and a second of latency stops being run, and a check nobody runs is worse
    than no check. This verifies the things that are cheap to verify: the key is
    present and shaped right, or the local server is reachable.
    """
    settings = get_settings()

    if settings.llm.provider == "deterministic":
        _line(_WARN, "llm", "offline stub in use — answers are placeholder text, not real")
        return True

    if settings.llm.provider == "openai":
        key_variable = settings.llm.api_key_env_var
        api_key = os.environ.get(key_variable, "").strip()
        is_local = any(
            host in settings.llm.base_url for host in ("localhost", "127.0.0.1", "0.0.0.0")
        )

        if is_local:
            _line(_PASS, "llm", f"{settings.llm.model} at {settings.llm.base_url} (local, no key)")
            return True

        if not api_key or api_key.endswith("replace_me"):
            _line(
                _FAIL,
                "llm",
                f"{key_variable} is not set — get a free key at "
                f"https://console.groq.com/keys and put it in .env",
            )
            return False

        # Groq keys start with gsk_. A wrong-provider key is a common paste error
        # and produces a 401 much later, which is harder to connect back to here.
        if "groq.com" in settings.llm.base_url and not api_key.startswith("gsk_"):
            _line(
                _WARN,
                "llm",
                f"{key_variable} does not start with 'gsk_' — is that a Groq key?",
            )
            return True

        masked = f"{api_key[:7]}…{api_key[-4:]}" if len(api_key) > 12 else "set"
        _line(_PASS, "llm", f"{settings.llm.model} via {settings.llm.base_url} (key {masked})")
        return True

    # provider == "ollama"
    base_url = settings.llm.base_url.rstrip("/")
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{base_url}/api/tags")
            response.raise_for_status()
            installed = {model.get("name", "") for model in response.json().get("models", [])}
    except httpx.HTTPError as connection_error:
        _line(
            _FAIL,
            "ollama server",
            f"{base_url} unreachable — run 'ollama serve' ({connection_error})",
        )
        return False

    is_installed = any(
        name == settings.llm.model or name.split(":")[0] == settings.llm.model.split(":")[0]
        for name in installed
    )
    if is_installed:
        _line(_PASS, "llm", f"ollama {settings.llm.model}")
        return True
    _line(_FAIL, "llm", f"missing — run 'ollama pull {settings.llm.model}'")
    return False


def _check_embeddings() -> bool:
    """Confirm the embedding provider is installed and its model can load.

    This one DOES construct the provider, because the failure it catches — a
    missing package or an undownloadable model — is exactly what would otherwise
    surface halfway through an ingest, after the fetch work is already done.
    """
    settings = get_settings()
    provider_name = settings.embeddings.provider

    if provider_name == "deterministic":
        _line(
            _WARN,
            "embeddings",
            "offline hashing stub — no semantic understanding; retrieval quality "
            "from this provider is not meaningful",
        )
        return True

    try:
        build_embedding_provider(settings.embeddings)
    except Exception as provider_failure:
        first_line = str(provider_failure).splitlines()[0]
        _line(_FAIL, "embeddings", f"{provider_name} unavailable — {first_line}")
        return False

    _line(_PASS, "embeddings", f"{provider_name} '{settings.embeddings.model}' (local)")
    return True


def _check_langsmith() -> bool:
    if tracing_enabled():
        _line(_PASS, "langsmith", f"project='{get_settings().observability.langsmith_project}'")
    else:
        _line(_WARN, "langsmith", "no LANGSMITH_API_KEY; the system runs but produces no traces")
    return True


def _check_index() -> bool:
    from app.services.container import get_container

    container = get_container()
    strategy = container.settings.retrieval.active_strategy
    try:
        chunk_count = container.vector_store.count(strategy)
    except Exception as vector_store_error:
        _line(_FAIL, "chromadb", str(vector_store_error))
        return False

    if chunk_count == 0:
        _line(_WARN, "chromadb index", f"empty for '{strategy}' — run 'python scripts/ingest.py'")
    else:
        _line(_PASS, "chromadb index", f"{chunk_count} chunk(s) for '{strategy}'")
    return True


def main() -> int:
    load_dotenv()
    configure_logging("WARNING")
    print("\nHome Loan RAG — environment check\n" + "-" * 60)
    checks = [
        _check_configuration(),
        _check_llm(),
        _check_embeddings(),
        _check_langsmith(),
        _check_index(),
    ]
    print("-" * 60)
    if all(checks):
        print("Ready.\n")
        return 0
    print("One or more checks failed. Fix the FAIL lines above, then re-run.\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
