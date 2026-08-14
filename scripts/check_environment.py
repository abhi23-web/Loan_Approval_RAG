#!/usr/bin/env python3
"""Pre-flight check.

Run this first. It tells you exactly which of the four things this project needs
is not ready — configuration, Ollama, the models, LangSmith — instead of letting
you discover it three commands later inside a stack trace.

    python scripts/check_environment.py
"""

from __future__ import annotations

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


def _check_ollama_models() -> bool:
    settings = get_settings()
    if settings.llm.provider != "ollama" and settings.embeddings.provider != "ollama":
        _line(_WARN, "ollama", "not configured as a provider; offline stubs are in use")
        return True

    base_url = settings.llm.base_url.rstrip("/")
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{base_url}/api/tags")
            response.raise_for_status()
            installed = {
                model.get("name", "") for model in response.json().get("models", [])
            }
    except httpx.HTTPError as connection_error:
        _line(_FAIL, "ollama server", f"{base_url} unreachable — run 'ollama serve' ({connection_error})")
        return False

    _line(_PASS, "ollama server", f"{base_url}, {len(installed)} model(s) installed")

    all_present = True
    for role, required_model in (("llm", settings.llm.model), ("embeddings", settings.embeddings.model)):
        # Ollama reports names with an explicit tag, e.g. "llama3.1:8b".
        is_installed = any(
            name == required_model or name.split(":")[0] == required_model.split(":")[0]
            for name in installed
        )
        if is_installed:
            _line(_PASS, f"{role} model", required_model)
        else:
            _line(_FAIL, f"{role} model", f"missing — run 'ollama pull {required_model}'")
            all_present = False
    return all_present


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
        _check_ollama_models(),
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
