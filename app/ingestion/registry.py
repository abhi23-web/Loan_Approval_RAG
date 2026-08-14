"""The document source registry.

The registry is read-only at runtime. Adding a policy source is a YAML edit and a
re-run of the ingestion script — never a code change — which is the whole point
of keeping URLs out of the modules that use them.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from app.core.config import PROJECT_ROOT
from app.core.exceptions import ConfigurationError
from app.core.logging_config import get_logger
from app.models.documents import DocumentSource

_logger = get_logger(__name__)

DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "documents" / "source_registry.yaml"


class SourceRegistry:
    """Loads and indexes ``documents/source_registry.yaml``."""

    def __init__(self, sources: list[DocumentSource]) -> None:
        self._sources_by_name: dict[str, DocumentSource] = {
            source.source_name: source for source in sources
        }
        if len(self._sources_by_name) != len(sources):
            raise ConfigurationError(
                "duplicate source_name in the registry; version history is keyed "
                "on source_name, so names must be unique"
            )

    @classmethod
    def load(cls, registry_path: Path | None = None) -> SourceRegistry:
        path = registry_path or DEFAULT_REGISTRY_PATH
        if not path.exists():
            raise ConfigurationError(f"source registry not found: {path}")

        with path.open("r", encoding="utf-8") as handle:
            raw_registry = yaml.safe_load(handle) or {}

        raw_sources = raw_registry.get("sources")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise ConfigurationError(f"{path} must define a non-empty 'sources' list")

        sources: list[DocumentSource] = []
        for entry_index, raw_source in enumerate(raw_sources):
            try:
                sources.append(DocumentSource.model_validate(raw_source))
            except ValidationError as validation_error:
                raise ConfigurationError(
                    f"invalid source at index {entry_index} in {path}: {validation_error}"
                ) from validation_error

        _logger.info(
            "loaded %d document sources (%d enabled) from %s",
            len(sources),
            sum(1 for source in sources if source.enabled),
            path.name,
        )
        return cls(sources)

    def all_sources(self) -> list[DocumentSource]:
        return list(self._sources_by_name.values())

    def enabled_sources(self) -> list[DocumentSource]:
        return [source for source in self._sources_by_name.values() if source.enabled]

    def get(self, source_name: str) -> DocumentSource:
        try:
            return self._sources_by_name[source_name]
        except KeyError as missing_source:
            raise ConfigurationError(
                f"unknown source '{source_name}'. Known: "
                f"{', '.join(sorted(self._sources_by_name))}"
            ) from missing_source

    def __len__(self) -> int:
        return len(self._sources_by_name)
