"""Typed configuration loaded from YAML, with environment-variable overrides.

Two rules drive this module:

1. Behaviour is configured in ``config/*.yaml``; secrets are configured in the
   environment. Nothing crosses over.
2. Every experiment must be reproducible from a config snapshot, so the loaded
   settings object is hashable to a dictionary and recorded alongside results.

Any leaf value can be overridden without editing YAML by exporting an
environment variable named ``HLR__<SECTION>__<KEY>`` (case-insensitive), for
example ``HLR__RETRIEVAL__TOP_K=8``. The experiment sweep relies on this so a
single variable can be changed per run without mutating the repository.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.exceptions import ConfigurationError

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DIR: Path = PROJECT_ROOT / "config"
DOCUMENTS_DIR: Path = PROJECT_ROOT / "documents"

ENV_OVERRIDE_PREFIX = "HLR__"
ENV_PATH_SEPARATOR = "__"


class _Section(BaseModel):
    """Base for configuration sections: unknown keys are an error, not a typo."""

    model_config = ConfigDict(extra="forbid")


class AppSection(_Section):
    name: str = "home-loan-rag"
    environment: str = "local"
    log_level: str = "INFO"


class PathsSection(_Section):
    raw_dir: Path
    processed_dir: Path
    chroma_dir: Path
    metadata_dir: Path
    experiment_results_dir: Path

    @model_validator(mode="after")
    def _resolve_against_project_root(self) -> PathsSection:
        """Relative paths in YAML are relative to the repository, not the CWD.

        Without this, running ``streamlit run frontend/streamlit_app.py`` from a
        different directory would silently create a second, empty ChromaDB.
        """
        for field_name in self.__class__.model_fields:
            configured_path: Path = getattr(self, field_name)
            if not configured_path.is_absolute():
                object.__setattr__(self, field_name, PROJECT_ROOT / configured_path)
        return self

    def create_all(self) -> None:
        for field_name in self.__class__.model_fields:
            directory: Path = getattr(self, field_name)
            directory.mkdir(parents=True, exist_ok=True)


class LLMSection(_Section):
    provider: Literal["ollama", "deterministic"] = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    num_ctx: int = 8192
    num_predict: int = 800
    request_timeout_seconds: float = 300.0


class EmbeddingsSection(_Section):
    provider: Literal["ollama", "deterministic"] = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "nomic-embed-text"
    batch_size: int = Field(default=16, ge=1)
    request_timeout_seconds: float = 300.0


class VectorStoreSection(_Section):
    collection_prefix: str = "home_loan_policy"
    distance: Literal["cosine", "l2", "ip"] = "cosine"


class IngestionSection(_Section):
    user_agent: str
    request_timeout_seconds: float = 60.0
    max_download_bytes: int = 26_214_400
    allowed_schemes: list[str] = Field(default_factory=lambda: ["https", "file"])
    max_retries: int = Field(default=3, ge=1)
    retry_backoff_seconds: float = 2.0


class ChunkingSection(_Section):
    active_strategy: str = "recursive_800_100"


class RetrievalSection(_Section):
    active_strategy: str = "recursive_800_100"
    top_k: int = Field(default=5, ge=1, le=50)
    min_similarity: float = Field(default=0.25, ge=0.0, le=1.0)
    restrict_to_active_versions: bool = True
    max_context_characters: int = Field(default=7000, ge=500)
    max_chunks_per_source: int = Field(default=3, ge=1)


class LtvSlab(_Section):
    """One row of the loan-to-value ladder.

    ``max_loan_amount_inr = None`` marks the open-ended top slab.
    """

    max_loan_amount_inr: float | None
    max_ltv_percent: float = Field(ge=0.0, le=100.0)


class RulesSection(_Section):
    min_applicant_age: int
    max_age_at_loan_maturity: int
    max_tenure_years: int
    min_monthly_income_inr: float
    min_credit_score: int
    conditional_credit_score: int
    max_foir_percent: float
    conditional_foir_percent: float
    min_employment_months_salaried: int
    min_employment_months_self_employed: int
    indicative_annual_interest_percent: float
    ltv_slabs: list[LtvSlab]

    @model_validator(mode="after")
    def _validate_slab_ladder(self) -> RulesSection:
        if not self.ltv_slabs:
            raise ConfigurationError("rules.ltv_slabs must contain at least one slab")
        open_ended_slabs = [slab for slab in self.ltv_slabs if slab.max_loan_amount_inr is None]
        if len(open_ended_slabs) != 1:
            raise ConfigurationError(
                "rules.ltv_slabs must contain exactly one open-ended slab "
                "(max_loan_amount_inr: null) to cover the largest loans"
            )
        if self.ltv_slabs[-1].max_loan_amount_inr is not None:
            raise ConfigurationError("the open-ended LTV slab must be listed last")
        if self.conditional_credit_score > self.min_credit_score:
            raise ConfigurationError(
                "rules.conditional_credit_score must not exceed rules.min_credit_score"
            )
        return self


class WatcherSection(_Section):
    poll_interval_seconds: int = Field(default=900, ge=30)
    jitter_seconds: int = Field(default=60, ge=0)
    run_on_start: bool = True
    max_consecutive_failures: int = Field(default=5, ge=1)


class ObservabilitySection(_Section):
    langsmith_project: str = "home-loan-rag"
    redact_applicant_pii: bool = True


class Settings(_Section):
    """The fully validated configuration for one process."""

    app: AppSection
    paths: PathsSection
    llm: LLMSection
    embeddings: EmbeddingsSection
    vector_store: VectorStoreSection
    ingestion: IngestionSection
    chunking: ChunkingSection
    retrieval: RetrievalSection
    rules: RulesSection
    watcher: WatcherSection
    observability: ObservabilitySection

    def experiment_fingerprint(self) -> dict[str, Any]:
        """The subset of configuration that can change an evaluation result.

        Recorded with every experiment so a number in a results file can always
        be traced back to the exact configuration that produced it.
        """
        return {
            "llm_provider": self.llm.provider,
            "llm_model": self.llm.model,
            "llm_temperature": self.llm.temperature,
            "llm_seed": self.llm.seed,
            "embedding_provider": self.embeddings.provider,
            "embedding_model": self.embeddings.model,
            "chunking_strategy": self.retrieval.active_strategy,
            "top_k": self.retrieval.top_k,
            "min_similarity": self.retrieval.min_similarity,
            "max_context_characters": self.retrieval.max_context_characters,
            "max_chunks_per_source": self.retrieval.max_chunks_per_source,
            "restrict_to_active_versions": self.retrieval.restrict_to_active_versions,
        }


class ChunkingStrategyConfig(BaseModel):
    """One named entry from ``config/chunking.yaml``."""

    model_config = ConfigDict(extra="forbid")

    name: str
    type: Literal["fixed", "recursive", "semantic"]
    description: str = ""
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    breakpoint_percentile: float | None = None
    buffer_sentences: int | None = None
    min_chunk_characters: int | None = None
    max_chunk_characters: int | None = None

    @model_validator(mode="after")
    def _validate_required_fields_per_type(self) -> ChunkingStrategyConfig:
        if self.type in {"fixed", "recursive"}:
            if self.chunk_size is None or self.chunk_overlap is None:
                raise ConfigurationError(
                    f"chunking strategy '{self.name}' needs chunk_size and chunk_overlap"
                )
            if self.chunk_overlap >= self.chunk_size:
                raise ConfigurationError(
                    f"chunking strategy '{self.name}': overlap must be smaller than size, "
                    "otherwise the splitter cannot make forward progress"
                )
        if self.type == "semantic" and self.breakpoint_percentile is None:
            raise ConfigurationError(
                f"chunking strategy '{self.name}' needs a breakpoint_percentile"
            )
        return self


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategies: dict[str, ChunkingStrategyConfig]
    recursive_separators: list[str]

    def get(self, strategy_name: str) -> ChunkingStrategyConfig:
        try:
            return self.strategies[strategy_name]
        except KeyError as missing_strategy:
            available = ", ".join(sorted(self.strategies))
            raise ConfigurationError(
                f"unknown chunking strategy '{strategy_name}'. Available: {available}"
            ) from missing_strategy


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigurationError(f"configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        parsed = yaml.safe_load(handle)
    if not isinstance(parsed, dict):
        raise ConfigurationError(f"configuration file {path} must contain a YAML mapping")
    return parsed


def _apply_environment_overrides(raw_settings: dict[str, Any]) -> dict[str, Any]:
    """Overlay ``HLR__SECTION__KEY`` environment variables onto the YAML mapping.

    Values stay strings; Pydantic coerces them during validation, which keeps a
    single source of truth for types.
    """
    for variable_name, variable_value in os.environ.items():
        if not variable_name.startswith(ENV_OVERRIDE_PREFIX):
            continue
        path_parts = variable_name[len(ENV_OVERRIDE_PREFIX) :].lower().split(ENV_PATH_SEPARATOR)
        if len(path_parts) < 2:
            raise ConfigurationError(
                f"override {variable_name} must name a section and a key, "
                f"e.g. {ENV_OVERRIDE_PREFIX}RETRIEVAL{ENV_PATH_SEPARATOR}TOP_K"
            )
        cursor: dict[str, Any] = raw_settings
        for part in path_parts[:-1]:
            next_level = cursor.get(part)
            if not isinstance(next_level, dict):
                raise ConfigurationError(
                    f"override {variable_name} does not match a section in settings.yaml"
                )
            cursor = next_level
        leaf_key = path_parts[-1]
        if leaf_key not in cursor:
            raise ConfigurationError(
                f"override {variable_name} targets unknown key '{leaf_key}'"
            )
        cursor[leaf_key] = variable_value
    return raw_settings


def load_settings(settings_path: Path | None = None) -> Settings:
    """Read, override and validate ``config/settings.yaml``. Not cached."""
    raw_settings = _read_yaml(settings_path or CONFIG_DIR / "settings.yaml")
    return Settings.model_validate(_apply_environment_overrides(raw_settings))


def load_chunking_config(chunking_path: Path | None = None) -> ChunkingConfig:
    """Read and validate ``config/chunking.yaml``. Not cached."""
    raw_chunking = _read_yaml(chunking_path or CONFIG_DIR / "chunking.yaml")
    raw_strategies = raw_chunking.get("strategies") or {}
    return ChunkingConfig(
        strategies={
            strategy_name: ChunkingStrategyConfig(name=strategy_name, **strategy_body)
            for strategy_name, strategy_body in raw_strategies.items()
        },
        recursive_separators=raw_chunking.get("recursive_separators") or ["\n\n", "\n", " ", ""],
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton.

    Cached because the FastAPI request path reads settings on every call and
    re-parsing YAML per request is pure waste. Tests that need a different
    configuration call :func:`load_settings` directly or clear the cache.
    """
    return load_settings()


@lru_cache(maxsize=1)
def get_chunking_config() -> ChunkingConfig:
    """Process-wide chunking configuration singleton."""
    return load_chunking_config()


def reset_settings_cache() -> None:
    """Drop cached configuration. Used by tests and by the experiment runner."""
    get_settings.cache_clear()
    get_chunking_config.cache_clear()
