"""LangSmith tracing.

The LangSmith SDK is used directly rather than through LangChain. That keeps the
dependency surface small and, more importantly, means the trace tree mirrors this
application's own function boundaries — query processing, retrieval, context
assembly, generation — instead of a framework's internal runnables. When a trace
shows a bad answer, the span that produced it maps to a file you can open.

Tracing is optional. With no ``LANGSMITH_API_KEY`` the decorators become
no-ops and the system runs exactly as before, so a missing key never breaks a
demo or a test run.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from app.core.logging_config import get_logger

_logger = get_logger(__name__)

_CallParams = ParamSpec("_CallParams")
_ReturnType = TypeVar("_ReturnType")


def tracing_enabled() -> bool:
    """True when a LangSmith key is present and tracing has not been disabled."""
    has_key = bool(os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY"))
    tracing_flag = (
        os.environ.get("LANGSMITH_TRACING")
        or os.environ.get("LANGCHAIN_TRACING_V2")
        or "true"
    ).strip().lower()
    return has_key and tracing_flag not in {"false", "0", "no"}


def configure_tracing(project_name: str) -> bool:
    """Point the LangSmith SDK at ``project_name``. Returns whether it is active.

    Called once per entry point. The SDK reads these variables lazily, so setting
    them here is enough for every ``@traced`` span in the process.
    """
    if not tracing_enabled():
        _logger.info(
            "LangSmith tracing disabled (no LANGSMITH_API_KEY); "
            "the pipeline runs normally but produces no traces"
        )
        return False

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    # The SDK still reads the legacy variable names in some code paths.
    os.environ["LANGCHAIN_TRACING_V2"] = os.environ.get("LANGSMITH_TRACING", "true")
    os.environ["LANGSMITH_PROJECT"] = project_name
    os.environ["LANGCHAIN_PROJECT"] = project_name
    _logger.info("LangSmith tracing enabled for project '%s'", project_name)
    return True


def traced(
    name: str,
    run_type: str = "chain",
    **trace_kwargs: Any,
) -> Callable[[Callable[_CallParams, _ReturnType]], Callable[_CallParams, _ReturnType]]:
    """Decorate a function as a LangSmith span, degrading to a no-op without a key.

    The import of ``langsmith.traceable`` is deferred to call time so that the
    decision to trace is made when the process is fully configured, not at import
    time when ``.env`` may not have been loaded yet.
    """

    def decorator(
        target_function: Callable[_CallParams, _ReturnType],
    ) -> Callable[_CallParams, _ReturnType]:
        wrapped_reference: dict[str, Callable[..., Any]] = {}

        @wraps(target_function)
        def wrapper(*args: _CallParams.args, **kwargs: _CallParams.kwargs) -> _ReturnType:
            if not tracing_enabled():
                return target_function(*args, **kwargs)

            traced_function = wrapped_reference.get("value")
            if traced_function is None:
                try:
                    from langsmith import traceable
                except ImportError:  # pragma: no cover - langsmith is a hard dependency
                    _logger.warning("langsmith is not installed; continuing untraced")
                    return target_function(*args, **kwargs)
                traced_function = traceable(
                    run_type=run_type, name=name, **trace_kwargs
                )(target_function)
                wrapped_reference["value"] = traced_function
            return traced_function(*args, **kwargs)

        return wrapper

    return decorator


def add_trace_metadata(**metadata: Any) -> None:
    """Attach metadata to the currently executing span, if there is one.

    Used to record retrieval configuration and version filters on the span that
    performed them, which is what makes a trace explain itself later.
    """
    if not tracing_enabled():
        return
    try:
        from langsmith.run_helpers import get_current_run_tree

        current_run = get_current_run_tree()
        if current_run is not None:
            current_run.extra.setdefault("metadata", {}).update(metadata)
    except Exception as tracing_failure:
        # Observability must never take down the request path it observes.
        _logger.debug("could not attach trace metadata: %s", tracing_failure)
