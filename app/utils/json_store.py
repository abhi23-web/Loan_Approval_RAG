"""Small atomic JSON store used for document version metadata.

A separate database for a few hundred version records would be overhead without
benefit at this scale. What does matter is that a crash mid-write cannot leave a
truncated version file, because that file is the only record of which policy
version answered which question — so writes go through a temporary file and an
atomic rename.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def read_json(path: Path, default: Any = None) -> Any:
    """Read JSON, returning ``default`` when the file does not exist yet."""
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Serialise to a sibling temp file, flush to disk, then rename into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_descriptor, temp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(temp_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=indent, ensure_ascii=False, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
