"""Meta-learning transparency record writer.

Writes append-only JSONL records for each meta-learning cycle so proposal generation,
validation, and application are externally inspectable.
"""

import dataclasses
import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from polaris.abstractions.observability import Logger


def _to_json_compatible(value: Any) -> Any:
    """Convert rich Python objects into JSON-serializable values."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _to_json_compatible(dataclasses.asdict(value))

    if isinstance(value, dict):
        return {str(k): _to_json_compatible(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_json_compatible(v) for v in value]

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Enum):
        return value.value

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


class MetaLearningTransparencyWriter:
    """Append-only JSONL writer for meta-learning lifecycle records."""

    def __init__(self, output_path: str, logger: Optional[Logger] = None) -> None:
        """Initialize the writer with the output file path and optional logger."""
        self._path = Path(output_path)
        self._logger = logger

    def record_cycle(self, record: Dict[str, Any]) -> None:
        """Append a single cycle record to disk as one JSON line."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = _to_json_compatible(record)
        with self._path.open("a", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True)
            handle.write("\n")

        if self._logger:
            self._logger.debug(
                "Meta-learning transparency record appended",
                output_path=str(self._path),
                cycle_id=record.get("cycle_id"),
                system_id=record.get("system_id"),
            )
