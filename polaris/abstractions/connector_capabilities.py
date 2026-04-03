"""Connector capability models used by contract-first runtime components."""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Tuple


def normalize_action_token(token: str) -> str:
    """Normalize action tokens for robust comparisons."""
    return re.sub(r"[\s\-]+", "_", (token or "").strip().lower())


@dataclass(frozen=True)
class ConnectorCapabilities:
    """Normalized action/capability metadata exposed by connectors."""

    supported_action_types: Tuple[str, ...] = ()
    action_aliases: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_supported_action_types(
        cls,
        action_types: Iterable[str],
        action_aliases: Dict[str, str] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> "ConnectorCapabilities":
        """Build capabilities from supported action names and optional aliases."""
        deduped: list[str] = []
        seen: set[str] = set()
        for raw_action in action_types:
            action = raw_action.strip()
            if not action:
                continue
            norm = normalize_action_token(action)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            deduped.append(action)

        supported_by_norm = {normalize_action_token(value): value for value in deduped}

        normalized_aliases: Dict[str, str] = {}
        for alias, canonical in (action_aliases or {}).items():
            alias_norm = normalize_action_token(alias)
            canonical_norm = normalize_action_token(canonical)
            if not alias_norm or canonical_norm not in supported_by_norm:
                continue
            normalized_aliases[alias_norm] = supported_by_norm[canonical_norm]

        return cls(
            supported_action_types=tuple(deduped),
            action_aliases=normalized_aliases,
            metadata=dict(metadata or {}),
        )
