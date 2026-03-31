"""Connector-aware strict action type resolution helpers."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from polaris.abstractions.connector_capabilities import normalize_action_token


class StrictContractViolation(RuntimeError):
    """Raised when strategy output violates strict runtime contracts."""


class ConnectorActionResolver:
    """Resolve model-emitted actions using canonical contract vocab only."""

    @staticmethod
    def normalize_token(token: str) -> str:
        """Normalize action token for matching."""
        return normalize_action_token(token)

    def _normalized_aliases(self, action_aliases: Optional[Dict[str, str]]) -> Dict[str, str]:
        """Normalize explicit contract aliases."""
        normalized: Dict[str, str] = {}
        for alias, canonical in (action_aliases or {}).items():
            alias_norm = self.normalize_token(alias)
            canonical_norm = self.normalize_token(canonical)
            if alias_norm and canonical_norm:
                normalized[alias_norm] = canonical_norm
        return normalized

    def resolve_action_type(
        self,
        raw_action_type: Optional[str],
        supported_action_types: Optional[Sequence[str]] = None,
        action_aliases: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """Resolve a raw action token into connector-supported canonical name."""
        token_norm = self.normalize_token((raw_action_type or "").strip())
        if not token_norm:
            return None

        supported = [s for s in (supported_action_types or []) if isinstance(s, str) and s.strip()]
        if not supported:
            return None

        supported_map = {
            self.normalize_token(action_type): action_type.strip() for action_type in supported
        }

        if token_norm in supported_map:
            return supported_map[token_norm]

        alias_norm = self._normalized_aliases(action_aliases).get(token_norm)
        if alias_norm and alias_norm in supported_map:
            return supported_map[alias_norm]

        return None
