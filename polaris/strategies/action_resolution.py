"""Connector-aware strict action type resolution helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple

from polaris.abstractions.connector_capabilities import normalize_action_token

if TYPE_CHECKING:
    from polaris.abstractions.strategy import AdaptationContext
    from polaris.abstractions.system_contract import SystemContract


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


def require_supported_action_contract(
    context: "AdaptationContext",
    strategy_name: str,
) -> Tuple["SystemContract", list[str], Dict[str, str]]:
    """Extract and validate the strict connector action contract from context.

    Returns:
        Tuple of (system_contract, supported_action_types, action_aliases)

    Raises:
        StrictContractViolation: If no contract exists or no supported actions are available.
    """
    system_contract = context.system_contract
    supported_action_types = (
        system_contract.supported_actions_list() if system_contract is not None else []
    )
    if not supported_action_types:
        raise StrictContractViolation(
            "Missing connector-supported action contract for strict "
            f"{strategy_name} strategy (system_id='{context.system_id}')"
        )
    if not system_contract:
        raise StrictContractViolation(
            "Missing system contract for strict "
            f"{strategy_name} strategy (system_id='{context.system_id}')"
        )
    action_aliases = dict(system_contract.action_aliases)
    return system_contract, supported_action_types, action_aliases


def resolve_strict_action_payload(
    *,
    resolver: ConnectorActionResolver,
    action_type: Any,
    parameters: Any,
    supported_action_types: Sequence[str],
    action_aliases: Optional[Dict[str, str]],
    system_id: str,
    missing_type_error: str,
    invalid_parameters_error: str,
) -> Tuple[str, Dict[str, Any]]:
    """Validate and resolve a strict action payload into canonical action contract form."""
    if not isinstance(action_type, str) or not action_type.strip():
        raise StrictContractViolation(missing_type_error)

    if not isinstance(parameters, dict):
        raise StrictContractViolation(invalid_parameters_error)

    resolved_action_type = resolver.resolve_action_type(
        action_type,
        supported_action_types,
        action_aliases,
    )
    if resolved_action_type is None:
        raise StrictContractViolation(
            f"Unsupported action type '{action_type}' for system '{system_id}'"
        )

    return resolved_action_type, dict(parameters)
