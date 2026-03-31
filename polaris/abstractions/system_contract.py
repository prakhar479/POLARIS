"""System contract model shared across orchestration, strategies, and tools."""

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from polaris.abstractions.connector_capabilities import ConnectorCapabilities


@dataclass(frozen=True)
class SystemContract:
    """Immutable system-level contract used during adaptation decisions."""

    system_id: str
    connector_type: str = ""
    supported_action_types: Tuple[str, ...] = ()
    action_aliases: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_capabilities(
        cls,
        system_id: str,
        connector_type: str,
        capabilities: ConnectorCapabilities,
        metadata: Dict[str, Any] | None = None,
    ) -> "SystemContract":
        """Create a contract from connector capabilities."""
        merged_metadata = dict(capabilities.metadata)
        merged_metadata.update(metadata or {})

        return cls(
            system_id=system_id,
            connector_type=connector_type,
            supported_action_types=tuple(capabilities.supported_action_types),
            action_aliases=dict(capabilities.action_aliases),
            metadata=merged_metadata,
        )

    def supported_actions_list(self) -> list[str]:
        """Return supported actions as a mutable list copy."""
        return list(self.supported_action_types)
