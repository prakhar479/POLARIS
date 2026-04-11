"""Factory registries for strategies and connectors.

These registries decouple configuration/type strings from concrete implementations, and
provide extension points for custom strategies and connectors without modifying the core
orchestrator.
"""

import logging
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from polaris.core.factory_plugins import ConnectorPluginLoader
from polaris.tools import register_tool_factory

if TYPE_CHECKING:
    from polaris.abstractions import (
        AdaptationStrategy,
        Connector,
        KnowledgeStore,
        Logger,
        MetricsCollector,
        WorldModel,
    )

from polaris.core.registry import ConnectorRegistry

# Type aliases for factory callables. We intentionally keep the
# config parameters typed as Any to avoid import cycles with the
# configuration module.
ConnectorFactory = Callable[[Any, "Logger", Optional["MetricsCollector"]], "Connector"]
ConnectorConfigValidator = Callable[[Dict[str, Any]], None]
StrategyFactory = Callable[
    [
        Any,
        "Logger",
        Optional["MetricsCollector"],
        "KnowledgeStore",
        "WorldModel",
        ConnectorRegistry,
    ],
    "AdaptationStrategy",
]


_CONNECTOR_FACTORIES: Dict[str, ConnectorFactory] = {}
_CONNECTOR_CONFIG_VALIDATORS: Dict[str, ConnectorConfigValidator] = {}
_STRATEGY_FACTORIES: Dict[str, StrategyFactory] = {}

CONNECTOR_PLUGIN_ENTRY_POINT_GROUP = "polaris.connectors"

_LOGGER = logging.getLogger("polaris.core.factories")


@dataclass
class _FactoryRegistryState:
    """Encapsulate mutable factory/loader state for safer lazy initialization."""

    connector_factories: Dict[str, ConnectorFactory] = field(default_factory=dict)
    connector_config_validators: Dict[str, ConnectorConfigValidator] = field(default_factory=dict)
    strategy_factories: Dict[str, StrategyFactory] = field(default_factory=dict)
    plugin_loader: Optional[ConnectorPluginLoader] = None
    factories_registered: bool = False
    init_lock: Lock = field(default_factory=Lock)


_STATE = _FactoryRegistryState(
    connector_factories=_CONNECTOR_FACTORIES,
    connector_config_validators=_CONNECTOR_CONFIG_VALIDATORS,
    strategy_factories=_STRATEGY_FACTORIES,
)


def _get_plugin_loader() -> ConnectorPluginLoader:
    """Return singleton plugin loader bound to current factory registries."""
    if _STATE.plugin_loader is None:
        _STATE.plugin_loader = ConnectorPluginLoader(
            register_connector_factory=register_connector_factory,
            register_connector_config_validator=register_connector_config_validator,
            register_tool_factory=register_tool_factory,
            entry_point_group=CONNECTOR_PLUGIN_ENTRY_POINT_GROUP,
            logger=_LOGGER,
        )
    return _STATE.plugin_loader


def _ensure_factories_registered() -> None:
    """Ensure default factories are registered (lazy initialization)."""
    with _STATE.init_lock:
        if _STATE.factories_registered:
            return

        _register_default_connector_factories()
        _register_default_strategy_factories()
        _STATE.factories_registered = True
        _get_plugin_loader().discover_entry_points()


def _reset_factory_state_for_tests() -> None:
    """Reset mutable registry/plugin-loader state for test isolation.

    This is an internal test utility and not part of the public runtime API.
    """
    with _STATE.init_lock:
        _STATE.connector_factories.clear()
        _STATE.connector_config_validators.clear()
        _STATE.strategy_factories.clear()
        _STATE.plugin_loader = None
        _STATE.factories_registered = False


def discover_connector_plugins(plugin_imports: Optional[List[str]] = None) -> List[str]:
    """Discover connector plugins from entry points and explicit import paths.

    Args:
        plugin_imports: Optional list of module paths to import. Imported modules
            can self-register connectors or expose a ``register_polaris_plugins``
            callable.

    Returns:
        List of explicit plugin module paths loaded during this call.
    """
    _ensure_factories_registered()
    return _get_plugin_loader().discover_explicit_plugins(plugin_imports)


def register_connector_factory(connector_type: str, factory: ConnectorFactory) -> None:
    """Register or override a connector factory for a given type string."""
    _STATE.connector_factories[connector_type] = factory


def register_connector_config_validator(
    connector_type: str,
    validator: ConnectorConfigValidator,
) -> None:
    """Register connector-specific configuration validator hook."""
    _STATE.connector_config_validators[connector_type] = validator


def get_connector_config_validator(connector_type: str) -> Optional[ConnectorConfigValidator]:
    """Get a connector configuration validator by connector type."""
    _ensure_factories_registered()
    return _STATE.connector_config_validators.get(connector_type)


def get_connector_factory(connector_type: str) -> Optional[ConnectorFactory]:
    """Get a connector factory by type."""
    _ensure_factories_registered()
    return _STATE.connector_factories.get(connector_type)


def get_strategy_factory(strategy_type: str) -> Optional[StrategyFactory]:
    """Get a strategy factory by type."""
    _ensure_factories_registered()
    return _STATE.strategy_factories.get(strategy_type)


def registered_connector_types() -> List[str]:
    """Return a sorted list of all registered connector types."""
    _ensure_factories_registered()
    return sorted(_STATE.connector_factories.keys())


def register_strategy_factory(strategy_type: str, factory: StrategyFactory) -> None:
    """Register or override a strategy factory for a given type string."""
    _STATE.strategy_factories[strategy_type] = factory


def registered_strategy_types() -> List[str]:
    """Return a sorted list of all registered strategy types."""
    _ensure_factories_registered()
    return sorted(_STATE.strategy_factories.keys())


# ---------------------------------------------------------------------------
# Default factories for built-in connectors and strategies
# ---------------------------------------------------------------------------


def _register_default_connector_factories() -> None:
    """Register factories for built-in connector types."""
    from polaris.core.factory_defaults.connectors import (
        register_default_connector_factories as register_default_connector_factories_impl,
    )

    register_default_connector_factories_impl(
        register_connector_factory,
        register_connector_config_validator,
    )


def _register_default_strategy_factories() -> None:
    """Register factories for built-in strategy types."""
    from polaris.core.factory_defaults.strategies import (
        register_default_strategy_factories as register_default_strategy_factories_impl,
    )

    register_default_strategy_factories_impl(
        register_strategy_factory,
        get_strategy_factory,
    )


# Register built-in factories lazily when first needed
# _register_default_connector_factories()
# _register_default_strategy_factories()
