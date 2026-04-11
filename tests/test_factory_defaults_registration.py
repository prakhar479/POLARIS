"""Registration contract tests for extracted default factory modules."""

from typing import Any, Callable, Dict, Optional

from polaris.core.factory_defaults.connectors import register_default_connector_factories
from polaris.core.factory_defaults.strategies import register_default_strategy_factories


def test_default_connector_registrations_are_complete():
    connector_factories: Dict[str, Callable[..., Any]] = {}
    connector_validators: Dict[str, Callable[..., Any]] = {}

    def register_connector_factory(name: str, factory: Callable[..., Any]) -> None:
        connector_factories[name] = factory

    def register_connector_validator(name: str, validator: Callable[..., Any]) -> None:
        connector_validators[name] = validator

    register_default_connector_factories(
        register_connector_factory,
        register_connector_validator,
    )

    assert set(connector_factories) == {"swim", "wildfire", "suave", "kubernetes"}
    assert set(connector_validators) == {"swim", "wildfire", "kubernetes"}


def test_default_strategy_registrations_are_complete():
    strategy_factories: Dict[str, Callable[..., Any]] = {}

    def register_strategy_factory(name: str, factory: Callable[..., Any]) -> None:
        strategy_factories[name] = factory

    def get_strategy_factory(name: str) -> Optional[Callable[..., Any]]:
        return strategy_factories.get(name)

    register_default_strategy_factories(
        register_strategy_factory,
        get_strategy_factory,
    )

    assert set(strategy_factories) == {
        "threshold",
        "llm_reasoning",
        "hybrid",
        "agentic_llm",
        "thread_agentic",
        "multi_agent",
    }
