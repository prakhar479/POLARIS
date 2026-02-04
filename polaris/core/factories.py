"""Factory registries for strategies and connectors.

These registries decouple configuration/type strings from concrete
implementations, and provide extension points for custom strategies
and connectors without modifying the core orchestrator.
"""

from typing import Any, Callable, Dict, List, Optional

from polaris.abstractions import (
    Connector,
    AdaptationStrategy,
    Logger,
    MetricsCollector,
    KnowledgeStore,
    WorldModel,
)
from polaris.core.registry import ConnectorRegistry


# Type aliases for factory callables. We intentionally keep the
# config parameters typed as Any to avoid import cycles with the
# configuration module.
ConnectorFactory = Callable[[Any, Logger, Optional[MetricsCollector]], Connector]
StrategyFactory = Callable[
    [Any, Logger, Optional[MetricsCollector], KnowledgeStore, WorldModel, ConnectorRegistry],
    AdaptationStrategy,
]


_CONNECTOR_FACTORIES: Dict[str, ConnectorFactory] = {}
_STRATEGY_FACTORIES: Dict[str, StrategyFactory] = {}


def register_connector_factory(connector_type: str, factory: ConnectorFactory) -> None:
    """Register or override a connector factory for a given type string."""
    _CONNECTOR_FACTORIES[connector_type] = factory


def get_connector_factory(connector_type: str) -> Optional[ConnectorFactory]:
    """Return the connector factory for the given type, if any."""
    return _CONNECTOR_FACTORIES.get(connector_type)


def registered_connector_types() -> List[str]:
    """Return a sorted list of all registered connector types."""
    return sorted(_CONNECTOR_FACTORIES.keys())


def register_strategy_factory(strategy_type: str, factory: StrategyFactory) -> None:
    """Register or override a strategy factory for a given type string."""
    _STRATEGY_FACTORIES[strategy_type] = factory


def get_strategy_factory(strategy_type: str) -> Optional[StrategyFactory]:
    """Return the strategy factory for the given type, if any."""
    return _STRATEGY_FACTORIES.get(strategy_type)


def registered_strategy_types() -> List[str]:
    """Return a sorted list of all registered strategy types."""
    return sorted(_STRATEGY_FACTORIES.keys())


# ---------------------------------------------------------------------------
# Default factories for built-in connectors and strategies
# ---------------------------------------------------------------------------

from polaris.connectors import SWIMConnector, WildfireConnector
from polaris.strategies import (
    ThresholdReactiveStrategy,
    LLMReasoningStrategy,
    HybridStrategy,
    AgenticLLMStrategy,
)
import polaris.infrastructure.llm as _llm


def _register_default_connector_factories() -> None:
    """Register factories for built-in connector types."""

    def _swim_factory(system_cfg: Any, logger: Logger, metrics: Optional[MetricsCollector]) -> Connector:
        host = system_cfg.connection.get("host", "localhost")
        port = system_cfg.connection.get("port", 4242)
        return SWIMConnector(host=host, port=port, logger=logger, metrics=metrics)

    register_connector_factory("swim", _swim_factory)

    def _wildfire_factory(system_cfg: Any, logger: Logger, metrics: Optional[MetricsCollector]) -> Connector:
        base_url = system_cfg.connection.get("base_url")
        if not base_url:
            host = system_cfg.connection.get("host", "localhost")
            port = system_cfg.connection.get("port", 5000)
            base_url = f"http://{host}:{port}"

        return WildfireConnector(
            base_url=base_url,
            system_id=system_cfg.id,
            timeout=system_cfg.connection.get("timeout", 10.0),
            session_id=system_cfg.connection.get("session_id"),
            logger=logger,
            metrics=metrics,
        )

    register_connector_factory("wildfire", _wildfire_factory)


def _register_default_strategy_factories() -> None:
    """Register factories for built-in strategy types."""

    def _threshold_factory(
        strategy_cfg: Any,
        logger: Logger,
        metrics: Optional[MetricsCollector],
        knowledge_store: KnowledgeStore,
        world_model: WorldModel,
        registry: ConnectorRegistry,
    ) -> AdaptationStrategy:
        if strategy_cfg.threshold:
            thresholds = {}
            threshold_data = strategy_cfg.threshold.get("thresholds", {})
            for metric, values in threshold_data.items():
                thresholds[metric] = values

            cooldown = strategy_cfg.threshold.get("cooldown_seconds", 60)
            return ThresholdReactiveStrategy(
                thresholds=thresholds,
                cooldown_seconds=cooldown,
                logger=logger,
                metrics=metrics,
            )

        return ThresholdReactiveStrategy(logger=logger, metrics=metrics)

    register_strategy_factory("threshold", _threshold_factory)

    def _llm_reasoning_factory(
        strategy_cfg: Any,
        logger: Logger,
        metrics: Optional[MetricsCollector],
        knowledge_store: KnowledgeStore,
        world_model: WorldModel,
        registry: ConnectorRegistry,
    ) -> AdaptationStrategy:
        if not strategy_cfg.llm:
            raise ValueError("LLM strategy requires 'llm_reasoning' configuration section")

        resilience_cfg = strategy_cfg.llm.get("resilience")
        provider = strategy_cfg.llm.get("provider", "google")
        llm_client = _llm.create_llm_client(provider, resilience=resilience_cfg)

        return LLMReasoningStrategy(
            llm_client=llm_client,
            system_description=strategy_cfg.llm.get("system_description", "Managed system"),
            adaptation_goals=strategy_cfg.llm.get("adaptation_goals", "Maintain optimal performance"),
            temperature=strategy_cfg.llm.get("temperature", 0.1),
            system_prompt=strategy_cfg.llm.get("system_prompt"),
            per_system_prompts=strategy_cfg.llm.get("per_system_prompts"),
            logger=logger,
            metrics=metrics,
        )

    register_strategy_factory("llm_reasoning", _llm_reasoning_factory)

    def _hybrid_factory(
        strategy_cfg: Any,
        logger: Logger,
        metrics: Optional[MetricsCollector],
        knowledge_store: KnowledgeStore,
        world_model: WorldModel,
        registry: ConnectorRegistry,
    ) -> AdaptationStrategy:
        hybrid_conf = strategy_cfg.hybrid or {}
        selection_mode = hybrid_conf.get("selection_mode", "confidence")
        min_confidence = float(hybrid_conf.get("min_confidence", 0.7))
        sub_defs = hybrid_conf.get("strategies", [])

        sub_strategies: List[tuple[AdaptationStrategy, float]] = []
        for s in sub_defs:
            s_type = s.get("type", "threshold")
            priority = float(s.get("priority", 0.5))

            if s_type == "threshold":
                thresholds = None
                cooldown = 60
                if "threshold" in s and isinstance(s["threshold"], dict):
                    th = s["threshold"]
                    thresholds = th.get("thresholds")
                    cooldown = th.get("cooldown_seconds", cooldown)

                sub = ThresholdReactiveStrategy(
                    thresholds=thresholds,
                    cooldown_seconds=cooldown,
                    logger=logger,
                    metrics=metrics,
                )
                sub_strategies.append((sub, priority))

            elif s_type == "llm_reasoning":
                llm_cfg = s.get("llm_reasoning", {}) or {}
                provider = llm_cfg.get("provider", "google")
                llm_client = _llm.create_llm_client(provider, resilience=llm_cfg.get("resilience"))
                sub = LLMReasoningStrategy(
                    llm_client=llm_client,
                    system_description=llm_cfg.get("system_description", "Managed system"),
                    adaptation_goals=llm_cfg.get("adaptation_goals", "Maintain optimal performance"),
                    temperature=llm_cfg.get("temperature", 0.1),
                    system_prompt=llm_cfg.get("system_prompt"),
                    per_system_prompts=llm_cfg.get("per_system_prompts"),
                    logger=logger,
                    metrics=metrics,
                )
                sub_strategies.append((sub, priority))

            else:
                # Fallback to threshold for unknown types
                sub = ThresholdReactiveStrategy(logger=logger, metrics=metrics)
                sub_strategies.append((sub, priority))

        if not sub_strategies:
            # Safety fallback
            return ThresholdReactiveStrategy(logger=logger, metrics=metrics)

        return HybridStrategy(
            strategies=sub_strategies,
            selection_mode=selection_mode,
            min_confidence=min_confidence,
            logger=logger,
            metrics=metrics,
        )

    register_strategy_factory("hybrid", _hybrid_factory)

    def _agentic_llm_factory(
        strategy_cfg: Any,
        logger: Logger,
        metrics: Optional[MetricsCollector],
        knowledge_store: KnowledgeStore,
        world_model: WorldModel,
        registry: ConnectorRegistry,
    ) -> AdaptationStrategy:
        agent_conf = strategy_cfg.agentic or {}
        steps_limit = int(agent_conf.get("steps_limit", 3))
        temperature = float(agent_conf.get("temperature", 0.1))
        allowed_tools = None
        tools_cfg = agent_conf.get("tools")
        if isinstance(tools_cfg, dict):
            allowed_tools = tools_cfg.get("enabled")

        provider = agent_conf.get("provider", "google")
        llm_client = _llm.create_llm_client(provider, resilience=agent_conf.get("resilience"))

        return AgenticLLMStrategy(
            llm_client=llm_client,
            knowledge_store=knowledge_store,
            world_model=world_model,
            connector_getter=registry.get,
            steps_limit=steps_limit,
            temperature=temperature,
            allowed_tools=allowed_tools,
            logger=logger,
            metrics=metrics,
        )

    register_strategy_factory("agentic_llm", _agentic_llm_factory)


# Register built-in factories at import time
_register_default_connector_factories()
_register_default_strategy_factories()
