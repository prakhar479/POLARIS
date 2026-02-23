"""Factory registries for strategies and connectors.

These registries decouple configuration/type strings from concrete
implementations, and provide extension points for custom strategies
and connectors without modifying the core orchestrator.
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import polaris.infrastructure.llm as _llm

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
_STRATEGY_FACTORIES: Dict[str, StrategyFactory] = {}


# Global flag to track if factories have been registered
_factories_registered = False


def _ensure_factories_registered() -> None:
    """Ensure default factories are registered (lazy initialization)."""
    global _factories_registered
    if _factories_registered:
        return

    _register_default_connector_factories()
    _register_default_strategy_factories()
    _factories_registered = True


def register_connector_factory(connector_type: str, factory: ConnectorFactory) -> None:
    """Register or override a connector factory for a given type string."""
    _CONNECTOR_FACTORIES[connector_type] = factory


def get_connector_factory(connector_type: str) -> Optional[ConnectorFactory]:
    """Get a connector factory by type."""
    _ensure_factories_registered()
    return _CONNECTOR_FACTORIES.get(connector_type)


def get_strategy_factory(strategy_type: str) -> Optional[StrategyFactory]:
    """Get a strategy factory by type."""
    _ensure_factories_registered()
    return _STRATEGY_FACTORIES.get(strategy_type)


def registered_connector_types() -> List[str]:
    """Return a sorted list of all registered connector types."""
    _ensure_factories_registered()
    return sorted(_CONNECTOR_FACTORIES.keys())


def register_strategy_factory(strategy_type: str, factory: StrategyFactory) -> None:
    """Register or override a strategy factory for a given type string."""
    _STRATEGY_FACTORIES[strategy_type] = factory


def registered_strategy_types() -> List[str]:
    """Return a sorted list of all registered strategy types."""
    _ensure_factories_registered()
    return sorted(_STRATEGY_FACTORIES.keys())


# ---------------------------------------------------------------------------
# Default factories for built-in connectors and strategies
# ---------------------------------------------------------------------------


def _register_default_connector_factories() -> None:
    """Register factories for built-in connector types."""
    # Import here to avoid circular imports
    from polaris.connectors import SWIMConnector, WildfireConnector

    def _swim_factory(
        system_cfg: Any, logger: "Logger", metrics: Optional["MetricsCollector"]
    ) -> "Connector":
        host = system_cfg.connection.get("host", "localhost")
        port = system_cfg.connection.get("port", 4242)
        return SWIMConnector(host=host, port=port, logger=logger, metrics=metrics)

    register_connector_factory("swim", _swim_factory)

    def _wildfire_factory(
        system_cfg: Any, logger: "Logger", metrics: Optional["MetricsCollector"]
    ) -> "Connector":
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
    # Import here to avoid circular imports
    from polaris.strategies import (
        AgenticLLMStrategy,
        HybridStrategy,
        LLMReasoningStrategy,
        ThresholdReactiveStrategy,
    )

    def _threshold_factory(
        strategy_cfg: Any,
        logger: "Logger",
        metrics: Optional["MetricsCollector"],
        knowledge_store: "KnowledgeStore",
        world_model: "WorldModel",
        registry: ConnectorRegistry,
    ) -> "AdaptationStrategy":
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
        logger: "Logger",
        metrics: Optional["MetricsCollector"],
        knowledge_store: "KnowledgeStore",
        world_model: "WorldModel",
        registry: ConnectorRegistry,
    ) -> "AdaptationStrategy":
        if not strategy_cfg.llm_reasoning:
            raise ValueError("LLM strategy requires 'llm_reasoning' configuration section")

        resilience_cfg = strategy_cfg.llm_reasoning.get("resilience")
        provider = strategy_cfg.llm_reasoning.get("provider", "google")
        llm_client = _llm.create_llm_client(provider, resilience=resilience_cfg)

        return LLMReasoningStrategy(
            llm_client=llm_client,
            system_description=strategy_cfg.llm_reasoning.get(
                "system_description", "Managed system"
            ),
            adaptation_goals=strategy_cfg.llm_reasoning.get(
                "adaptation_goals", "Maintain optimal performance"
            ),
            temperature=strategy_cfg.llm_reasoning.get("temperature", 0.1),
            system_prompt=strategy_cfg.llm_reasoning.get("system_prompt"),
            per_system_prompts=strategy_cfg.llm_reasoning.get("per_system_prompts"),
            logger=logger,
            metrics=metrics,
        )

    register_strategy_factory("llm_reasoning", _llm_reasoning_factory)

    def _hybrid_factory(
        strategy_cfg: Any,
        logger: "Logger",
        metrics: Optional["MetricsCollector"],
        knowledge_store: "KnowledgeStore",
        world_model: "WorldModel",
        registry: ConnectorRegistry,
    ) -> "AdaptationStrategy":
        hybrid_conf = strategy_cfg.hybrid or {}
        selection_mode = hybrid_conf.get("selection_mode", "confidence")
        min_confidence = float(hybrid_conf.get("min_confidence", 0.7))
        sub_defs = hybrid_conf.get("strategies", [])

        sub_strategies: List[Tuple["AdaptationStrategy", float]] = []
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
                sub_llm = LLMReasoningStrategy(
                    llm_client=llm_client,
                    system_description=llm_cfg.get("system_description", "Managed system"),
                    adaptation_goals=llm_cfg.get(
                        "adaptation_goals", "Maintain optimal performance"
                    ),
                    temperature=llm_cfg.get("temperature", 0.1),
                    system_prompt=llm_cfg.get("system_prompt"),
                    per_system_prompts=llm_cfg.get("per_system_prompts"),
                    logger=logger,
                    metrics=metrics,
                )
                sub_strategies.append((sub_llm, priority))

            elif s_type == "agentic_llm":
                agent_cfg = s.get("agentic_llm", {}) or {}
                steps_limit = int(agent_cfg.get("steps_limit", 3))
                temperature = float(agent_cfg.get("temperature", 0.1))
                allowed_tools = None
                tools_cfg = agent_cfg.get("tools")
                if isinstance(tools_cfg, dict):
                    allowed_tools = tools_cfg.get("enabled")

                provider = agent_cfg.get("provider", "google")
                llm_client = _llm.create_llm_client(
                    provider, resilience=agent_cfg.get("resilience")
                )

                sub_agent = AgenticLLMStrategy(
                    llm_client=llm_client,
                    knowledge_store=knowledge_store,
                    world_model=world_model,
                    connector_getter=registry.get,
                    steps_limit=steps_limit,
                    temperature=temperature,
                    allowed_tools=allowed_tools,
                    system_prompt=agent_cfg.get("system_prompt"),
                    per_system_prompts=agent_cfg.get("per_system_prompts"),
                    logger=logger,
                    metrics=metrics,
                )
                sub_strategies.append((sub_agent, priority))

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
        logger: "Logger",
        metrics: Optional["MetricsCollector"],
        knowledge_store: "KnowledgeStore",
        world_model: "WorldModel",
        registry: ConnectorRegistry,
    ) -> "AdaptationStrategy":
        agent_conf = strategy_cfg.agentic_llm or {}
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
            system_prompt=agent_conf.get("system_prompt"),
            per_system_prompts=agent_conf.get("per_system_prompts"),
            logger=logger,
            metrics=metrics,
        )

    register_strategy_factory("agentic_llm", _agentic_llm_factory)


# Register built-in factories lazily when first needed
# _register_default_connector_factories()
# _register_default_strategy_factories()
