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
from polaris.infrastructure.constants import DEFAULT_CONNECTOR_TIMEOUT, DEFAULT_WILDFIRE_PORT

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
    from polaris.connectors import (
        KubernetesConnector,
        SUAVEConnector,
        SWIMConnector,
        WildfireConnector,
    )

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
            port = system_cfg.connection.get("port", DEFAULT_WILDFIRE_PORT)
            base_url = f"http://{host}:{port}"

        return WildfireConnector(
            base_url=base_url,
            system_id=system_cfg.id,
            timeout=system_cfg.connection.get("timeout", DEFAULT_CONNECTOR_TIMEOUT),
            session_id=system_cfg.connection.get("session_id"),
            logger=logger,
            metrics=metrics,
        )

    register_connector_factory("wildfire", _wildfire_factory)

    def _suave_factory(
        system_cfg: Any, logger: "Logger", metrics: Optional["MetricsCollector"]
    ) -> "Connector":
        host = system_cfg.connection.get("host", "localhost")
        port = system_cfg.connection.get("port", 9090)
        connect_timeout = system_cfg.connection.get("connect_timeout", 10.0)
        service_timeout = system_cfg.connection.get("service_timeout", 5.0)

        return SUAVEConnector(
            host=host,
            port=port,
            connect_timeout=connect_timeout,
            service_timeout=service_timeout,
            logger=logger,
            metrics=metrics,
        )

    register_connector_factory("suave", _suave_factory)

    def _kubernetes_factory(
        system_cfg: Any, logger: "Logger", metrics: Optional["MetricsCollector"]
    ) -> "Connector":
        kubeconfig = system_cfg.connection.get("kubeconfig_path")
        in_cluster = system_cfg.connection.get("in_cluster", False)
        namespace = system_cfg.connection.get("namespace", "default")
        return KubernetesConnector(
            kubeconfig_path=kubeconfig,
            in_cluster=in_cluster,
            namespace=namespace,
            logger=logger,
            metrics=metrics,
        )

    register_connector_factory("kubernetes", _kubernetes_factory)


def _register_default_strategy_factories() -> None:
    """Register factories for built-in strategy types."""
    # Import here to avoid circular imports
    from polaris.strategies import (
        AgenticLLMStrategy,
        HybridStrategy,
        LLMReasoningStrategy,
        MultiAgentStrategy,
        SuaveThresholdStrategy,
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

            elif s_type == "suave_threshold":
                st_cfg = s.get("suave_threshold", {}) or {}

                vis_metric_names = st_cfg.get("visibility_metric_names")
                thr_metric_names = st_cfg.get("thruster_failure_metric_names")
                perf_metric_names = st_cfg.get("performance_metric_names")

                trigger_cfg = st_cfg.get("trigger", {}) if isinstance(st_cfg.get("trigger"), dict) else {}
                mode_cfg = st_cfg.get("modes", {}) if isinstance(st_cfg.get("modes"), dict) else {}
                vis_mode_cfg = (
                    mode_cfg.get("visibility", {})
                    if isinstance(mode_cfg.get("visibility"), dict)
                    else {}
                )
                motion_mode_cfg = (
                    mode_cfg.get("maintain_motion", {})
                    if isinstance(mode_cfg.get("maintain_motion"), dict)
                    else {}
                )

                sub_suave = SuaveThresholdStrategy(
                    visibility_metric_names=vis_metric_names,
                    thruster_failure_metric_names=thr_metric_names,
                    performance_metric_names=perf_metric_names,
                    trigger_visibility_below=float(
                        trigger_cfg.get("visibility_below", 1.0)
                    ),
                    trigger_performance_at_or_above=float(
                        trigger_cfg.get("performance_above_or_equal", 1.0)
                    ),
                    trigger_thruster_failure_at_or_above=float(
                        trigger_cfg.get("thruster_failure_at_or_above", 0.5)
                    ),
                    visibility_medium_at_or_above=float(
                        vis_mode_cfg.get("medium_at_or_above", 1.0)
                    ),
                    visibility_high_at_or_above=float(
                        vis_mode_cfg.get("high_at_or_above", 2.0)
                    ),
                    search_path_function_node=str(
                        mode_cfg.get("search_path_function_node", "f_generate_search_path")
                    ),
                    maintain_motion_function_node=str(
                        mode_cfg.get("maintain_motion_function_node", "f_maintain_motion")
                    ),
                    spiral_low_mode=str(vis_mode_cfg.get("low_mode", "fd_spiral_low")),
                    spiral_medium_mode=str(
                        vis_mode_cfg.get("medium_mode", "fd_spiral_medium")
                    ),
                    spiral_high_mode=str(vis_mode_cfg.get("high_mode", "fd_spiral_high")),
                    recover_thrusters_mode=str(
                        motion_mode_cfg.get("failure_mode", "fd_recover_thrusters")
                    ),
                    all_thrusters_mode=str(
                        motion_mode_cfg.get("healthy_mode", "fd_all_thrusters")
                    ),
                    cooldown_seconds=int(st_cfg.get("cooldown_seconds", 0)),
                    logger=logger,
                    metrics=metrics,
                )
                sub_strategies.append((sub_suave, priority))

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

            elif s_type == "multi_agent":
                from polaris.strategies.multi_agent import AgentConfig

                ma_cfg = s.get("multi_agent", {}) or {}
                ma_provider = ma_cfg.get("provider", "google")
                ma_shared_llm = _llm.create_llm_client(
                    ma_provider, resilience=ma_cfg.get("resilience")
                )

                def _build_agent_cfg_hybrid(
                    role_cfg: Optional[dict], ma_provider: str = ma_provider
                ) -> Optional[AgentConfig]:
                    if not isinstance(role_cfg, dict) or not role_cfg:
                        return None
                    rp = role_cfg.get("provider")
                    rr = role_cfg.get("resilience")
                    rc = None
                    if rp:
                        rc = _llm.create_llm_client(rp, resilience=rr)
                    elif rr:
                        rc = _llm.create_llm_client(ma_provider, resilience=rr)
                    return AgentConfig(
                        llm_client=rc,
                        temperature=role_cfg.get("temperature"),
                        system_prompt=role_cfg.get("system_prompt"),
                        max_tokens=role_cfg.get("max_tokens"),
                        steps_limit=role_cfg.get("steps_limit"),
                        allowed_tools=role_cfg.get("tools"),
                    )

                sub_ma = MultiAgentStrategy(
                    llm_client=ma_shared_llm,
                    knowledge_store=knowledge_store,
                    world_model=world_model,
                    temperature=float(ma_cfg.get("temperature", 0.1)),
                    system_description=ma_cfg.get(
                        "system_description", "A generic managed cloud system"
                    ),
                    steps_limit=int(ma_cfg.get("steps_limit", 3)),
                    allowed_tools=ma_cfg.get("tools"),
                    diagnostician_config=_build_agent_cfg_hybrid(ma_cfg.get("diagnostician")),
                    planner_config=_build_agent_cfg_hybrid(ma_cfg.get("planner")),
                    validator_config=_build_agent_cfg_hybrid(ma_cfg.get("validator")),
                    agent_prompts=ma_cfg.get("agent_prompts"),
                    logger=logger,
                    metrics=metrics,
                )
                sub_strategies.append((sub_ma, priority))

            else:
                # Fallback to threshold for unknown types
                sub = ThresholdReactiveStrategy(logger=logger, metrics=metrics)
                sub_strategies.append((sub, priority))

        if not sub_strategies:
            # Safety fallback
            return ThresholdReactiveStrategy(logger=logger, metrics=metrics)

        cooldown_seconds = int(hybrid_conf.get("cooldown_seconds", 0))

        return HybridStrategy(
            strategies=sub_strategies,
            selection_mode=selection_mode,
            min_confidence=min_confidence,
            cooldown_seconds=cooldown_seconds,
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

    def _multi_agent_factory(
        strategy_cfg: Any,
        logger: "Logger",
        metrics: Optional["MetricsCollector"],
        knowledge_store: "KnowledgeStore",
        world_model: "WorldModel",
        registry: ConnectorRegistry,
    ) -> "AdaptationStrategy":
        from polaris.strategies.multi_agent import AgentConfig

        agent_conf = strategy_cfg.multi_agent or {}
        temperature = float(agent_conf.get("temperature", 0.1))
        system_description = agent_conf.get("system_description", "A generic managed cloud system")

        provider = agent_conf.get("provider", "google")
        shared_llm = _llm.create_llm_client(provider, resilience=agent_conf.get("resilience"))

        def _build_agent_config(role_cfg: Optional[dict]) -> Optional[AgentConfig]:
            """Build an AgentConfig from a per-agent config dict."""
            if not isinstance(role_cfg, dict) or not role_cfg:
                return None
            role_provider = role_cfg.get("provider")
            role_resilience = role_cfg.get("resilience")
            role_client = None
            if role_provider:
                role_client = _llm.create_llm_client(role_provider, resilience=role_resilience)
            elif role_resilience:
                # Same provider as shared but different resilience
                role_client = _llm.create_llm_client(provider, resilience=role_resilience)
            return AgentConfig(
                llm_client=role_client,
                temperature=role_cfg.get("temperature"),
                system_prompt=role_cfg.get("system_prompt"),
                max_tokens=role_cfg.get("max_tokens"),
                steps_limit=role_cfg.get("steps_limit"),
                allowed_tools=role_cfg.get("tools"),
            )

        diagnostician_config = _build_agent_config(agent_conf.get("diagnostician"))
        planner_config = _build_agent_config(agent_conf.get("planner"))
        validator_config = _build_agent_config(agent_conf.get("validator"))

        # Top-level agent_prompts dict alternative (shorthand)
        agent_prompts: Optional[Dict[str, str]] = agent_conf.get("agent_prompts")

        return MultiAgentStrategy(
            llm_client=shared_llm,
            knowledge_store=knowledge_store,
            world_model=world_model,
            temperature=temperature,
            system_description=system_description,
            steps_limit=int(agent_conf.get("steps_limit", 3)),
            allowed_tools=agent_conf.get("tools"),
            diagnostician_config=diagnostician_config,
            planner_config=planner_config,
            validator_config=validator_config,
            agent_prompts=agent_prompts,
            logger=logger,
            metrics=metrics,
        )

    register_strategy_factory("multi_agent", _multi_agent_factory)


# Register built-in factories lazily when first needed
# _register_default_connector_factories()
# _register_default_strategy_factories()
