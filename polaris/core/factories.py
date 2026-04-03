"""Factory registries for strategies and connectors.

These registries decouple configuration/type strings from concrete implementations, and
provide extension points for custom strategies and connectors without modifying the core
orchestrator.
"""

import importlib
import importlib.metadata
import inspect
from types import ModuleType
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
from polaris.infrastructure.constants import (
    DEFAULT_CONNECTOR_TIMEOUT,
    DEFAULT_WILDFIRE_PORT,
    MAX_PORT,
    MIN_PORT,
)

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

# Loaded plugin bookkeeping.
_entry_points_discovered = False
_loaded_plugin_modules: set[str] = set()
_loaded_entry_points: set[str] = set()


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
    _discover_connector_entry_points()


def _invoke_plugin_registration(registration_hook: Callable[..., Any]) -> None:
    """Invoke plugin registration hook with strict canonical signature."""
    signature = inspect.signature(registration_hook)
    names = set(signature.parameters.keys())
    required = {"register_connector_factory", "register_connector_config_validator"}
    if not required.issubset(names):
        raise TypeError(
            "register_polaris_plugins must accept keyword parameters "
            "'register_connector_factory' and 'register_connector_config_validator'"
        )

    registration_hook(
        register_connector_factory=register_connector_factory,
        register_connector_config_validator=register_connector_config_validator,
    )


def _activate_plugin(plugin: Any) -> None:
    """Activate a loaded plugin object if it exposes registration hooks."""
    registration_hook = getattr(plugin, "register_polaris_plugins", None)
    if callable(registration_hook):
        _invoke_plugin_registration(registration_hook)
        return

    if isinstance(plugin, ModuleType):
        # Module may have already self-registered via import side effects.
        return

    if callable(plugin):
        _invoke_plugin_registration(plugin)


def _discover_connector_entry_points() -> None:
    """Load connector plugins exposed via entry points once per process."""
    global _entry_points_discovered
    if _entry_points_discovered:
        return

    _entry_points_discovered = True

    try:
        entry_points = importlib.metadata.entry_points(group=CONNECTOR_PLUGIN_ENTRY_POINT_GROUP)
    except Exception:
        return

    for entry_point in entry_points:
        entry_point_id = f"{entry_point.group}:{entry_point.name}"
        if entry_point_id in _loaded_entry_points:
            continue

        try:
            plugin = entry_point.load()
            _activate_plugin(plugin)
            _loaded_entry_points.add(entry_point_id)
        except Exception:
            # Keep startup resilient when optional third-party plugins fail to load.
            continue


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

    loaded_now: List[str] = []
    for module_path in plugin_imports or []:
        normalized_path = module_path.strip()
        if not normalized_path or normalized_path in _loaded_plugin_modules:
            continue

        plugin_module = importlib.import_module(normalized_path)
        _activate_plugin(plugin_module)
        _loaded_plugin_modules.add(normalized_path)
        loaded_now.append(normalized_path)

    return loaded_now


def register_connector_factory(connector_type: str, factory: ConnectorFactory) -> None:
    """Register or override a connector factory for a given type string."""
    _CONNECTOR_FACTORIES[connector_type] = factory


def register_connector_config_validator(
    connector_type: str,
    validator: ConnectorConfigValidator,
) -> None:
    """Register connector-specific configuration validator hook."""
    _CONNECTOR_CONFIG_VALIDATORS[connector_type] = validator


def get_connector_config_validator(connector_type: str) -> Optional[ConnectorConfigValidator]:
    """Get a connector configuration validator by connector type."""
    _ensure_factories_registered()
    return _CONNECTOR_CONFIG_VALIDATORS.get(connector_type)


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
    from polaris.connectors import (
        KubernetesConnector,
        SUAVEConnector,
        SWIMConnector,
        WildfireConnector,
    )

    def _validate_port(port: Any, connector_name: str) -> None:
        if not isinstance(port, int):
            raise ValueError(f"{connector_name} connection port must be an integer")
        if not (MIN_PORT <= port <= MAX_PORT):
            raise ValueError(
                f"{connector_name} connection port must be between {MIN_PORT} and {MAX_PORT}"
            )

    def _validate_swim_connection(connection: Dict[str, Any]) -> None:
        if not isinstance(connection, dict):
            raise ValueError("SWIM connection config must be a dictionary")

        host = connection.get("host")
        if host is not None and not isinstance(host, str):
            raise ValueError("SWIM connection host must be a string")

        port = connection.get("port")
        if port is not None:
            _validate_port(port, "SWIM")

    def _validate_wildfire_connection(connection: Dict[str, Any]) -> None:
        if not isinstance(connection, dict):
            raise ValueError("Wildfire connection config must be a dictionary")

        base_url = connection.get("base_url")
        if base_url is not None and not isinstance(base_url, str):
            raise ValueError("Wildfire base_url must be a string")

        host = connection.get("host")
        if host is not None and not isinstance(host, str):
            raise ValueError("Wildfire host must be a string")

        port = connection.get("port")
        if port is not None:
            _validate_port(port, "Wildfire")

    def _validate_kubernetes_connection(connection: Dict[str, Any]) -> None:
        if not isinstance(connection, dict):
            raise ValueError("Kubernetes connection config must be a dictionary")

        kubeconfig_path = connection.get("kubeconfig_path")
        if kubeconfig_path is not None and not isinstance(kubeconfig_path, str):
            raise ValueError("Kubernetes kubeconfig_path must be a string")

        in_cluster = connection.get("in_cluster")
        if in_cluster is not None and not isinstance(in_cluster, bool):
            raise ValueError("Kubernetes in_cluster must be a boolean")

        namespace = connection.get("namespace")
        if namespace is not None and not isinstance(namespace, str):
            raise ValueError("Kubernetes namespace must be a string")

    def _validate_port(port: Any, connector_name: str) -> None:
        if not isinstance(port, int):
            raise ValueError(f"{connector_name} connection port must be an integer")
        if not (MIN_PORT <= port <= MAX_PORT):
            raise ValueError(
                f"{connector_name} connection port must be between {MIN_PORT} and {MAX_PORT}"
            )

    def _validate_swim_connection(connection: Dict[str, Any]) -> None:
        if not isinstance(connection, dict):
            raise ValueError("SWIM connection config must be a dictionary")

        host = connection.get("host")
        if host is not None and not isinstance(host, str):
            raise ValueError("SWIM connection host must be a string")

        port = connection.get("port")
        if port is not None:
            _validate_port(port, "SWIM")

    def _validate_wildfire_connection(connection: Dict[str, Any]) -> None:
        if not isinstance(connection, dict):
            raise ValueError("Wildfire connection config must be a dictionary")

        base_url = connection.get("base_url")
        if base_url is not None and not isinstance(base_url, str):
            raise ValueError("Wildfire base_url must be a string")

        host = connection.get("host")
        if host is not None and not isinstance(host, str):
            raise ValueError("Wildfire host must be a string")

        port = connection.get("port")
        if port is not None:
            _validate_port(port, "Wildfire")

    def _validate_kubernetes_connection(connection: Dict[str, Any]) -> None:
        if not isinstance(connection, dict):
            raise ValueError("Kubernetes connection config must be a dictionary")

        kubeconfig_path = connection.get("kubeconfig_path")
        if kubeconfig_path is not None and not isinstance(kubeconfig_path, str):
            raise ValueError("Kubernetes kubeconfig_path must be a string")

        in_cluster = connection.get("in_cluster")
        if in_cluster is not None and not isinstance(in_cluster, bool):
            raise ValueError("Kubernetes in_cluster must be a boolean")

        namespace = connection.get("namespace")
        if namespace is not None and not isinstance(namespace, str):
            raise ValueError("Kubernetes namespace must be a string")

    def _swim_factory(
        system_cfg: Any, logger: "Logger", metrics: Optional["MetricsCollector"]
    ) -> "Connector":
        host = system_cfg.connection.get("host", "localhost")
        port = system_cfg.connection.get("port", 4242)
        return SWIMConnector(host=host, port=port, logger=logger, metrics=metrics)

    register_connector_factory("swim", _swim_factory)
    register_connector_config_validator("swim", _validate_swim_connection)

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
    register_connector_config_validator("wildfire", _validate_wildfire_connection)

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
    register_connector_config_validator("kubernetes", _validate_kubernetes_connection)


def _register_default_strategy_factories() -> None:
    """Register factories for built-in strategy types."""
    # Import here to avoid circular imports
    from polaris.strategies import (
        AgenticLLMStrategy,
        HybridStrategy,
        LLMReasoningStrategy,
        MultiAgentStrategy,
        ThreadAgenticStrategy,
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
        params = getattr(strategy_cfg, "params", {})
        if params:
            thresholds = {}
            threshold_data = params.get("thresholds", {})
            for metric, values in threshold_data.items():
                thresholds[metric] = values

            cooldown = params.get("cooldown_seconds", 60)
            action_templates = params.get("action_templates")
            return ThresholdReactiveStrategy(
                thresholds=thresholds,
                action_templates=action_templates,
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
        params = getattr(strategy_cfg, "params", {})
        if not params:
            raise ValueError("LLM strategy requires configuration params")

        llm_reasoning_cfg = params
        provider = llm_reasoning_cfg.get("provider", "google")
        resilience_cfg = llm_reasoning_cfg.get("resilience")
        llm_kwargs = dict(llm_reasoning_cfg)
        llm_kwargs.pop("provider", None)
        llm_kwargs.pop("resilience", None)
        llm_client = _llm.create_llm_client(provider, resilience=resilience_cfg, **llm_kwargs)

        return LLMReasoningStrategy(
            llm_client=llm_client,
            system_description=params.get("system_description", "Managed system"),
            adaptation_goals=params.get(
                "adaptation_goals",
                "Maintain reliability, performance, and policy objectives",
            ),
            temperature=params.get("temperature", 0.1),
            system_prompt=params.get("system_prompt"),
            per_system_prompts=params.get("per_system_prompts"),
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
        hybrid_conf = getattr(strategy_cfg, "params", {})
        selection_mode = hybrid_conf.get("selection_mode", "confidence")
        min_confidence = float(hybrid_conf.get("min_confidence", 0.7))
        sub_defs = hybrid_conf.get("strategies", [])

        sub_strategies: List[Tuple["AdaptationStrategy", float]] = []
        for s in sub_defs:
            s_type = s.get("type", "threshold")
            priority = float(s.get("priority", 0.5))
            sub_params = s.get("params", {})
            if sub_params is None:
                sub_params = {}

            sub_factory = get_strategy_factory(s_type)
            if not sub_factory:
                logger.error(f"Unknown sub-strategy type '{s_type}' in hybrid config")
                continue

            if not isinstance(sub_params, dict):
                logger.error(
                    f"Invalid params for hybrid sub-strategy '{s_type}': params must be a dictionary"
                )
                continue

            from polaris.infrastructure.config import StrategyConfig

            try:
                sub_cfg = StrategyConfig(type=s_type, params=sub_params)
                sub_strategy = sub_factory(
                    sub_cfg, logger, metrics, knowledge_store, world_model, registry
                )
                sub_strategies.append((sub_strategy, priority))
            except Exception as exc:
                logger.error(f"Failed to build sub-strategy {s_type}: {exc}")
                continue

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
            raise ValueError("Hybrid strategy requires at least one valid sub-strategy")

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
        agent_conf = getattr(strategy_cfg, "params", {})
        steps_limit = int(agent_conf.get("steps_limit", 3))
        temperature = float(agent_conf.get("temperature", 0.1))
        decision_cooldown_seconds = float(agent_conf.get("decision_cooldown_seconds", 60.0))
        allowed_tools = None
        tools_cfg = agent_conf.get("tools")
        if isinstance(tools_cfg, dict):
            allowed_tools = tools_cfg.get("enabled")

        provider = agent_conf.get("provider", "google")
        resilience_cfg = agent_conf.get("resilience")
        llm_kwargs = dict(agent_conf)
        llm_kwargs.pop("provider", None)
        llm_kwargs.pop("resilience", None)
        llm_client = _llm.create_llm_client(provider, resilience=resilience_cfg, **llm_kwargs)

        return AgenticLLMStrategy(
            llm_client=llm_client,
            knowledge_store=knowledge_store,
            world_model=world_model,
            steps_limit=steps_limit,
            temperature=temperature,
            decision_cooldown_seconds=decision_cooldown_seconds,
            allowed_tools=allowed_tools,
            system_prompt=agent_conf.get("system_prompt"),
            per_system_prompts=agent_conf.get("per_system_prompts"),
            logger=logger,
            metrics=metrics,
        )

    register_strategy_factory("agentic_llm", _agentic_llm_factory)

    def _thread_agentic_factory(
        strategy_cfg: Any,
        logger: "Logger",
        metrics: Optional["MetricsCollector"],
        knowledge_store: "KnowledgeStore",
        world_model: "WorldModel",
        registry: ConnectorRegistry,
    ) -> "AdaptationStrategy":
        thread_conf = getattr(strategy_cfg, "params", {})
        steps_limit = int(thread_conf.get("steps_limit", 4))
        temperature = float(thread_conf.get("temperature", 0.1))
        max_thread_depth = int(thread_conf.get("max_thread_depth", 3))
        max_total_threads = int(thread_conf.get("max_total_threads", 16))
        child_timeout_seconds = float(thread_conf.get("child_timeout_seconds", 20.0))
        max_repeated_spawns = int(thread_conf.get("max_repeated_spawns", 2))
        assessment_cooldown_seconds = float(thread_conf.get("assessment_cooldown_seconds", 0.0))
        max_tool_result_chars = int(thread_conf.get("max_tool_result_chars", 1200))
        max_child_payload_chars = int(thread_conf.get("max_child_payload_chars", 800))
        phi_mode = str(thread_conf.get("phi_mode", "last_line"))
        phi_max_lines = int(thread_conf.get("phi_max_lines", 6))
        listen_token = str(thread_conf.get("listen_token", "=>"))
        return_token = str(thread_conf.get("return_token", "<="))

        allowed_tools = None
        tools_cfg = thread_conf.get("tools")
        if isinstance(tools_cfg, dict):
            allowed_tools = tools_cfg.get("enabled")

        provider = thread_conf.get("provider", "google")
        resilience_cfg = thread_conf.get("resilience")
        llm_kwargs = dict(thread_conf)
        llm_kwargs.pop("provider", None)
        llm_kwargs.pop("resilience", None)
        llm_client = _llm.create_llm_client(provider, resilience=resilience_cfg, **llm_kwargs)

        return ThreadAgenticStrategy(
            llm_client=llm_client,
            knowledge_store=knowledge_store,
            world_model=world_model,
            steps_limit=steps_limit,
            temperature=temperature,
            max_thread_depth=max_thread_depth,
            max_total_threads=max_total_threads,
            child_timeout_seconds=child_timeout_seconds,
            max_repeated_spawns=max_repeated_spawns,
            assessment_cooldown_seconds=assessment_cooldown_seconds,
            max_tool_result_chars=max_tool_result_chars,
            max_child_payload_chars=max_child_payload_chars,
            phi_mode=phi_mode,
            phi_max_lines=phi_max_lines,
            listen_token=listen_token,
            return_token=return_token,
            allowed_tools=allowed_tools,
            system_prompt=thread_conf.get("system_prompt"),
            per_system_prompts=thread_conf.get("per_system_prompts"),
            logger=logger,
            metrics=metrics,
        )

    register_strategy_factory("thread_agentic", _thread_agentic_factory)

    def _multi_agent_factory(
        strategy_cfg: Any,
        logger: "Logger",
        metrics: Optional["MetricsCollector"],
        knowledge_store: "KnowledgeStore",
        world_model: "WorldModel",
        registry: ConnectorRegistry,
    ) -> "AdaptationStrategy":
        from polaris.strategies.multi_agent import AgentConfig

        agent_conf = getattr(strategy_cfg, "params", {})
        temperature = float(agent_conf.get("temperature", 0.1))
        system_description = agent_conf.get("system_description", "Managed system")

        provider = agent_conf.get("provider", "google")
        resilience_cfg = agent_conf.get("resilience")
        llm_kwargs = dict(agent_conf)
        llm_kwargs.pop("provider", None)
        llm_kwargs.pop("resilience", None)
        shared_llm = _llm.create_llm_client(provider, resilience=resilience_cfg, **llm_kwargs)

        def _parse_tools_config(raw_tools: Any) -> Optional[List[str]]:
            if isinstance(raw_tools, list):
                return [tool for tool in raw_tools if isinstance(tool, str)]
            if isinstance(raw_tools, dict):
                enabled = raw_tools.get("enabled")
                if isinstance(enabled, list):
                    return [tool for tool in enabled if isinstance(tool, str)]
            return None

        def _build_agent_config(role_cfg: Optional[dict]) -> Optional[AgentConfig]:
            if not isinstance(role_cfg, dict) or not role_cfg:
                return None
            role_provider = role_cfg.get("provider")
            role_resilience = role_cfg.get("resilience")
            role_client = None
            if role_provider:
                role_kwargs = dict(role_cfg)
                role_kwargs.pop("provider", None)
                role_kwargs.pop("resilience", None)
                role_client = _llm.create_llm_client(
                    role_provider,
                    resilience=role_resilience,
                    **role_kwargs,
                )
            elif role_resilience:
                role_client = _llm.create_llm_client(
                    provider, resilience=role_resilience, **llm_kwargs
                )
            role_tools = _parse_tools_config(role_cfg.get("tools"))
            return AgentConfig(
                llm_client=role_client,
                temperature=role_cfg.get("temperature"),
                system_prompt=role_cfg.get("system_prompt"),
                max_tokens=role_cfg.get("max_tokens"),
                steps_limit=role_cfg.get("steps_limit"),
                allowed_tools=role_tools,
            )

        diagnostician_config = _build_agent_config(agent_conf.get("diagnostician"))
        planner_config = _build_agent_config(agent_conf.get("planner"))
        validator_config = _build_agent_config(agent_conf.get("validator"))
        shared_tools = _parse_tools_config(agent_conf.get("tools"))

        return MultiAgentStrategy(
            llm_client=shared_llm,
            knowledge_store=knowledge_store,
            world_model=world_model,
            temperature=temperature,
            system_description=system_description,
            steps_limit=int(agent_conf.get("steps_limit", 3)),
            allowed_tools=shared_tools,
            diagnostician_config=diagnostician_config,
            planner_config=planner_config,
            validator_config=validator_config,
            logger=logger,
            metrics=metrics,
        )

    register_strategy_factory("multi_agent", _multi_agent_factory)


# Register built-in factories lazily when first needed
# _register_default_connector_factories()
# _register_default_strategy_factories()
