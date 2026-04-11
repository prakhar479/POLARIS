"""Built-in strategy factory registrations."""

from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

import polaris.infrastructure.llm as _llm
from polaris.core.registry import ConnectorRegistry

if TYPE_CHECKING:
    from polaris.abstractions import (
        AdaptationStrategy,
        KnowledgeStore,
        Logger,
        MetricsCollector,
        WorldModel,
    )

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
RegisterStrategyFactory = Callable[[str, StrategyFactory], None]
GetStrategyFactory = Callable[[str], Optional[StrategyFactory]]


def register_default_strategy_factories(
    register_strategy_factory: RegisterStrategyFactory,
    get_strategy_factory: GetStrategyFactory,
) -> None:
    """Register factories for built-in strategy types."""
    # Import here to avoid circular imports
    from polaris.strategies import (
        AgenticLLMStrategy,
        HybridStrategy,
        LLMReasoningStrategy,
        MultiAgentStrategy,
        ThreadAgenticStrategy,
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
        llm_client = _llm.create_llm_client_from_config(llm_reasoning_cfg)

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
        max_tool_result_chars = int(agent_conf.get("max_tool_result_chars", 1200))
        native_tools_unsupported_policy = str(
            agent_conf.get("native_tools_unsupported_policy", "skip_cycle")
        )
        allowed_tools = None
        tools_cfg = agent_conf.get("tools")
        if isinstance(tools_cfg, list):
            allowed_tools = [tool for tool in tools_cfg if isinstance(tool, str)]
        elif isinstance(tools_cfg, dict):
            allowed_tools = tools_cfg.get("enabled")

        # Native tool calling: OpenAI-format function definitions (optional)
        native_tools = agent_conf.get("native_tools")

        llm_client = _llm.create_llm_client_from_config(agent_conf)

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
            native_tools=native_tools,
            max_tool_result_chars=max_tool_result_chars,
            native_tools_unsupported_policy=native_tools_unsupported_policy,
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
        if isinstance(tools_cfg, list):
            allowed_tools = [tool for tool in tools_cfg if isinstance(tool, str)]
        elif isinstance(tools_cfg, dict):
            allowed_tools = tools_cfg.get("enabled")

        llm_client = _llm.create_llm_client_from_config(thread_conf)

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
        max_tool_result_chars = int(agent_conf.get("max_tool_result_chars", 1200))

        provider = agent_conf.get("provider", "google")
        shared_llm = _llm.create_llm_client_from_config(agent_conf)

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
                role_client = _llm.create_llm_client_from_config(
                    role_cfg,
                    default_provider=str(provider),
                )
            elif role_resilience:
                role_llm_cfg = dict(agent_conf)
                role_llm_cfg["provider"] = provider
                role_llm_cfg["resilience"] = role_resilience
                role_client = _llm.create_llm_client_from_config(role_llm_cfg)
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
            max_tool_result_chars=max_tool_result_chars,
            allowed_tools=shared_tools,
            diagnostician_config=diagnostician_config,
            planner_config=planner_config,
            validator_config=validator_config,
            logger=logger,
            metrics=metrics,
        )

    register_strategy_factory("multi_agent", _multi_agent_factory)
