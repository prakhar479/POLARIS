"""Core domain component builders used by ComponentBuilder."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from polaris.infrastructure.constants import DEFAULT_MAX_STATES_PER_SYSTEM

if TYPE_CHECKING:
    from polaris.abstractions import (
        AdaptationStrategy,
        Connector,
        KnowledgeStore,
        Logger,
        MetaLearner,
        MetricsCollector,
        WorldModel,
    )
    from polaris.core.registry import ConnectorRegistry
    from polaris.infrastructure.config import PolarisConfig


def build_knowledge_store(
    config: "PolarisConfig",
    logger: "Logger",
    metrics: Optional["MetricsCollector"],
) -> "KnowledgeStore":
    """Create the knowledge store from canonical ``knowledge_store`` config."""
    ks_cfg: Dict[str, Any] = {}
    if isinstance(config.knowledge_store, dict):
        ks_cfg = config.knowledge_store

    ks_type = str(ks_cfg.get("type", "memory")).lower()

    max_states = 1000
    db_path: Optional[str] = None

    if ks_type == "sqlite":
        sqlite_cfg = ks_cfg.get("sqlite", {}) if isinstance(ks_cfg.get("sqlite"), dict) else {}
        db_path = sqlite_cfg.get("db_path") or ks_cfg.get("db_path")
        max_states = int(
            sqlite_cfg.get(
                "max_states_per_system",
                ks_cfg.get("max_states_per_system", DEFAULT_MAX_STATES_PER_SYSTEM),
            )
        )

        if not db_path:
            raise ValueError(
                "knowledge_store type 'sqlite' requires 'knowledge_store.sqlite.db_path' "
                "(or legacy 'knowledge_store.db_path')"
            )

        from polaris.knowledge.sqlite_store import SQLiteKnowledgeStore

        return SQLiteKnowledgeStore(
            db_path=db_path,
            max_states_per_system=max_states,
            logger=logger,
            metrics=metrics,
        )
    if ks_type == "memory":
        memory_cfg = ks_cfg.get("memory", {}) if isinstance(ks_cfg.get("memory"), dict) else {}
        max_states = int(
            memory_cfg.get("max_states_per_system", ks_cfg.get("max_states_per_system", 1000))
        )
    else:
        raise ValueError(f"Unknown knowledge store type '{ks_type}'")

    from polaris.knowledge import InMemoryKnowledgeStore

    return InMemoryKnowledgeStore(
        max_states_per_system=max_states,
        logger=logger,
        metrics=metrics,
    )


def build_world_model(
    config: "PolarisConfig",
    knowledge_store: "KnowledgeStore",
    logger: "Logger",
    metrics: Optional["MetricsCollector"],
) -> "WorldModel":
    """Create the default statistical world model."""
    from polaris.world_model import StatisticalWorldModel

    wm_cfg: Dict[str, Any] = {}
    if isinstance(config.world_model, dict):
        wm_cfg = config.world_model

    wm_type = str(wm_cfg.get("type", "statistical")).lower()
    if wm_type != "statistical":
        raise ValueError(f"Unknown world model type '{wm_type}'")

    stat_cfg = wm_cfg.get("statistical", {}) if isinstance(wm_cfg.get("statistical"), dict) else {}
    use_kalman = bool(stat_cfg.get("use_kalman", False))
    try:
        window_size = int(stat_cfg.get("window_size", 100))
    except Exception as exc:
        raise ValueError("world_model.statistical.window_size must be an integer") from exc
    if window_size <= 0:
        raise ValueError("world_model.statistical.window_size must be > 0")

    return StatisticalWorldModel(
        knowledge_store,
        use_kalman=use_kalman,
        window_size=window_size,
        logger=logger,
        metrics=metrics,
    )


def build_strategy(
    strategy_config: Any,
    logger: "Logger",
    metrics: Optional["MetricsCollector"],
    knowledge_store: "KnowledgeStore",
    world_model: "WorldModel",
    registry: "ConnectorRegistry",
) -> "AdaptationStrategy":
    """Create a strategy from configuration."""
    from polaris.core.factories import get_strategy_factory

    factory = get_strategy_factory(strategy_config.type)
    if not factory:
        raise ValueError(f"No strategy factory registered for type '{strategy_config.type}'")

    return factory(
        strategy_config,
        logger,
        metrics,
        knowledge_store,
        world_model,
        registry,
    )


def build_meta_learner(
    meta_config: Optional[Dict[str, Any]],
    knowledge_store: "KnowledgeStore",
    world_model: "WorldModel",
    logger: "Logger",
    metrics: Optional["MetricsCollector"],
) -> Optional["MetaLearner"]:
    """Create a meta-learner from configuration."""
    if not isinstance(meta_config, dict):
        return None

    meta_type = meta_config.get("type", "statistical")

    if meta_type == "statistical":
        from polaris.meta_learner.bayesian_optimizer import AcquisitionFunction
        from polaris.meta_learner.statistical import StatisticalMetaLearner

        stat_cfg = meta_config.get("statistical", {}) or {}
        conservative_mode = bool(stat_cfg.get("conservative_mode", True))
        enable_bayesian = bool(stat_cfg.get("enable_bayesian_optimization", True))
        min_samples = int(stat_cfg.get("min_samples_for_optimization", 10))

        acq_func_str = stat_cfg.get("acquisition_function", "expected_improvement")
        try:
            acquisition_function = AcquisitionFunction(acq_func_str)
        except ValueError:
            logger.warning(f"Unknown acquisition function '{acq_func_str}', using default")
            acquisition_function = AcquisitionFunction.EXPECTED_IMPROVEMENT

        exploration_weight = float(stat_cfg.get("exploration_weight", 0.1))

        return StatisticalMetaLearner(
            knowledge_store=knowledge_store,
            logger=logger,
            conservative_mode=conservative_mode,
            world_model=world_model,
            enable_bayesian_optimization=enable_bayesian,
            acquisition_function=acquisition_function,
            exploration_weight=exploration_weight,
            min_samples_for_optimization=min_samples,
        )

    if meta_type == "llm":
        try:
            from polaris.infrastructure.llm import create_llm_client_from_config
            from polaris.meta_learner.llm_based import LLMMetaLearner

            llm_cfg = meta_config.get("llm", {}) or {}
            llm_client = create_llm_client_from_config(llm_cfg)

            return LLMMetaLearner(
                llm_client=llm_client,
                knowledge_store=knowledge_store,
                logger=logger,
                auto_apply=bool(llm_cfg.get("auto_apply", False)),
                temperature=float(llm_cfg.get("temperature", 0.1)),
                analysis_system_prompt=llm_cfg.get("analysis_system_prompt"),
                optimization_system_prompt=llm_cfg.get("optimization_system_prompt"),
                per_system_prompts=llm_cfg.get("per_system_prompts"),
                metrics=metrics,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize LLM meta-learner from config: {e}")
            return None

    logger.warning(f"Unknown meta_learner type '{meta_type}'. Meta-learning will be disabled.")
    return None


def build_connectors(
    systems_config: Any,
    logger: "Logger",
    metrics: Optional["MetricsCollector"],
) -> List["Connector"]:
    """Create connectors for all enabled systems in the configuration."""
    from polaris.core.factories import get_connector_factory

    connectors: List[Any] = []

    for system in systems_config:
        if not system.enabled:
            continue

        factory = get_connector_factory(system.connector_type)
        if not factory:
            logger.error(
                f"No connector factory registered for type '{system.connector_type}', "
                f"skipping system '{system.id}'"
            )
            continue

        try:
            connector = factory(system, logger, metrics)
        except Exception as e:
            logger.error(
                f"Failed to create connector for system '{system.id}' "
                f"(type '{system.connector_type}'): {e}"
            )
            continue

        connectors.append(connector)

    return connectors
