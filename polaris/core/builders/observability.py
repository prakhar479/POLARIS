"""Focused observability builders used by ComponentBuilder."""

from typing import TYPE_CHECKING, Any, Dict, Optional, cast

if TYPE_CHECKING:
    from polaris.abstractions import Logger, MetricsCollector
    from polaris.core.events import EventBus
    from polaris.infrastructure.config import PolarisConfig


def build_logger(
    config: "PolarisConfig",
    cli_overrides: Dict[str, Any],
) -> "Logger":
    """Create logger from config + CLI overrides."""
    from polaris.infrastructure.observability.logger import create_logger

    logger_type = "structured"
    level = "INFO"
    console = True
    log_file = None
    use_colors = True

    if hasattr(config, "observability") and config.observability:
        logging_config = config.observability.get("logging", {})
        logger_type = logging_config.get("type", logger_type)
        level = logging_config.get("level", level)
        console = logging_config.get("console", console)
        use_colors = logging_config.get("use_colors", use_colors)
        if logging_config.get("file", False):
            log_file = logging_config.get("file_path", "./logs/polaris.log")

    if "log_format" in cli_overrides:
        logger_type = cli_overrides["log_format"]
    if "log_level" in cli_overrides:
        level = cli_overrides["log_level"]
    if "console_logging" in cli_overrides:
        console = bool(cli_overrides["console_logging"])
    if "log_file" in cli_overrides:
        log_file = cli_overrides["log_file"]

    return create_logger(
        logger_type=logger_type,
        name="polaris",
        level=level,
        log_file=log_file,
        console=console,
        use_colors=use_colors,
    )


def build_metrics(
    config: "PolarisConfig",
    cli_overrides: Dict[str, Any],
) -> "MetricsCollector":
    """Create metrics collector from config + CLI overrides."""
    from polaris.infrastructure.observability.null_metrics import NullMetricsCollector

    if cli_overrides.get("metrics_enabled", True) is False:
        return NullMetricsCollector()

    metrics_config: Dict[str, Any] = {}
    if hasattr(config, "observability") and config.observability:
        metrics_config = config.observability.get("metrics", {})

    if not metrics_config.get("enabled", True):
        return NullMetricsCollector()

    collector_type = metrics_config.get("collector_type", "simple")

    if collector_type == "simple":
        from polaris.infrastructure.observability.metrics import SimpleMetricsCollector

        return SimpleMetricsCollector()

    raise ValueError(f"Unknown metrics collector type '{collector_type}'")


def build_event_bus(
    config: "PolarisConfig",
    metrics: Optional["MetricsCollector"],
    logger: "Logger",
) -> "EventBus":
    """Create event bus with optional metrics wiring."""
    from polaris.core.events import InMemoryEventBus

    RedisEventBus: Optional[Any] = None
    try:
        from polaris.infrastructure.events.redis_bus import RedisEventBus as RedisEventBusCls

        RedisEventBus = RedisEventBusCls
    except ImportError:
        pass

    bus_type = "memory"
    obs: Dict[str, Any] = {}
    if hasattr(config, "observability") and config.observability:
        obs = config.observability
        if "event_bus" in obs and isinstance(obs["event_bus"], dict):
            bus_type = obs["event_bus"].get("type", "memory")

    if bus_type == "redis" and RedisEventBus is not None:
        redis_url = obs.get("event_bus", {}).get("url", "redis://localhost:6379")
        return cast(
            "EventBus",
            RedisEventBus(redis_url=redis_url, metrics=metrics, logger=logger),
        )

    if bus_type not in ("memory", "redis"):
        raise ValueError(f"Unknown event bus type '{bus_type}'")
    if bus_type == "redis" and RedisEventBus is None:
        raise ImportError("RedisEventBus requested but redis dependency is not available")

    return InMemoryEventBus(metrics=metrics, logger=logger)
