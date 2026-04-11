"""Runtime settings helpers used by ComponentBuilder."""

from typing import TYPE_CHECKING, Any, Dict, Optional

from polaris.infrastructure.constants import DEFAULT_MONITORING_INTERVAL

if TYPE_CHECKING:
    from polaris.abstractions import MetricsCollector
    from polaris.infrastructure.config import PolarisConfig


def build_metrics_export_config(
    config: "PolarisConfig",
    cli_overrides: Dict[str, Any],
    metrics: Optional["MetricsCollector"],
) -> Dict[str, Any]:
    """Build metrics auto-export configuration."""
    if not metrics or not hasattr(metrics, "export_to_file"):
        return {"enabled": False}

    export_config: Dict[str, Any] = {}
    if hasattr(config, "observability") and config.observability:
        metrics_config = config.observability.get("metrics", {})
        export_config = metrics_config.get("export", {})

    export_enabled = export_config.get("enabled", False)
    export_dir = cli_overrides.get("metrics_export_dir") or export_config.get(
        "output_dir", "./metrics"
    )
    auto_interval = cli_overrides.get("metrics_auto_export_interval")
    if auto_interval is None:
        auto_interval = export_config.get("auto_export_interval_minutes", 0)

    if export_enabled and auto_interval is not None and auto_interval > 0:
        return {
            "enabled": True,
            "interval_minutes": auto_interval,
            "output_dir": export_dir,
            "formats": cli_overrides.get("metrics_export_formats")
            or export_config.get("formats", ["json"]),
            "experiment_name": cli_overrides.get("metrics_experiment_name")
            or export_config.get("experiment_name"),
            "include_timestamp": export_config.get("include_timestamp", True),
        }

    return {"enabled": False}


def resolve_meta_learning_interval(meta_config: Optional[Dict[str, Any]]) -> float:
    """Resolve meta-learning interval in seconds from config."""
    if not isinstance(meta_config, dict):
        return 3600.0
    try:
        interval_hours = float(meta_config.get("analysis_interval_hours", 1.0))
        if interval_hours > 0:
            return interval_hours * 3600.0
    except Exception:
        pass
    return 3600.0


def resolve_meta_learning_transparency_config(
    meta_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Normalize transparency config for meta-learning records."""
    defaults = {
        "enabled": True,
        "output_path": "./logs/meta_learning_updates.jsonl",
    }
    if not isinstance(meta_config, dict):
        return defaults

    transparency = meta_config.get("transparency")
    if transparency is None:
        return defaults
    if not isinstance(transparency, dict):
        return defaults

    enabled = transparency.get("enabled", defaults["enabled"])
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() in {"1", "true", "yes", "on"}
    elif not isinstance(enabled, bool):
        enabled = defaults["enabled"]

    output_path = transparency.get("output_path", defaults["output_path"])
    if not isinstance(output_path, str) or not output_path.strip():
        output_path = defaults["output_path"]

    return {
        "enabled": bool(enabled),
        "output_path": output_path,
    }


def resolve_monitoring_interval(
    config: "PolarisConfig",
    cli_overrides: Dict[str, Any],
) -> float:
    """Resolve and validate monitoring loop interval in seconds."""
    interval: float = DEFAULT_MONITORING_INTERVAL

    if hasattr(config, "monitoring") and config.monitoring:
        interval = config.monitoring.get("interval_seconds", interval)
    if "monitoring_interval" in cli_overrides:
        interval = cli_overrides["monitoring_interval"]

    try:
        interval = float(interval)
    except Exception as exc:
        raise ValueError("monitoring.interval_seconds must be a number") from exc

    if interval <= 0:
        raise ValueError("monitoring.interval_seconds must be > 0")

    return interval
