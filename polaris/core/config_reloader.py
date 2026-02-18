"""Hot-reload configuration watcher.

Extracted from ``Polaris._maybe_hot_reload_config`` and
``Polaris._apply_strategy_hot_reload`` so the config-reload logic can be
tested and reused independently of the monitoring loop.
"""

import os
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from polaris.abstractions import AdaptationStrategy, Logger, MetricsCollector
    from polaris.infrastructure.config import PolarisConfig


class ConfigReloader:
    """Watches a config file for changes and applies live updates to the strategy.

    On each call to :meth:`maybe_reload`, the file's modification time is
    compared against the last-seen mtime.  If the file has changed, the config
    is reloaded and strategy parameters are updated in-place (without
    restarting the framework).  A full strategy-type change still requires a
    restart and is logged as such.
    """

    def __init__(
        self,
        config_path: Optional[str],
        strategy: Optional["AdaptationStrategy"],
        logger: "Logger",
        metrics: Optional["MetricsCollector"],
        config: "PolarisConfig",
    ) -> None:
        """Initialize the reloader."""
        self._config_path = config_path
        self._strategy = strategy
        self._logger = logger
        self._metrics = metrics
        self._config = config
        self._config_mtime: Optional[float] = None

        if config_path:
            try:
                self._config_mtime = os.path.getmtime(config_path)
            except Exception:
                self._config_mtime = None

    def update_strategy(self, strategy: Optional["AdaptationStrategy"]) -> None:
        """Update the strategy reference (called when Polaris swaps strategies)."""
        self._strategy = strategy

    async def maybe_reload(self) -> Optional["PolarisConfig"]:
        """Check for config changes and apply strategy/resilience updates.

        Returns:
            The newly loaded ``PolarisConfig`` if the file changed, or
            ``None`` if no reload was performed.
        """
        if not self._config_path:
            return None

        try:
            mtime = os.path.getmtime(self._config_path)
        except Exception:
            return None

        if self._config_mtime is not None and mtime <= self._config_mtime:
            return None

        self._emit("polaris.config.hot_reload.attempts")
        try:
            from polaris.infrastructure.config import load_config

            new_conf = load_config(self._config_path)
            await self._apply_strategy_hot_reload(new_conf.strategy)
            self._config = new_conf
            self._config_mtime = mtime
            self._emit("polaris.config.hot_reload.success")
            self._logger.info("Applied hot-reload from updated configuration")
            return new_conf
        except Exception as e:
            self._emit("polaris.config.hot_reload.errors")
            self._logger.warning(f"Hot-reload skipped due to error: {e}")
            return None

    async def _apply_strategy_hot_reload(self, strategy_config: Any) -> None:
        """Apply parameter updates for the current strategy from new config."""
        if not self._strategy or not strategy_config:
            return

        # Map strategy type strings to class names
        _TYPE_TO_CLASS: Dict[str, str] = {
            "threshold": "ThresholdReactiveStrategy",
            "llm_reasoning": "LLMReasoningStrategy",
            "hybrid": "HybridStrategy",
            "agentic_llm": "AgenticLLMStrategy",
        }

        current_class = type(self._strategy).__name__
        desired_type = strategy_config.type
        expected_class = _TYPE_TO_CLASS.get(desired_type)

        if expected_class and current_class != expected_class:
            self._logger.info("Strategy type changed in config; restart required to apply.")
            return

        # Build a type-specific configuration payload and delegate to the strategy
        config_payload: Dict[str, Any]
        if desired_type == "threshold":
            config_payload = strategy_config.threshold or {}
        elif desired_type == "llm_reasoning":
            config_payload = strategy_config.llm or {}
        elif desired_type == "hybrid":
            config_payload = strategy_config.hybrid or {}
        elif desired_type == "agentic_llm":
            config_payload = strategy_config.agentic or {}
        else:
            config_payload = {}

        try:
            await self._strategy.apply_config_update(config_payload)
        except Exception as e:
            self._logger.warning(f"Failed to apply strategy config update: {e}")

    def _emit(self, metric: str) -> None:
        """Increment a metric if metrics collection is enabled."""
        if not self._metrics:
            return
        from polaris.core.component_builder import ComponentBuilder

        if ComponentBuilder.should_collect(self._config, "core_framework", self._metrics):
            self._metrics.increment(metric)
