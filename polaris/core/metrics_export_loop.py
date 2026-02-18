"""Metrics auto-export background loop.

Extracted from ``Polaris._metrics_export_loop`` so the export logic can be
tested and reused independently of the monitoring loop.
"""

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from polaris.abstractions import Logger, MetricsCollector
    from polaris.infrastructure.config import PolarisConfig


class MetricsExportLoop:
    """Background loop that periodically exports metrics to disk.

    Reads the export configuration dict produced by
    :meth:`ComponentBuilder.build_metrics_export_config` and, if enabled,
    sleeps for the configured interval then calls the metrics exporter.
    """

    def __init__(
        self,
        metrics: Optional["MetricsCollector"],
        export_config: Dict[str, Any],
        logger: "Logger",
        config: "PolarisConfig",
    ) -> None:
        """Initialize the metrics export loop."""
        self._metrics = metrics
        self._export_config = export_config
        self._logger = logger
        self._config = config
        self._running = False

    async def run(self) -> None:
        """Run the metrics export loop until cancelled.

        Returns immediately if metrics export is not configured or the metrics
        collector does not support file export.
        """
        if not self._metrics or not hasattr(self._metrics, "export_to_file"):
            return
        if not self._export_config.get("enabled", False):
            return

        interval_seconds = self._export_config["interval_minutes"] * 60
        self._running = True
        self._logger.info(
            f"Starting metrics auto-export every {self._export_config['interval_minutes']} minutes"
        )
        self._emit("polaris.metrics.auto_export_started")

        while self._running:
            try:
                await asyncio.sleep(interval_seconds)

                if not self._running:
                    break  # type: ignore[unreachable]

                await self._do_export()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in metrics export loop: {e}")
                self._emit("polaris.metrics.export_loop_errors")

        self._logger.info("Metrics auto-export loop stopped")

    async def _do_export(self) -> None:
        """Perform a single metrics export."""
        from polaris.infrastructure.observability.export import export_polaris_metrics

        try:
            export_start = datetime.now(timezone.utc)
            exported_files = export_polaris_metrics(
                metrics_collector=self._metrics,  # type: ignore[arg-type]
                output_dir=self._export_config["output_dir"],
                experiment_name=self._export_config.get("experiment_name"),
                formats=self._export_config["formats"],
            )
            self._logger.info(f"Auto-exported metrics to {len(exported_files)} files")
            self._emit("polaris.metrics.auto_exports_completed")
            export_duration = (datetime.now(timezone.utc) - export_start).total_seconds()
            if self._metrics and self._should_collect():
                self._metrics.histogram(
                    "polaris.metrics.auto_export_duration_seconds", export_duration
                )
        except Exception as e:
            self._logger.error(f"Failed to auto-export metrics: {e}")
            self._emit("polaris.metrics.auto_export_errors")

    def _should_collect(self) -> bool:
        from polaris.core.component_builder import ComponentBuilder

        return ComponentBuilder.should_collect(self._config, "core_framework", self._metrics)

    def _emit(self, metric: str) -> None:
        if self._metrics and self._should_collect():
            self._metrics.increment(metric)
