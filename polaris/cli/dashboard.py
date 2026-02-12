"""Interactive dashboard for Polaris.

Real-time terminal UI showing system state, summarized metrics, and recent
events/logs in a minimal, developer-friendly layout.
"""

import asyncio
import logging
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class Dashboard:
    """
    Interactive TUI dashboard for Polaris.

    Displays real-time system state, metrics, and adaptation events.
    Requires 'rich' library: pip install rich
    """

    def __init__(self, polaris: Any) -> None:
        """Initialize dashboard with Polaris instance."""
        if not RICH_AVAILABLE:
            raise ImportError(
                "Dashboard requires 'rich' library. " "Install with: pip install rich"
            )

        self.polaris = polaris
        self.console = Console()
        self.running = False

        # Event tracking
        self.recent_events: List[Dict[str, Any]] = []
        self.max_events = 10
        self.metric_history: Dict[str, List[Any]] = defaultdict(list)
        self.max_history = 50
        self._cached_perf_metrics: Dict[str, Any] = {}

        # Lightweight in-memory log buffer for summarized dashboard logs
        self._log_records: Deque[Dict[str, str]] = deque(maxlen=50)
        self._log_handler: Optional[logging.Handler] = None

        # Subscribe to events and capture logs for dashboard view
        self._subscribe_to_events()
        self._setup_log_capture()

    def _subscribe_to_events(self) -> None:
        """Subscribe to framework events."""
        from polaris.core.events import AdaptationEvent, TelemetryEvent

        self.polaris.event_bus.subscribe(TelemetryEvent, self._on_telemetry)

        self.polaris.event_bus.subscribe(AdaptationEvent, self._on_adaptation)

    def _setup_log_capture(self) -> None:
        """Attach a logging handler that feeds recent logs into the dashboard.

        This provides a concise, human-friendly log view without dumping full
        raw logs into the terminal. Full raw logs still go to configured log
        files as usual.
        """
        logger = logging.getLogger("polaris")

        class _DashboardLogHandler(logging.Handler):
            def __init__(self, buffer: Deque[Dict[str, str]]):
                super().__init__(level=logging.NOTSET)
                self._buffer = buffer

            def emit(self, record: logging.LogRecord) -> None:
                try:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    level = record.levelname
                    component = record.name.split(".")[-1]
                    message = record.getMessage()
                    self._buffer.append(
                        {
                            "time": timestamp,
                            "level": level,
                            "component": component,
                            "message": message,
                        }
                    )
                except Exception:
                    # Logging to dashboard should never break the app
                    pass

        handler = _DashboardLogHandler(self._log_records)
        logger.addHandler(handler)
        self._log_handler = handler

    def _on_telemetry(self, event: Any) -> None:
        """Handle telemetry events."""
        # Track metrics
        for metric_name, metric in event.state.metrics.items():
            self.metric_history[metric_name].append((event.timestamp, metric.value))
            # Keep only recent history
            if len(self.metric_history[metric_name]) > self.max_history:
                self.metric_history[metric_name] = self.metric_history[metric_name][
                    -self.max_history :
                ]

    def _on_adaptation(self, event: Any) -> None:
        """Handle adaptation events."""
        self.recent_events.append(
            {
                "time": event.timestamp,
                "type": "adaptation",
                "action": event.action.action_type,
                "status": event.result.status.value,
                "system": event.action.target_system,
            }
        )

        # Keep only recent events
        if len(self.recent_events) > self.max_events:
            self.recent_events = self.recent_events[-self.max_events :]

    def _build_layout(self) -> Layout:
        """Build dashboard layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3), Layout(name="body"), Layout(name="footer", size=3)
        )

        layout["body"].split_row(Layout(name="left"), Layout(name="right"))

        layout["left"].split_column(
            Layout(name="systems", ratio=1), Layout(name="metrics", ratio=2)
        )

        layout["right"].split_column(
            Layout(name="events", ratio=1),
            Layout(name="logs", ratio=1),
            Layout(name="strategy", ratio=1),
            Layout(name="system_metrics", ratio=1),
        )

        return layout

    def _render(self) -> Layout:
        """Render current dashboard state."""
        layout = self._build_layout()

        # Header
        header_text = Text()
        header_text.append("POLARIS", style="bold cyan")
        header_text.append(" - Self-Adaptive Systems Framework", style="dim")
        layout["header"].update(Panel(header_text, border_style="cyan"))

        # Systems panel
        systems_table = Table(title="Connected Systems", show_header=True)
        systems_table.add_column("System ID", style="cyan")
        systems_table.add_column("Status", style="green")

        for system_id in self.polaris.registry.system_ids():
            systems_table.add_row(system_id, "✓ Connected")

        layout["systems"].update(Panel(systems_table, border_style="green"))

        # Metrics panel
        metrics_table = Table(title="Current Metrics", show_header=True)
        metrics_table.add_column("Metric", style="yellow")
        metrics_table.add_column("Value", style="white")
        metrics_table.add_column("Trend", style="dim")

        for metric_name, history in self.metric_history.items():
            if history:
                current = history[-1][1]
                trend = self._calculate_trend(history)
                metrics_table.add_row(metric_name, str(current), trend)

        layout["metrics"].update(Panel(metrics_table, border_style="yellow"))

        # Events panel
        events_table = Table(title="Recent Events", show_header=True)
        events_table.add_column("Time", style="dim")
        events_table.add_column("Event", style="white")
        events_table.add_column("Status", style="green")

        for event in self.recent_events[-10:]:
            time_str = event["time"].strftime("%H:%M:%S")
            event_str = f"{event['action']} on {event['system']}"
            status = "✓" if event["status"] == "success" else "✗"
            events_table.add_row(time_str, event_str, status)

        layout["events"].update(Panel(events_table, border_style="magenta"))

        # Latest Logs panel (summarized, not full raw log stream)
        logs_table = Table(title="Latest Logs", show_header=True, show_lines=False)
        logs_table.add_column("Time", style="dim", no_wrap=True)
        logs_table.add_column("Lvl", style="cyan", no_wrap=True)
        logs_table.add_column("Comp", style="magenta", no_wrap=True)
        logs_table.add_column("Message", style="white", overflow="fold")

        for rec in list(self._log_records)[-20:]:
            logs_table.add_row(
                rec.get("time", ""),
                rec.get("level", ""),
                rec.get("component", ""),
                rec.get("message", ""),
            )

        layout["logs"].update(Panel(logs_table, border_style="white"))

        # Strategy panel
        strategy_info = Table(title="Strategy Info", show_header=False)
        strategy_info.add_column("Key", style="cyan")
        strategy_info.add_column("Value", style="white")

        if self.polaris.strategy:
            strategy_name = self.polaris.strategy.__class__.__name__
            strategy_info.add_row("Type", strategy_name)

            # Use cached performance metrics
            if hasattr(self, "_cached_perf_metrics") and self._cached_perf_metrics:
                perf = self._cached_perf_metrics
                if "success_rate" in perf:
                    strategy_info.add_row("Success Rate", f"{perf['success_rate']: .1%}")
                if "total_adaptations" in perf:
                    strategy_info.add_row("Total Adaptations", str(int(perf["total_adaptations"])))

        layout["strategy"].update(Panel(strategy_info, border_style="blue"))

        # System Metrics panel
        system_metrics_table = Table(title="System Metrics", show_header=True)
        system_metrics_table.add_column("Component", style="cyan")
        system_metrics_table.add_column("Metric", style="yellow")
        system_metrics_table.add_column("Value", style="white")

        # Get system metrics from Polaris metrics collector
        if self.polaris.metrics:
            try:
                metrics_summary = self.polaris.metrics.get_summary()

                # Display key system metrics
                counters = metrics_summary.get("counters", {})
                gauges = metrics_summary.get("gauges", {})
                histograms = metrics_summary.get("histograms", {})

                # Show monitoring metrics
                for metric_name, value in counters.items():
                    if "polaris.monitoring" in metric_name:
                        component = "Monitoring"
                        clean_name = metric_name.replace("polaris.monitoring.", "")
                        system_metrics_table.add_row(component, clean_name, str(int(value)))

                # Show telemetry metrics
                for metric_name, value in counters.items():
                    if "polaris.telemetry" in metric_name:
                        component = "Telemetry"
                        clean_name = metric_name.replace("polaris.telemetry.", "")
                        system_metrics_table.add_row(component, clean_name, str(int(value)))

                # Show adaptation metrics
                for metric_name, value in counters.items():
                    if "polaris.adaptations" in metric_name:
                        component = "Adaptations"
                        clean_name = metric_name.replace("polaris.adaptations.", "")
                        system_metrics_table.add_row(component, clean_name, str(int(value)))

                # Show knowledge store metrics
                for metric_name, value in counters.items():
                    if "polaris.knowledge" in metric_name:
                        component = "Knowledge"
                        clean_name = metric_name.replace("polaris.knowledge.", "")
                        system_metrics_table.add_row(component, clean_name, str(int(value)))

                # Show world model metrics
                for metric_name, value in counters.items():
                    if "polaris.world_model" in metric_name:
                        component = "World Model"
                        clean_name = metric_name.replace("polaris.world_model.", "")
                        system_metrics_table.add_row(component, clean_name, str(int(value)))

                # Show gauge metrics (current values)
                for metric_name, value in gauges.items():
                    if "polaris.monitoring" in metric_name:
                        component = "Monitoring"
                        clean_name = metric_name.replace("polaris.monitoring.", "")
                        system_metrics_table.add_row(component, clean_name, str(int(value)))

                # Show histogram averages for performance metrics
                for metric_name, hist_data in histograms.items():
                    if "polaris.monitoring.loop_duration" in metric_name:
                        component = "Performance"
                        avg_duration = hist_data.get("avg", 0)
                        system_metrics_table.add_row(
                            component, "Avg Loop Duration", f"{avg_duration: .2f}s"
                        )

            except Exception as e:
                system_metrics_table.add_row("Error", "Metrics", f"Failed to load: {str(e)}")

        layout["system_metrics"].update(Panel(system_metrics_table, border_style="red"))

        # Footer
        footer_text = Text()
        footer_text.append("Status: ", style="dim")
        status = "Running" if self.polaris.is_running() else "Stopped"
        footer_text.append(status, style="green bold" if self.polaris.is_running() else "red bold")
        footer_text.append(f" | Time: {datetime.now().strftime('%H:%M:%S')}", style="dim")
        footer_text.append(f" | Metrics tracked: {len(self.metric_history)}", style="dim")

        # Add system component status
        if self.polaris.metrics:
            try:
                summary = self.polaris.metrics.get_summary()
                total_metrics = len(summary.get("counters", {})) + len(summary.get("gauges", {}))
                footer_text.append(f" | System metrics: {total_metrics}", style="dim")
            except Exception:
                pass

        footer_text.append(" | Press Ctrl+C to exit", style="dim")

        layout["footer"].update(Panel(footer_text, border_style="dim"))

        return layout

    def _calculate_trend(self, history: list) -> str:
        """Calculate simple trend indicator."""
        if len(history) < 2:
            return "—"

        recent = [v for _, v in history[-10:]]
        try:
            recent_floats = [float(v) for v in recent]
            if len(recent_floats) < 2:
                return "—"

            avg_first = sum(recent_floats[: len(recent_floats) // 2]) / (len(recent_floats) // 2)
            avg_last = sum(recent_floats[len(recent_floats) // 2 :]) / (
                len(recent_floats) - len(recent_floats) // 2
            )

            if avg_last > avg_first * 1.05:
                return "↑"
            elif avg_last < avg_first * 0.95:
                return "↓"
            else:
                return "→"
        except (ValueError, TypeError, ZeroDivisionError):
            return "—"

    async def _update_metrics_cache(self) -> None:
        """Update cached performance metrics in background."""
        while self.running:
            try:
                if self.polaris.strategy:
                    self._cached_perf_metrics = (
                        await self.polaris.strategy.get_performance_metrics()
                    )
            except Exception as e:
                # Log error but continue running (to framework logger, not stdout)
                try:
                    logger = logging.getLogger("polaris.dashboard")
                    logger.error("Error updating dashboard metrics cache: %s", str(e))
                except Exception:
                    pass
            await asyncio.sleep(5)  # Update every 5 seconds

    async def run(self, refresh_rate: float = 1.0) -> None:
        """
        Run the dashboard.

        Args:
            refresh_rate: Update frequency in seconds
        """
        self.running = True

        # Start background metrics update task
        metrics_task = asyncio.create_task(self._update_metrics_cache())

        with Live(
            self._render(), console=self.console, refresh_per_second=1 / refresh_rate
        ) as live:
            try:
                while self.running and self.polaris.is_running():
                    await asyncio.sleep(refresh_rate)
                    live.update(self._render())
            except KeyboardInterrupt:
                pass
            finally:
                self.running = False
                metrics_task.cancel()
                try:
                    await metrics_task
                except asyncio.CancelledError:
                    pass

                # Detach dashboard log handler on exit to avoid leaks
                if self._log_handler is not None:
                    try:
                        root_logger = logging.getLogger("polaris")
                        root_logger.removeHandler(self._log_handler)
                    except Exception:
                        pass
