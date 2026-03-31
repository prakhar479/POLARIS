"""Interactive dashboard for Polaris.

Real-time terminal UI showing system state, summarized metrics, and recent events/logs
in a minimal, developer-friendly layout.
"""

import asyncio
import logging
import os
import sys
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime
from io import StringIO
from typing import Any, Deque, Dict, List, Optional, cast

from polaris.infrastructure.constants import DEFAULT_JSON_INDENT

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


class _EmbeddedInteractiveCLI:
    """Interactive command runner for split dashboard mode."""

    def __init__(self, polaris: Any, output: Deque[str]) -> None:
        from polaris.cli.interactive import PolarisInteractiveCLI

        self._output = output
        self._cli = PolarisInteractiveCLI(polaris)

        # Route all interactive output into the dashboard output pane.
        self._cli._print = self._print  # type: ignore[method-assign]
        self._cli._print_table = self._print_table  # type: ignore[method-assign]
        self._cli._print_json = self._print_json  # type: ignore[method-assign]
        self._cli.do_clear = self.do_clear  # type: ignore[method-assign]

    def _append(self, value: Any) -> None:
        text = str(value).strip("\n")
        for line in text.splitlines():
            self._output.append(line)

    def _print(self, content: Any, style: Optional[str] = None) -> None:
        _ = style
        self._append(content)

    def _print_table(self, table: Any) -> None:
        if RICH_AVAILABLE:
            buffer = StringIO()
            console = Console(file=buffer, force_terminal=False, color_system=None, width=120)
            console.print(table)
            self._append(buffer.getvalue())
        else:
            self._append("Table output requires 'rich' library")

    def _print_json(self, data: Any) -> None:
        import json

        self._append(json.dumps(data, indent=DEFAULT_JSON_INDENT, default=str))

    def do_clear(self, arg: str) -> None:
        _ = arg
        self._output.clear()
        self._output.append("Cleared interactive output.")

    def execute(self, command: str) -> bool:
        """Execute one CLI command. Returns True when command requests exit."""
        return bool(self._cli.onecmd(command))

    def command_names(self) -> List[str]:
        """Return available command names for interactive completion hints."""
        names = []
        for attr in dir(self._cli):
            if attr.startswith("do_"):
                names.append(attr[3:])
        return sorted(names)


class Dashboard:
    """Interactive TUI dashboard for Polaris.

    Displays real-time system state, metrics, and adaptation events. Requires 'rich'
    library: pip install rich
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
        self._started_at = datetime.now()

        # Event tracking
        self.recent_events: Deque[Dict[str, Any]] = deque(maxlen=10)
        self.max_events = 10  # kept for public interface compatibility
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

        This provides a concise, human-friendly log view without dumping full raw logs
        into the terminal. Full raw logs still go to configured log files as usual.
        """
        logger = logging.getLogger("polaris")

        class _DashboardLogHandler(logging.Handler):
            def __init__(self, buffer: Deque[Dict[str, str]]):
                # Use WARNING as the default floor so the handler only
                # captures meaningful operational messages by default.  Callers
                # that need DEBUG-level dashboard output can lower this via the
                # standard logging.setLevel() API on the "polaris" logger.
                super().__init__(level=logging.WARNING)
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
        for metric_name, metric in event.state.metrics.items():
            self.metric_history[metric_name].append((event.timestamp, metric.value))
            # Keep only recent history
            if len(self.metric_history[metric_name]) > self.max_history:
                self.metric_history[metric_name] = self.metric_history[metric_name][
                    -self.max_history :
                ]

    def _on_adaptation(self, event: Any) -> None:
        """Handle adaptation events."""
        # Update deque maxlen if max_events changed
        if self.recent_events.maxlen != self.max_events:
            self.recent_events = deque(self.recent_events, maxlen=self.max_events)

        self.recent_events.append(
            {
                "time": event.timestamp,
                "type": "adaptation",
                "action": event.action.action_type,
                "status": event.result.status.value,
                "system": event.action.target_system,
            }
        )

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
        """Render current dashboard state.

        Snapshot shared state once. Both recent_events and metric_history may be written
        by event callbacks that fire from a different coroutine or thread. Taking local
        copies here means the rest of _render works on stable data.
        """
        layout = self._build_layout()

        recent_events_snapshot: List[Dict[str, Any]] = list(self.recent_events)
        metric_history_snapshot: Dict[str, List[Any]] = {
            k: list(v) for k, v in self.metric_history.items()
        }

        # Header
        header_text = Text()
        header_text.append("POLARIS", style="bold cyan")
        header_text.append(" - Self-Adaptive Systems Framework", style="dim")
        header_text.append("  |  ", style="dim")
        status = "RUNNING" if self.polaris.is_running() else "STOPPED"
        status_style = "green bold" if self.polaris.is_running() else "red bold"
        header_text.append(status, style=status_style)
        header_text.append("  |  ", style="dim")
        header_text.append(f"Systems: {len(self.polaris.registry.system_ids())}", style="white")
        header_text.append("  |  ", style="dim")
        header_text.append(f"Events: {len(recent_events_snapshot)}", style="white")
        layout["header"].update(Panel(header_text, border_style="cyan"))

        # Systems panel
        systems_table = Table(title="Connected Systems", show_header=True)
        systems_table.add_column("System ID", style="cyan")
        systems_table.add_column("Status", style="green")
        systems = list(self.polaris.registry.system_ids())
        if not systems:
            systems_table.add_row("No systems connected", "—")
        for system_id in systems:
            systems_table.add_row(system_id, "✓ Connected")

        layout["systems"].update(Panel(systems_table, border_style="green"))

        # Metrics panel
        metrics_table = Table(title="Current Metrics", show_header=True)
        metrics_table.add_column("Metric", style="yellow")
        metrics_table.add_column("Value", style="white")
        metrics_table.add_column("Trend", style="dim")
        metrics_table.add_column("History", style="magenta")

        metric_items = sorted(metric_history_snapshot.items(), key=lambda item: item[0])[:30]
        if not metric_items:
            metrics_table.add_row("No telemetry yet", "—", "—", "")

        for metric_name, history in metric_items:
            if history:
                current = history[-1][1]
                trend = self._calculate_trend(history)
                spark = self._render_sparkline(history)
                metrics_table.add_row(metric_name, self._format_metric_value(current), trend, spark)

        layout["metrics"].update(Panel(metrics_table, border_style="yellow"))

        # Events panel
        events_table = Table(title="Recent Events", show_header=True)
        events_table.add_column("Time", style="dim")
        events_table.add_column("Event", style="white")
        events_table.add_column("Status", style="green")

        if not recent_events_snapshot:
            events_table.add_row("—", "No events yet", "—")

        for event in recent_events_snapshot[-10:]:
            time_str = event["time"].strftime("%H:%M:%S")
            event_str = f"{event['action']} on {event['system']}"
            status_ok = event["status"] == "success"
            status = "[green]✓[/green]" if status_ok else "[red]✗[/red]"
            events_table.add_row(time_str, event_str, status)

        layout["events"].update(Panel(events_table, border_style="magenta"))

        # Latest Logs panel (summarized, not full raw log stream)
        logs_table = Table(title="Latest Logs", show_header=True, show_lines=False)
        logs_table.add_column("Time", style="dim", no_wrap=True)
        logs_table.add_column("Lvl", style="cyan", no_wrap=True)
        logs_table.add_column("Comp", style="magenta", no_wrap=True)
        logs_table.add_column("Message", style="white", overflow="fold")

        recent_logs = list(self._log_records)[-20:]
        if not recent_logs:
            logs_table.add_row("—", "—", "—", "No logs yet")
        else:
            for rec in recent_logs:
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
            strategy_info.add_row(
                "Meta Learner",
                (
                    self.polaris.meta_learner.__class__.__name__
                    if self.polaris.meta_learner
                    else "Disabled"
                ),
            )

            if hasattr(self, "_cached_perf_metrics") and self._cached_perf_metrics:
                perf = self._cached_perf_metrics
                if "success_rate" in perf:
                    strategy_info.add_row("Success Rate", f"{perf['success_rate']:.1%}")
                if "total_adaptations" in perf:
                    strategy_info.add_row("Total Adaptations", str(int(perf["total_adaptations"])))

        layout["strategy"].update(Panel(strategy_info, border_style="blue"))

        # System Metrics panel
        system_metrics_table = Table(title="System Metrics", show_header=True)
        system_metrics_table.add_column("Component", style="cyan")
        system_metrics_table.add_column("Metric", style="yellow")
        system_metrics_table.add_column("Value", style="white")

        if self.polaris.metrics:
            try:
                metrics_summary = self.polaris.metrics.get_summary()

                counters = metrics_summary.get("counters", {})
                gauges = metrics_summary.get("gauges", {})
                histograms = metrics_summary.get("histograms", {})

                max_rows = 22
                rows_added = 0

                def add_row(component: str, name: str, value: str) -> None:
                    nonlocal rows_added
                    if rows_added >= max_rows:
                        return
                    system_metrics_table.add_row(component, name, value)
                    rows_added += 1

                for metric_name, value in counters.items():
                    if "polaris.monitoring" in metric_name:
                        add_row(
                            "Monitoring",
                            metric_name.replace("polaris.monitoring.", ""),
                            str(int(value)),
                        )

                for metric_name, value in counters.items():
                    if "polaris.telemetry" in metric_name:
                        add_row(
                            "Telemetry",
                            metric_name.replace("polaris.telemetry.", ""),
                            str(int(value)),
                        )

                for metric_name, value in counters.items():
                    if "polaris.adaptations" in metric_name:
                        add_row(
                            "Adaptations",
                            metric_name.replace("polaris.adaptations.", ""),
                            str(int(value)),
                        )

                for metric_name, value in counters.items():
                    if "polaris.knowledge" in metric_name:
                        add_row(
                            "Knowledge",
                            metric_name.replace("polaris.knowledge.", ""),
                            str(int(value)),
                        )

                for metric_name, value in counters.items():
                    if "polaris.world_model" in metric_name:
                        add_row(
                            "World Model",
                            metric_name.replace("polaris.world_model.", ""),
                            str(int(value)),
                        )

                for metric_name, value in gauges.items():
                    if "polaris.monitoring" in metric_name:
                        add_row(
                            "Monitoring",
                            metric_name.replace("polaris.monitoring.", ""),
                            str(int(value)),
                        )

                for metric_name, hist_data in histograms.items():
                    if "polaris.monitoring.loop_duration" in metric_name:
                        avg_duration = hist_data.get("avg", 0)
                        add_row("Performance", "Avg Loop Duration", f"{avg_duration:.2f}s")

                if rows_added == 0:
                    system_metrics_table.add_row("—", "No system metrics yet", "—")

            except Exception as e:
                system_metrics_table.add_row("Error", "Metrics", f"Failed to load: {str(e)}")
        else:
            system_metrics_table.add_row("—", "Metrics collector disabled", "—")

        layout["system_metrics"].update(Panel(system_metrics_table, border_style="red"))

        # Footer
        footer_text = Text()
        footer_text.append("Status: ", style="dim")
        footer_status = "Running" if self.polaris.is_running() else "Stopped"
        footer_text.append(
            footer_status, style="green bold" if self.polaris.is_running() else "red bold"
        )
        footer_text.append(f" | Uptime: {self._format_uptime()}", style="dim")
        footer_text.append(f" | Time: {datetime.now().strftime('%H:%M:%S')}", style="dim")
        footer_text.append(f" | Metrics tracked: {len(metric_history_snapshot)}", style="dim")

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

    def _format_metric_value(self, value: Any) -> str:
        try:
            numeric = float(value)
        except Exception:
            return str(value)
        if abs(numeric) >= 1000:
            return f"{numeric:,.0f}"
        if abs(numeric) >= 10:
            return f"{numeric:.2f}"
        return f"{numeric:.3f}"

    def _format_uptime(self) -> str:
        elapsed = datetime.now() - self._started_at
        seconds = int(elapsed.total_seconds())
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        if minutes > 0:
            return f"{minutes}m {secs}s"
        return f"{secs}s"

    def _calculate_trend(self, history: list) -> str:
        """Calculate simple trend indicator.

        Returns "↑", "↓", "→", or "—" if not enough data or non-numeric.
        """
        if len(history) < 2:
            return "—"

        recent = [v for _, v in history[-10:]]
        try:
            recent_floats = [float(v) for v in recent]
            if len(recent_floats) < 2:
                # Not enough numeric data points for a meaningful trend
                return "—"

            half = len(recent_floats) // 2
            if half == 0:
                return "—"

            avg_first = sum(recent_floats[:half]) / half
            avg_last = sum(recent_floats[half:]) / (len(recent_floats) - half)

            if avg_last > avg_first * 1.05:
                return "↑"
            elif avg_last < avg_first * 0.95:
                return "↓"
            else:
                return "→"
        except (ValueError, TypeError, ZeroDivisionError):
            return "—"

    def _render_sparkline(self, history: list) -> str:
        """Render a small unicode sparkline of recent metric history."""
        if len(history) < 2:
            return ""

        recent = [v for _, v in history[-15:]]
        try:
            recent_floats = [float(v) for v in recent]
            if len(recent_floats) < 2:
                return ""

            min_val = min(recent_floats)
            max_val = max(recent_floats)
            range_val = max_val - min_val

            bars = " ▂▃▄▅▆▇█"
            if range_val == 0:
                # If values are constant, return a middle bar
                return "▄" * len(recent_floats)

            sparkline = ""
            for val in recent_floats:
                idx = int(((val - min_val) / range_val) * (len(bars) - 1))
                sparkline += bars[idx]

            return sparkline
        except (ValueError, TypeError):
            return ""

    @contextmanager
    def _raw_terminal_input(self) -> Any:
        """Enable non-blocking, character-at-a-time stdin reads on POSIX."""
        if os.name == "nt" or not sys.stdin.isatty():
            yield
            return

        import fcntl
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_attrs = termios.tcgetattr(fd)
        old_flags = fcntl.fcntl(fd, fcntl.F_GETFL)

        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
        old_stdout_flags = fcntl.fcntl(stdout_fd, fcntl.F_GETFL)
        old_stderr_flags = fcntl.fcntl(stderr_fd, fcntl.F_GETFL)

        # Track which descriptors we actually modified so the finally block
        # can restore only those, regardless of where an exception may occur.
        stdin_nonblocking_set = False
        stdout_blocking_forced = False
        stderr_blocking_forced = False

        try:
            tty.setcbreak(fd)

            # Set stdin non-blocking for key polling
            fcntl.fcntl(fd, fcntl.F_SETFL, old_flags | os.O_NONBLOCK)
            stdin_nonblocking_set = True

            # Ensure stdout/stderr stay blocking so Rich can write freely
            if old_stdout_flags & os.O_NONBLOCK:
                fcntl.fcntl(stdout_fd, fcntl.F_SETFL, old_stdout_flags & ~os.O_NONBLOCK)
                stdout_blocking_forced = True

            if old_stderr_flags & os.O_NONBLOCK:
                fcntl.fcntl(stderr_fd, fcntl.F_SETFL, old_stderr_flags & ~os.O_NONBLOCK)
                stderr_blocking_forced = True

            yield
        finally:
            # Always restore terminal attributes for stdin
            termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)

            # Restore stdin flags only if we actually changed them
            if stdin_nonblocking_set:
                fcntl.fcntl(fd, fcntl.F_SETFL, old_flags)

            # Restore stdout/stderr only if we forced them to blocking
            if stdout_blocking_forced:
                fcntl.fcntl(stdout_fd, fcntl.F_SETFL, old_stdout_flags)
            if stderr_blocking_forced:
                fcntl.fcntl(stderr_fd, fcntl.F_SETFL, old_stderr_flags)

    @contextmanager
    def _live_display_safe(self, renderable: Any, refresh_per_second: int) -> Any:
        """Safely create and manage a Rich Live display context."""
        live = None
        try:
            try:
                # Keep rendering on the event-loop thread to avoid _RefreshThread
                # write races in non-blocking PTY environments.
                live = Live(
                    renderable,
                    console=self.console,
                    auto_refresh=False,
                    refresh_per_second=refresh_per_second,
                )
                live.start(refresh=True)
            except BlockingIOError:
                # Terminal FD may be in an inconsistent non-blocking state.
                # Attempt to coerce stdout/stderr back to blocking before retry.
                try:
                    import fcntl as _fcntl

                    for _fd in (sys.stdout.fileno(), sys.stderr.fileno()):
                        _flags = _fcntl.fcntl(_fd, _fcntl.F_GETFL)
                        if _flags & os.O_NONBLOCK:
                            _fcntl.fcntl(_fd, _fcntl.F_SETFL, _flags & ~os.O_NONBLOCK)
                except Exception:
                    pass

                # Retry once after fixing FD state
                live = Live(
                    renderable,
                    console=self.console,
                    auto_refresh=False,
                    refresh_per_second=refresh_per_second,
                )
                live.start(refresh=True)
            yield live

        finally:
            if live is not None:
                try:
                    live.stop()
                except BlockingIOError:
                    try:
                        sys.stdout.flush()
                        sys.stderr.flush()
                    except Exception:
                        pass
                except Exception:
                    pass

    def _safe_live_update(self, live: Any, renderable: Any, max_retries: int = 3) -> None:
        """Safely update Live display, handling BlockingIOError gracefully."""
        for attempt in range(max_retries):
            try:
                try:
                    live.update(renderable, refresh=True)
                except TypeError as exc:
                    # Some test doubles / Live-like wrappers may not accept refresh.
                    if "refresh" not in str(exc):
                        raise
                    live.update(renderable)
                return
            except BlockingIOError:
                if attempt < max_retries - 1:
                    import time

                    time.sleep(0.001)
                else:
                    try:
                        logging.getLogger("polaris.dashboard").debug(
                            "BlockingIOError in Live update after retries, skipping frame"
                        )
                    except Exception:
                        pass
            except Exception as e:
                try:
                    logging.getLogger("polaris.dashboard").debug(
                        "Error updating Live display: %s", str(e)
                    )
                except Exception:
                    pass
                return

    def _read_key_nonblocking(self) -> Optional[str]:
        """Read one key from stdin without blocking.

        Returns None if no key is available or on error.
        """
        if os.name == "nt":
            try:
                import msvcrt

                if not msvcrt.kbhit():  # type: ignore[attr-defined]
                    return None
                ch = cast(str, msvcrt.getwch())  # type: ignore[attr-defined]
                if ch in ("\x00", "\xe0"):
                    code = cast(str, msvcrt.getwch())  # type: ignore[attr-defined]
                    windows_arrow_map: Dict[str, str] = {
                        "H": "<UP>",
                        "P": "<DOWN>",
                        "K": "<LEFT>",
                        "M": "<RIGHT>",
                    }
                    return windows_arrow_map.get(code)
                return ch
            except Exception:
                return None

        if not sys.stdin.isatty():
            return None

        try:
            ch = sys.stdin.read(1)
        except OSError:
            return None

        if not ch:
            return None

        if ch == "\x1b":
            bracket = self._read_nonblocking_byte_with_retry()
            if bracket is None:
                return None  # Lone ESC — discard

            code_str = self._read_nonblocking_byte_with_retry()
            if code_str is None:
                return None

            if bracket == "[":
                unix_arrow_map: Dict[str, str] = {
                    "A": "<UP>",
                    "B": "<DOWN>",
                    "C": "<RIGHT>",
                    "D": "<LEFT>",
                }
                return unix_arrow_map.get(code_str)
            return None

        return ch

    @staticmethod
    def _read_nonblocking_byte_with_retry(
        max_attempts: int = 20, sleep_s: float = 0.001
    ) -> Optional[str]:
        """Read exactly one byte from non-blocking stdin, retrying briefly.

        Used when consuming the 2nd/3rd bytes of a multi-byte escape sequence where the
        kernel may not have buffered the remaining bytes yet.

        Returns the character, or None if it did not arrive within the retry window.
        """
        import time

        for _ in range(max_attempts):
            try:
                byte = sys.stdin.read(1)
                if byte:
                    return byte
            except OSError:
                pass
            time.sleep(sleep_s)
        return None

    def _render_with_interactive(
        self,
        input_buffer: str,
        output_lines: Deque[str],
        command_running: bool,
    ) -> Layout:
        """Render dashboard with an interactive command pane."""
        base = self._render()

        panel_text = Text()
        visible_lines = list(output_lines)[-16:]
        if not visible_lines:
            panel_text.append("No command output yet.\n", style="dim")
        else:
            for line in visible_lines:
                panel_text.append(line)
                panel_text.append("\n")

        if command_running:
            panel_text.append("[running] ", style="yellow")
        else:
            panel_text.append("[idle] ", style="green")
        panel_text.append(f"polaris> {input_buffer}", style="bold cyan")

        root = Layout()
        root.split_column(
            Layout(name="dashboard", ratio=5),
            Layout(name="interactive", size=12),
        )
        root["dashboard"].update(base)
        root["interactive"].update(
            Panel(
                panel_text,
                title="Interactive CLI (Enter=run, Tab=complete, Up/Down=history, Ctrl+C=exit)",
                border_style="cyan",
            )
        )
        return root

    async def _update_metrics_cache(self) -> None:
        """Update cached performance metrics in background."""
        while self.running:
            try:
                if self.polaris.strategy:
                    self._cached_perf_metrics = (
                        await self.polaris.strategy.get_performance_metrics()
                    )
            except Exception as e:
                try:
                    logging.getLogger("polaris.dashboard").error(
                        "Error updating dashboard metrics cache: %s", str(e)
                    )
                except Exception:
                    pass
            await asyncio.sleep(5)

    def _detach_log_handler(self) -> None:
        """Remove the dashboard log handler from the root logger.

        Centralised teardown that nulls out self._log_handler after removal so
        idempotent calls (e.g. both run() and run_with_interactive_cli() reaching their
        finally blocks) do not attempt a second removeHandler.
        """
        if self._log_handler is not None:
            try:
                logging.getLogger("polaris").removeHandler(self._log_handler)
            except Exception:
                pass
            self._log_handler = None

    async def run(self, refresh_rate: float = 1.0) -> None:
        """Run the dashboard.

        Args:
            refresh_rate: Update frequency in seconds
        """
        self.running = True
        metrics_task = asyncio.create_task(self._update_metrics_cache())

        with self._live_display_safe(
            self._render(), refresh_per_second=int(1 / refresh_rate)
        ) as live:
            try:
                while self.running and self.polaris.is_running():
                    await asyncio.sleep(refresh_rate)
                    self._safe_live_update(live, self._render())
            except KeyboardInterrupt:
                pass
            finally:
                self.running = False
                metrics_task.cancel()
                try:
                    await metrics_task
                except asyncio.CancelledError:
                    pass
                self._detach_log_handler()

    async def run_with_interactive_cli(self, refresh_rate: float = 0.2) -> None:
        """Run split-screen dashboard + interactive command mode."""
        self.running = True

        metrics_task = asyncio.create_task(self._update_metrics_cache())
        output_lines: Deque[str] = deque(maxlen=200)
        input_buffer = ""
        command_task: Optional[asyncio.Task] = None
        runner = _EmbeddedInteractiveCLI(self.polaris, output_lines)
        known_commands = runner.command_names()
        command_history: List[str] = []
        history_cursor = 0

        output_lines.append("Split mode active.")
        output_lines.append("Type a command and press Enter (help, status, systems, metrics, ...).")

        with self._raw_terminal_input():
            with self._live_display_safe(
                self._render_with_interactive(
                    input_buffer=input_buffer,
                    output_lines=output_lines,
                    command_running=False,
                ),
                refresh_per_second=max(1, int(1.0 / refresh_rate)),
            ) as live:
                try:
                    while self.running and self.polaris.is_running():
                        if command_task is not None and command_task.done():
                            should_exit = False
                            try:
                                should_exit = bool(command_task.result())
                            except asyncio.CancelledError:
                                # Task was cancelled externally — treat as no-exit
                                pass
                            except Exception as exc:
                                output_lines.append(f"Command failed: {exc}")
                            command_task = None
                            if should_exit:
                                break

                        while True:
                            ch = self._read_key_nonblocking()
                            if ch is None:
                                break

                            if ch in ("\r", "\n"):
                                command = input_buffer.strip()
                                input_buffer = ""
                                if command:
                                    output_lines.append(f"> {command}")
                                    command_history.append(command)
                                    history_cursor = len(command_history)
                                    if command_task is None:
                                        command_task = asyncio.create_task(
                                            asyncio.to_thread(runner.execute, command)
                                        )
                                    else:
                                        output_lines.append(
                                            "A command is already running. Please wait."
                                        )
                            elif ch in ("\x7f", "\b", "\x08"):
                                input_buffer = input_buffer[:-1]
                                history_cursor = len(command_history)
                            elif ch == "\t":
                                prefix = input_buffer.strip()
                                if prefix and " " not in prefix:
                                    matches = [
                                        cmd for cmd in known_commands if cmd.startswith(prefix)
                                    ]
                                    if len(matches) == 1:
                                        input_buffer = matches[0]
                                    elif len(matches) > 1:
                                        output_lines.append(
                                            f"Completions: {', '.join(matches[:8])}"
                                        )
                                history_cursor = len(command_history)
                            elif ch == "<UP>":
                                if command_history:
                                    history_cursor = max(0, history_cursor - 1)
                                    input_buffer = command_history[history_cursor]
                            elif ch == "<DOWN>":
                                if command_history:
                                    history_cursor = min(len(command_history), history_cursor + 1)
                                    if history_cursor == len(command_history):
                                        input_buffer = ""
                                    else:
                                        input_buffer = command_history[history_cursor]
                            elif ch == "\x03":
                                raise KeyboardInterrupt
                            elif ch == "\x04":
                                self.running = False
                                break
                            elif ch.isprintable():
                                input_buffer += ch
                                history_cursor = len(command_history)

                        self._safe_live_update(
                            live,
                            self._render_with_interactive(
                                input_buffer=input_buffer,
                                output_lines=output_lines,
                                command_running=command_task is not None,
                            ),
                        )
                        await asyncio.sleep(refresh_rate)

                except KeyboardInterrupt:
                    pass
                finally:
                    self.running = False
                    metrics_task.cancel()
                    try:
                        await metrics_task
                    except asyncio.CancelledError:
                        pass

                    if command_task is not None and not command_task.done():
                        command_task.cancel()

                    self._detach_log_handler()
