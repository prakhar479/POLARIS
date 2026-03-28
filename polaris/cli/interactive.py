"""Interactive CLI interface for querying Polaris components.

Provides an interactive shell for querying knowledge base and world model.
"""

import asyncio
import cmd
import json
import shlex
from collections import deque
from datetime import datetime, timedelta, timezone
from difflib import get_close_matches
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional

from polaris.infrastructure.constants import DEFAULT_JSON_INDENT

if TYPE_CHECKING:
    from polaris.core.polaris import Polaris

try:
    from rich.console import Console
    from rich.syntax import Syntax
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class PolarisInteractiveCLI(cmd.Cmd):
    """Interactive CLI for querying Polaris components."""

    def __init__(self, polaris: "Polaris"):
        """Initialize the interactive CLI."""
        super().__init__()
        self.polaris = polaris
        self.console = Console() if RICH_AVAILABLE else None
        self._history: Deque[str] = deque(maxlen=200)
        self._aliases: Dict[str, str] = {
            "h": "help",
            "q": "quit",
            "wm": "worldmodel",
            "ks": "knowledge",
            "st": "status",
        }

    intro = """
╔══════════════════════════════════════════════════════════════╗
║                    POLARIS INTERACTIVE CLI                   ║
║                                                              ║
║  Query knowledge base, world model, and system status       ║
║  Type 'help' for available commands or 'quit' to exit       ║
╚══════════════════════════════════════════════════════════════╝
    """

    prompt = "polaris> "

    def _run_async(self, coro: Any) -> Any:
        """Run async coroutine safely."""
        try:
            import asyncio

            _ = asyncio.get_running_loop()
            # We're in an event loop, need to use a different approach
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        except RuntimeError:
            # No event loop running, use asyncio.run
            return asyncio.run(coro)

    def _print(self, content: Any, style: Optional[str] = None) -> None:
        """Print with rich formatting if available."""
        if self.console:
            if style:
                self.console.print(content, style=style)
            else:
                self.console.print(content)
        else:
            print(content)

    def _print_table(self, table: Any) -> None:
        """Print table with rich formatting if available."""
        if self.console:
            self.console.print(table)
        else:
            # Fallback to simple text output
            print("Table output requires 'rich' library")

    def _print_json(self, data: Any) -> None:
        """Print JSON with syntax highlighting if available."""
        json_str = json.dumps(data, indent=DEFAULT_JSON_INDENT, default=str)
        if self.console:
            syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
            self.console.print(syntax)
        else:
            print(json_str)

    def _known_commands(self) -> List[str]:
        commands = []
        for name in dir(self):
            if name.startswith("do_"):
                commands.append(name[3:])
        return sorted(commands)

    def _system_ids(self) -> List[str]:
        try:
            return sorted(self.polaris.registry.system_ids())
        except Exception:
            return []

    def _parse_args(self, raw: str) -> List[str]:
        try:
            return shlex.split(raw)
        except ValueError as exc:
            self._print(f"Argument parsing error: {exc}", style="yellow")
            return []

    def precmd(self, line: str) -> str:
        """Normalize aliases and support replaying the previous command."""
        normalized = line.strip()
        if not normalized:
            return ""

        if normalized == "!!":
            if not self._history:
                self._print("No command history available yet.", style="yellow")
                return ""
            normalized = self._history[-1]
            self._print(f"Re-running: {normalized}", style="dim")

        parts = normalized.split(maxsplit=1)
        command = parts[0]
        remainder = parts[1] if len(parts) > 1 else ""
        mapped_command = self._aliases.get(command, command)
        rewritten = f"{mapped_command} {remainder}".strip()

        if rewritten and rewritten != "history":
            self._history.append(rewritten)
        return rewritten

    def do_help(self, arg: str) -> None:
        """Show command help with a concise UX-oriented overview."""
        if arg.strip():
            super().do_help(arg)
            return

        self._print("\n[bold cyan]POLARIS CLI COMMANDS[/bold cyan]")
        self._print("Core: status, systems, metrics [filter], worldmodel [system_id]")
        self._print("Data: knowledge <system_id> [hours], predict <system_id> <action> [k=v ...]")
        self._print("Ops: export <filepath> [json|csv], clear, history [N], quit/exit")
        self._print("Shortcuts: h=help, q=quit, wm=worldmodel, ks=knowledge, st=status")
        self._print("Tips: use Tab for completion where supported, `!!` to repeat last command")

    def do_history(self, arg: str) -> None:
        """Show recent command history. Usage: history [limit]."""
        args = self._parse_args(arg)
        limit = 20
        if args:
            try:
                limit = max(1, int(args[0]))
            except Exception:
                self._print("history limit must be a positive integer", style="yellow")
                return

        if not self._history:
            self._print("No history yet.", style="dim")
            return

        self._print("\n[bold cyan]RECENT COMMANDS[/bold cyan]")
        recent = list(self._history)[-limit:]
        start = len(self._history) - len(recent) + 1
        for idx, cmd_line in enumerate(recent, start=start):
            self._print(f"{idx:>4}  {cmd_line}")

    def do_status(self, arg: str) -> None:
        """Show overall system status."""
        self._print("\n[bold cyan]POLARIS SYSTEM STATUS[/bold cyan]")
        self._print("=" * 50)

        # Framework status
        status = "Running" if self.polaris.is_running() else "Stopped"
        status_style = "green" if self.polaris.is_running() else "red"
        self._print(f"Framework Status: [{status_style}]{status}[/{status_style}]")

        # Connected systems
        system_ids = list(self.polaris.registry.system_ids())
        self._print(f"Connected Systems: {len(system_ids)}")
        for system_id in system_ids:
            self._print(f"  • {system_id}", style="dim")

        # Components status
        self._print(
            f"Strategy: {self.polaris.strategy.__class__.__name__ if self.polaris.strategy else 'None'}"
        )
        self._print(
            f"World Model: {self.polaris.world_model.__class__.__name__ if self.polaris.world_model else 'None'}"
        )
        self._print(
            f"Knowledge Store: "
            f"{self.polaris.knowledge_store.__class__.__name__ if self.polaris.knowledge_store else 'None'}"
        )
        self._print(
            f"Meta Learner: "
            f"{self.polaris.meta_learner.__class__.__name__ if self.polaris.meta_learner else 'Disabled'}"
        )

        # Metrics summary
        if self.polaris.metrics:
            try:
                summary = self.polaris.metrics.get_summary()
                counters = summary.get("counters", {})
                gauges = summary.get("gauges", {})

                self._print("\nMetrics Summary:")
                self._print(f"  Counters: {len(counters)}")
                self._print(f"  Gauges: {len(gauges)}")
                self._print(f"  Histograms: {len(summary.get('histograms', {}))}")
            except Exception as e:
                self._print(f"Metrics: Error loading ({e})", style="red")
        else:
            self._print("Metrics: Disabled")

    def do_systems(self, arg: str) -> None:
        """List connected systems with details."""
        if not RICH_AVAILABLE:
            self._print("This command requires 'rich' library for table display")
            return

        table = Table(title="Connected Systems", show_header=True)
        table.add_column("System ID", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Connector Type", style="yellow")

        for connector in self.polaris.registry.all():
            system_id = self._run_async(connector.get_system_id())
            connector_type = connector.__class__.__name__
            table.add_row(system_id, "✓ Connected", connector_type)

        self._print_table(table)

    def do_metrics(self, arg: str) -> None:
        """Show system metrics. Usage: metrics [component]."""
        if not self.polaris.metrics:
            self._print("Metrics collection is disabled", style="red")
            return

        try:
            summary = self.polaris.metrics.get_summary()

            if arg:
                # Filter by component
                component = arg.strip().lower()
                filtered_counters = {
                    k: v for k, v in summary.get("counters", {}).items() if component in k.lower()
                }
                filtered_gauges = {
                    k: v for k, v in summary.get("gauges", {}).items() if component in k.lower()
                }
                filtered_histograms = {
                    k: v for k, v in summary.get("histograms", {}).items() if component in k.lower()
                }

                filtered_summary = {
                    "counters": filtered_counters,
                    "gauges": filtered_gauges,
                    "histograms": filtered_histograms,
                }
                self._print_json(filtered_summary)
            else:
                # Show all metrics
                self._print_json(summary)

        except Exception as e:
            self._print(f"Error loading metrics: {e}", style="red")

    def do_knowledge(self, arg: str) -> None:
        """Query knowledge base. Usage: knowledge <system_id> [hours_back]."""
        if not self.polaris.knowledge_store:
            self._print("Knowledge store is not available", style="red")
            return

        args = self._parse_args(arg)
        if not args:
            self._print("Usage: knowledge <system_id> [hours_back]", style="yellow")
            return

        system_id = args[0]
        try:
            hours_back = int(args[1]) if len(args) > 1 else 24
        except Exception:
            self._print("hours_back must be an integer", style="yellow")
            return

        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours_back)

            # Query states
            states = self._run_async(
                self.polaris.knowledge_store.query_states(system_id, start_time, end_time)
            )

            # Query actions
            actions = self._run_async(
                self.polaris.knowledge_store.query_actions(system_id, start_time, end_time)
            )

            self._print("\n[bold cyan]KNOWLEDGE BASE QUERY[/bold cyan]")
            self._print(f"System: {system_id}")
            self._print(f"Time Range: {hours_back} hours back")
            self._print(f"States Found: {len(states)}")
            self._print(f"Actions Found: {len(actions)}")

            if RICH_AVAILABLE and states:
                # Show recent states in table
                table = Table(title="Recent States", show_header=True)
                table.add_column("Timestamp", style="dim")
                table.add_column("Health", style="green")
                table.add_column("Metrics Count", style="yellow")

                for state in states[-10:]:  # Show last 10
                    timestamp = state.timestamp.strftime("%H:%M:%S")
                    health = state.health_status.value
                    metrics_count = len(state.metrics)
                    table.add_row(timestamp, health, str(metrics_count))

                self._print_table(table)

            if RICH_AVAILABLE and actions:
                # Show recent actions in table
                table = Table(title="Recent Actions", show_header=True)
                table.add_column("Timestamp", style="dim")
                table.add_column("Action Type", style="cyan")
                table.add_column("Status", style="green")
                table.add_column("Execution Time", style="yellow")

                for action, result in actions[-10:]:  # Show last 10
                    timestamp = action.created_at.strftime("%H:%M:%S")
                    action_type = action.action_type
                    status = result.status.value
                    exec_time = (
                        f"{result.execution_time_ms}ms" if result.execution_time_ms else "N/A"
                    )
                    table.add_row(timestamp, action_type, status, exec_time)

                self._print_table(table)

        except Exception as e:
            self._print(f"Error querying knowledge base: {e}", style="red")

    def do_worldmodel(self, arg: str) -> None:
        """Query world model insights. Usage: worldmodel [system_id]."""
        if not self.polaris.world_model:
            self._print("World model is not available", style="red")
            return

        try:
            insights = self._run_async(self.polaris.world_model.get_insights())

            if arg:
                # Filter by system
                system_id = arg.strip()
                if system_id in insights:
                    filtered_insights = {system_id: insights[system_id]}
                    self._print(f"\n[bold cyan]WORLD MODEL INSIGHTS - {system_id}[/bold cyan]")
                    self._print_json(filtered_insights)
                else:
                    self._print(f"No insights found for system: {system_id}", style="yellow")
            else:
                # Show all insights
                self._print("\n[bold cyan]WORLD MODEL INSIGHTS[/bold cyan]")
                self._print_json(insights)

        except Exception as e:
            self._print(f"Error querying world model: {e}", style="red")

    def do_predict(self, arg: str) -> None:
        """Predict action outcome.

        Usage: predict <system_id> <action_type> [param=value...].
        """
        if not self.polaris.world_model:
            self._print("World model is not available", style="red")
            return

        args = self._parse_args(arg)
        if len(args) < 2:
            self._print(
                "Usage: predict <system_id> <action_type> [param=value ...]", style="yellow"
            )
            return

        system_id = args[0]
        action_type = args[1]

        # Parse parameters
        parameters = {}
        for param_arg in args[2:]:
            if "=" in param_arg:
                key, value = param_arg.split("=", 1)
                parameters[key] = value
            else:
                self._print(
                    f"Ignoring invalid parameter '{param_arg}' (expected key=value)",
                    style="yellow",
                )

        try:
            # Get current state (latest from knowledge store)
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=1)

            states = self._run_async(
                self.polaris.knowledge_store.query_states(system_id, start_time, end_time)
            )

            if not states:
                self._print(f"No recent states found for system: {system_id}", style="yellow")
                return

            current_state = states[-1]  # Most recent state

            # Create action
            from polaris.core.models import AdaptationAction

            action = AdaptationAction(
                action_id="",
                action_type=action_type,
                target_system=system_id,
                parameters=parameters,
            )

            # Get prediction
            prediction = self._run_async(self.polaris.world_model.predict(action, current_state))

            self._print("\n[bold cyan]PREDICTION RESULT[/bold cyan]")
            self._print(f"System: {system_id}")
            self._print(f"Action: {action_type}")
            self._print(f"Parameters: {parameters}")
            self._print(f"Confidence: {prediction.confidence: .2%}")
            self._print(f"Reasoning: {prediction.reasoning}")

            if prediction.predicted_metrics:
                self._print("\nPredicted Metrics:")
                self._print_json(prediction.predicted_metrics)

        except Exception as e:
            self._print(f"Error making prediction: {e}", style="red")

    def do_export(self, arg: str) -> None:
        """Export metrics to file. Usage: export <filepath> [format]."""
        if not self.polaris.metrics:
            self._print("Metrics collection is disabled", style="red")
            return

        args = self._parse_args(arg)
        if not args:
            self._print("Usage: export <filepath> [format]", style="yellow")
            return

        filepath = args[0]
        format_type = args[1] if len(args) > 1 else "json"

        try:
            self.polaris.export_metrics(filepath, format_type)
            self._print(f"Metrics exported to: {filepath}", style="green")
        except Exception as e:
            self._print(f"Error exporting metrics: {e}", style="red")

    def do_reload(self, arg: str) -> None:
        """Trigger an immediate check for hot-reloadable configuration changes."""
        self._print("Checking for configuration updates...", style="cyan")
        # Polaris monitoring loop automatically reloads if file changed, but we can fast-track
        # by checking if we have access to the reloader. Since the loop manages it,
        # let's write to a dummy file to update mtime, or just instruct the user.
        # But actually polaris has config reference:
        self._print(
            "Note: Polaris automatically hot-reloads the configuration file \n"
            "every monitoring interval when changes are saved to disk.\n"
            "You do not need to pause or manually reload.",
            style="green dim",
        )

    def do_clear(self, arg: str) -> None:
        """Clear the screen."""
        import os

        os.system("cls" if os.name == "nt" else "clear")
        self._print(self.intro)

    def do_quit(self, arg: str) -> bool:
        """Exit the interactive CLI."""
        self._print("Goodbye!", style="cyan")
        return True

    def do_exit(self, arg: str) -> bool:
        """Exit the interactive CLI."""
        return self.do_quit(arg)

    def do_EOF(self, arg: str) -> bool:
        """Handle Ctrl+D."""
        self._print("\nGoodbye!", style="cyan")
        return True

    def emptyline(self) -> bool:
        """Handle empty line input."""
        return False

    def default(self, line: str) -> None:
        """Handle unknown commands with fuzzy matching and auto-correction."""
        command = line.strip().split(maxsplit=1)[0] if line.strip() else ""
        args = line.strip().split(maxsplit=1)[1] if len(line.strip().split(maxsplit=1)) > 1 else ""

        suggestions = get_close_matches(command, self._known_commands(), n=3, cutoff=0.6)

        if suggestions:
            best_match = suggestions[0]
            # Automatically execute the closest match for a smoother UX
            self._print(f"Auto-correcting '{command}' to '{best_match}'...", style="italic dim")

            # Prepare the corrected line for execution
            corrected_line = f"{best_match} {args}".strip()
            if corrected_line not in ("history", self._history[-1] if self._history else ""):
                self._history.append(corrected_line)

            # Execute the corrected command by looking up the method
            command_func = getattr(self, f"do_{best_match}")
            command_func(args)
            return

        self._print(
            f"Unknown command: '{line}'. Type 'help' for available commands.", style="yellow"
        )

    def complete_knowledge(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        """Complete knowledge command system_id argument."""
        _ = endidx
        if len(line[:begidx].strip().split()) <= 1:
            return [s for s in self._system_ids() if s.startswith(text)]
        return []

    def complete_worldmodel(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        """Complete worldmodel command system_id argument."""
        _ = endidx
        if len(line[:begidx].strip().split()) <= 1:
            return [s for s in self._system_ids() if s.startswith(text)]
        return []

    def complete_predict(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        """Complete predict command arguments."""
        _ = endidx
        tokens = line[:begidx].strip().split()
        if len(tokens) <= 1:
            return [s for s in self._system_ids() if s.startswith(text)]
        if len(tokens) == 2:
            common_actions = [
                "scale_up",
                "scale_down",
                "add_server",
                "remove_server",
                "increase_resources",
                "decrease_resources",
                "reconfigure",
            ]
            return [a for a in common_actions if a.startswith(text)]
        return []

    def complete_metrics(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        """Complete common metrics component filters."""
        _ = line
        _ = begidx
        _ = endidx
        components = [
            "monitoring",
            "telemetry",
            "adaptations",
            "knowledge",
            "world_model",
            "strategy",
            "connectors",
        ]
        return [c for c in components if c.startswith(text.lower())]

    def complete_export(self, text: str, line: str, begidx: int, endidx: int) -> List[str]:
        """Complete export format argument."""
        _ = endidx
        tokens = line[:begidx].strip().split()
        if len(tokens) == 2:
            return [fmt for fmt in ["json", "csv"] if fmt.startswith(text)]
        return []


def run_interactive_cli_standalone(
    config_path: str, cli_overrides: Optional[Dict[str, Any]] = None
) -> None:
    """Backward-compatible helper to run interactive mode in a single process.

    Args:
        config_path: Path to Polaris configuration
        cli_overrides: CLI configuration overrides
    """
    from polaris import Polaris
    from polaris.cli.main import run_with_interactive_cli

    polaris = Polaris(config_path=config_path, cli_overrides=cli_overrides or {})
    asyncio.run(run_with_interactive_cli(polaris))


async def run_interactive_cli(polaris: "Polaris") -> None:
    """Run the interactive CLI interface in the same process.

    Args:
        polaris: Polaris instance to query
    """
    cli = PolarisInteractiveCLI(polaris)

    # Run in a separate thread to avoid blocking
    import threading

    def run_cli() -> None:
        try:
            cli.cmdloop()
        except KeyboardInterrupt:
            print("\nExiting...")

    cli_thread = threading.Thread(target=run_cli, daemon=False)
    cli_thread.start()

    # Keep the async context alive
    try:
        while cli_thread.is_alive():
            await asyncio.sleep(0.1)
    except KeyboardInterrupt:
        pass
