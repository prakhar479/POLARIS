"""
Interactive CLI interface for querying Polaris components.

Provides an interactive shell for querying knowledge base and world model.
"""

import asyncio
import cmd
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

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
        json_str = json.dumps(data, indent=2, default=str)
        if self.console:
            syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
            self.console.print(syntax)
        else:
            print(json_str)

    def do_status(self, arg: str) -> None:
        """Show overall system status.."""
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
        """List connected systems with details.."""
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
        """Show system metrics. Usage: metrics [component].."""
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
        """Query knowledge base. Usage: knowledge <system_id> [hours_back].."""
        if not self.polaris.knowledge_store:
            self._print("Knowledge store is not available", style="red")
            return

        args = arg.split()
        if not args:
            self._print("Usage: knowledge <system_id> [hours_back]", style="yellow")
            return

        system_id = args[0]
        hours_back = int(args[1]) if len(args) > 1 else 24

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
        """Query world model insights. Usage: worldmodel [system_id].."""
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
        """Predict action outcome. Usage: predict <system_id> <action_type> [param=value ...].."""
        if not self.polaris.world_model:
            self._print("World model is not available", style="red")
            return

        args = arg.split()
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
        """Export metrics to file. Usage: export <filepath> [format].."""
        if not self.polaris.metrics:
            self._print("Metrics collection is disabled", style="red")
            return

        args = arg.split()
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

    def do_clear(self, arg: str) -> None:
        """Clear the screen.."""
        import os

        os.system("cls" if os.name == "nt" else "clear")
        self._print(self.intro)

    def do_quit(self, arg: str) -> bool:
        """Exit the interactive CLI.."""
        self._print("Goodbye!", style="cyan")
        return True

    def do_exit(self, arg: str) -> bool:
        """Exit the interactive CLI.."""
        return self.do_quit(arg)

    def do_EOF(self, arg: str) -> bool:
        """Handle Ctrl+D.."""
        self._print("\nGoodbye!", style="cyan")
        return True

    def emptyline(self) -> bool:
        """Handle empty line input."""
        return False

    def default(self, line: str) -> None:
        """Handle unknown commands."""
        self._print(f"Unknown command: {line}. Type 'help' for available commands.", style="yellow")


def run_interactive_cli_standalone(
    config_path: str, cli_overrides: Optional[Dict[str, Any]] = None
) -> None:
    """
    Run the interactive CLI as a standalone process.

    Args:
        config_path: Path to Polaris configuration
        cli_overrides: CLI configuration overrides
    """
    import json
    import os
    import subprocess
    import sys
    import tempfile

    # Create a temporary script to run the CLI
    cli_script = f'''
import sys
import asyncio
import json
from pathlib import Path

# Add the polaris package to path
sys.path.insert(0, "{Path(__file__).parent.parent}")

from polaris import Polaris
from polaris.cli.interactive import PolarisInteractiveCLI

async def main():
    """Main function for standalone CLI."""
    # Load CLI overrides
    cli_overrides = {json.dumps(cli_overrides or {})}
    cli_overrides = json.loads(cli_overrides) if cli_overrides != "null" else {{}}

    # Create Polaris instance
    polaris = Polaris(config_path= "{config_path}", cli_overrides=cli_overrides)

    # Start Polaris in background
    polaris_task = asyncio.create_task(polaris.run())

    # Wait a moment for initialization
    await asyncio.sleep(2)

    # Create and run CLI
    cli = PolarisInteractiveCLI(polaris)

    print("\\n" + "="*60)
    print("POLARIS INTERACTIVE CLI - STANDALONE MODE")
    print("="*60)
    print("Connected to Polaris framework")
    print("Type 'help' for available commands or 'quit'  to exit")
    print("="*60)

    try:
        # Run CLI in main thread
        cli.cmdloop()
    except KeyboardInterrupt:
        print("\\nShutting down...")
    finally:
        await polaris.stop()
        polaris_task.cancel()
        try:
            await polaris_task
        except asyncio.CancelledError:
            pass

if __name__ == "__main__":
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from polaris.core.polaris import Polaris
    asyncio.run(main())

'''  # noqa: E271

    # Write script to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(cli_script)
        temp_script = f.name

    try:
        # Launch in new terminal/process
        if sys.platform == "win32":
            # Windows
            subprocess.Popen(["cmd", "/c", "start", "cmd", "/k", f"python {temp_script} && pause"])
        else:
            # Linux/Mac - try different terminal emulators
            terminals = [
                ["gnome-terminal", "--", "python", temp_script],
                ["xterm", "-e", f"python {temp_script} && read -p 'Press Enter to close...'"],
                ["konsole", "-e", "python", temp_script],
                ["terminal", "-e", "python", temp_script],
            ]

            launched = False
            for terminal_cmd in terminals:
                try:
                    subprocess.Popen(terminal_cmd)
                    launched = True
                    break
                except FileNotFoundError:
                    continue

            if not launched:
                # Fallback: run in background with nohup
                print("No terminal emulator found. Running CLI in background...")
                print("Check the process list for the interactive CLI.")
                subprocess.Popen(
                    ["nohup", "python", temp_script],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

        print("Interactive CLI launched in separate terminal/process")

    except Exception as e:
        print(f"Failed to launch interactive CLI: {e}")
        print("You can run it manually with:")
        print(f"python {temp_script}")

    # Clean up after a delay (let the process start first)
    import threading

    def cleanup() -> None:
        import time

        time.sleep(10)  # Wait 10 seconds
        try:
            os.unlink(temp_script)
        except Exception:
            pass

    threading.Thread(target=cleanup, daemon=True).start()


async def run_interactive_cli(polaris: "Polaris") -> None:
    """
    Run the interactive CLI interface in the same process.

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

    cli_thread = threading.Thread(target=run_cli, daemon=True)
    cli_thread.start()

    # Keep the async context alive
    try:
        while cli_thread.is_alive():
            await asyncio.sleep(0.1)
    except KeyboardInterrupt:
        pass
