"""
Enhanced CLI with dashboard support.
"""

import asyncio
import argparse
import os
import sys
from pathlib import Path

from polaris import Polaris


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Polaris - Modular Self-Adaptive Systems Framework"
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to configuration file (YAML)"
    )

    parser.add_argument(
        "--dashboard",
        "-d",
        action="store_true",
        help="Launch interactive dashboard with real-time system metrics"
    )

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Launch interactive CLI interface in separate terminal for querying knowledge base and world model"
    )

    parser.add_argument(
        "--both",
        "-b",
        action="store_true",
        help="Launch both dashboard and interactive CLI together"
    )

    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not clear the terminal when launching the dashboard"
    )

    parser.add_argument(
        "--version",
        "-v",
        action="store_true",
        help="Show version and exit"
    )

    parser.add_argument(
        "--export-logs",
        "-e",
        type=str,
        metavar="FILE",
        help="Export logs to specified file (overrides config file setting)"
    )

    parser.add_argument(
        "--log-format",
        choices=["structured", "human"],
        help="Log format type (overrides config file setting)"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (overrides config file setting)"
    )

    parser.add_argument(
        "--metrics-export",
        "-m",
        type=str,
        metavar="DIR",
        help="Export metrics to specified directory (overrides config file setting)"
    )

    parser.add_argument(
        "--metrics-format",
        choices=["json", "csv", "both"],
        default="both",
        help="Metrics export format (default: both)"
    )

    parser.add_argument(
        "--metrics-experiment",
        type=str,
        metavar="NAME",
        help="Experiment name for metrics files"
    )

    parser.add_argument(
        "--disable-metrics",
        action="store_true",
        help="Disable metrics collection entirely"
    )

    parser.add_argument(
        "--auto-export-metrics",
        type=int,
        metavar="MINUTES",
        help="Auto-export metrics every N minutes (0 to disable)"
    )

    parser.add_argument(
        "--monitoring-interval",
        type=int,
        metavar="SECONDS",
        help="Monitoring loop interval in seconds (overrides config file setting)"
    )

    args = parser.parse_args()

    if args.version:
        from polaris import __version__
        print(f"Polaris {__version__}")
        return 0

    # Load configuration
    config_path = args.config or "config/default.yaml"

    if not Path(config_path).exists():
        print(f"Error: Configuration file not found: {config_path}")
        print("Please specify a valid config file with --config")
        return 1

    try:
        # Create Polaris instance with CLI overrides
        cli_overrides = {}
        if args.export_logs:
            cli_overrides['log_file'] = args.export_logs
        if args.log_format:
            cli_overrides['log_format'] = args.log_format
        if args.log_level:
            cli_overrides['log_level'] = args.log_level
        # In dashboard modes, suppress raw console logging to keep TUI clean.
        if args.dashboard or args.both:
            cli_overrides['console_logging'] = False
        
        # Metrics CLI overrides
        if args.disable_metrics:
            cli_overrides['metrics_enabled'] = False
        if args.metrics_export:
            cli_overrides['metrics_export_dir'] = args.metrics_export
        if args.metrics_format:
            formats = ['json', 'csv'] if args.metrics_format == 'both' else [args.metrics_format]
            cli_overrides['metrics_export_formats'] = formats
        if args.metrics_experiment:
            cli_overrides['metrics_experiment_name'] = args.metrics_experiment
        if args.auto_export_metrics is not None:
            cli_overrides['metrics_auto_export_interval'] = args.auto_export_metrics
        if args.monitoring_interval:
            cli_overrides['monitoring_interval'] = args.monitoring_interval

        polaris = Polaris(config_path=config_path, cli_overrides=cli_overrides)

        if args.dashboard:
            # Launch with dashboard
            print(f"Starting Polaris with interactive dashboard")
            print(f"Config: {config_path}")
            if args.export_logs:
                print(f"Exporting logs to: {args.export_logs}")
            print()
            # Clear screen by default for a clean dashboard unless --no-clear is set
            asyncio.run(run_with_dashboard(polaris, clear_screen=not args.no_clear))
        elif args.interactive:
            # Launch interactive CLI in separate process
            print(f"Launching Polaris interactive CLI in separate terminal")
            print(f"Config: {config_path}")
            if args.export_logs:
                print(f"Logs will be exported to: {args.export_logs}")
            print()
            
            from polaris.cli.interactive import run_interactive_cli_standalone
            run_interactive_cli_standalone(config_path, cli_overrides)
            
            # Also start the main framework
            print(f"Starting main Polaris framework")
            print("Press Ctrl+C to stop\n")
            asyncio.run(run_framework(polaris))
        elif args.both:
            # Launch both dashboard and interactive CLI
            print(f"Starting Polaris with both dashboard and interactive CLI")
            print(f"Config: {config_path}")
            if args.export_logs:
                print(f"Exporting logs to: {args.export_logs}")
            print()
            
            # Launch interactive CLI in separate terminal
            from polaris.cli.interactive import run_interactive_cli_standalone
            run_interactive_cli_standalone(config_path, cli_overrides)
            
            # Run dashboard in main process
            asyncio.run(run_with_dashboard(polaris, clear_screen=not args.no_clear))
        else:
            # Standard CLI
            print(f"Starting Polaris with config: {config_path}")
            if args.export_logs:
                print(f"Exporting logs to: {args.export_logs}")
            if args.metrics_export:
                print(f"Exporting metrics to: {args.metrics_export}")
            if args.disable_metrics:
                print("Metrics collection disabled")
            print("Press Ctrl+C to stop\n")
            asyncio.run(run_framework(polaris))

        return 0

    except KeyboardInterrupt:
        print("\nShutting down...")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


async def run_framework(polaris: Polaris):
    """Run Polaris without dashboard."""
    try:
        await polaris.run()
    except KeyboardInterrupt:
        await polaris.stop()


async def run_with_interactive_cli(polaris: Polaris):
    """Run Polaris with interactive CLI interface."""
    try:
        from polaris.cli.interactive import run_interactive_cli
    except ImportError:
        print("Error: Interactive CLI requires 'rich' library")
        print("Install with: pip install rich")
        return

    # Run Polaris and interactive CLI concurrently
    polaris_task = asyncio.create_task(polaris.run())
    cli_task = asyncio.create_task(run_interactive_cli(polaris))

    try:
        # Wait for either to complete
        done, pending = await asyncio.wait(
            [polaris_task, cli_task],
            return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel remaining tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    except KeyboardInterrupt:
        pass
    finally:
        await polaris.stop()


async def run_with_dashboard(polaris: Polaris, clear_screen: bool = False):
    """Run Polaris with interactive dashboard.

    Args:
        polaris: Polaris framework instance.
        clear_screen: If True, clear the terminal before starting dashboard.
    """
    try:
        from polaris.cli.dashboard import Dashboard
    except ImportError:
        print("Error: Dashboard requires 'rich' library")
        print("Install with: pip install rich")
        return

    # Optionally clear the terminal for a clean TUI
    if clear_screen:
        try:
            os.system('cls' if os.name == 'nt' else 'clear')
        except Exception:
            # Best-effort only; failing to clear is non-fatal
            pass

    # Create dashboard
    dashboard = Dashboard(polaris)

    # Run Polaris and dashboard concurrently
    polaris_task = asyncio.create_task(polaris.run())
    dashboard_task = asyncio.create_task(dashboard.run(refresh_rate=1.0))

    try:
        # Wait for either to complete
        done, pending = await asyncio.wait(
            [polaris_task, dashboard_task],
            return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel remaining tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    except KeyboardInterrupt:
        pass
    finally:
        await polaris.stop()


if __name__ == "__main__":
    sys.exit(main())
