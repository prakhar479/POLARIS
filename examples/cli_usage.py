"""
Example usage of Polaris CLI features.

Demonstrates the enhanced dashboard and interactive CLI interface.
"""

import asyncio

from polaris import Polaris
from polaris.cli import Dashboard, run_interactive_cli


async def demo_dashboard():
    """Demonstrate the enhanced dashboard with system metrics."""
    print("=== POLARIS DASHBOARD DEMO ===")
    print("Starting Polaris with enhanced dashboard...")

    # Create Polaris instance
    polaris = Polaris(config_path="config/default.yaml")

    # Create and run dashboard
    dashboard = Dashboard(polaris)

    # Run both concurrently
    polaris_task = asyncio.create_task(polaris.run())
    dashboard_task = asyncio.create_task(dashboard.run(refresh_rate=1.0))

    try:
        # Wait for either to complete
        done, pending = await asyncio.wait(
            [polaris_task, dashboard_task], return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel remaining tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    except KeyboardInterrupt:
        print("\nShutting down dashboard demo...")
    finally:
        await polaris.stop()


async def demo_interactive_cli():
    """Demonstrate the interactive CLI interface."""
    print("=== POLARIS INTERACTIVE CLI DEMO ===")
    print("Starting Polaris with interactive CLI...")

    # Create Polaris instance
    polaris = Polaris(config_path="config/default.yaml")

    # Run both concurrently
    polaris_task = asyncio.create_task(polaris.run())
    cli_task = asyncio.create_task(run_interactive_cli(polaris))

    try:
        # Wait for either to complete
        done, pending = await asyncio.wait(
            [polaris_task, cli_task], return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel remaining tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    except KeyboardInterrupt:
        print("\nShutting down interactive CLI demo...")
    finally:
        await polaris.stop()


async def demo_programmatic_queries():
    """Demonstrate programmatic querying of components."""
    print("=== PROGRAMMATIC QUERY DEMO ===")

    # Create Polaris instance
    polaris = Polaris(config_path="config/default.yaml")

    try:
        # Start Polaris
        _ = asyncio.create_task(polaris.run())

        # Wait a bit for initialization
        await asyncio.sleep(2)

        print("Querying system status...")

        # Query system metrics
        if polaris.metrics:
            summary = polaris.metrics.get_summary()
            print(
                f"System metrics: {len(summary.get('counters', {}))} counters, "
                f"{len(summary.get('gauges', {}))} gauges"
            )

        # Query world model insights
        if polaris.world_model:
            insights = await polaris.world_model.get_insights()
            print(f"World model insights for {len(insights)} systems")

        # Query knowledge store
        if polaris.knowledge_store:
            from datetime import datetime, timedelta, timezone

            end_time = datetime.now(timezone.utc)
            _ = end_time - timedelta(hours=1)

            # This would work if we had connected systems
            # states = await polaris.knowledge_store.query_states("system1", start_time, end_time)
            # print(f"Found {len(states)} states in knowledge store")

        print("Demo completed successfully!")

    except KeyboardInterrupt:
        print("\nShutting down programmatic demo...")
    finally:
        await polaris.stop()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        demo_type = sys.argv[1]

        if demo_type == "dashboard":
            asyncio.run(demo_dashboard())
        elif demo_type == "interactive":
            asyncio.run(demo_interactive_cli())
        elif demo_type == "programmatic":
            asyncio.run(demo_programmatic_queries())
        else:
            print("Usage: python cli_usage.py [dashboard|interactive|programmatic]")
    else:
        print("Available demos:")
        print("  python cli_usage.py dashboard     - Enhanced dashboard with system metrics")
        print("  python cli_usage.py interactive   - Interactive CLI interface")
        print("  python cli_usage.py programmatic  - Programmatic component querying")
