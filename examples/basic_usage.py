"""
Basic usage example for Polaris.

Demonstrates simple integration with SWIM system.
"""

import asyncio

from polaris import Polaris
from polaris.connectors import SWIMConnector
from polaris.strategies import ThresholdReactiveStrategy


async def main():
    """Run Polaris with SWIM system."""
    # Create SWIM connector
    swim = SWIMConnector(host="localhost", port=4242)

    # Create threshold strategy
    strategy = ThresholdReactiveStrategy(
        thresholds={
            # Scale up if response time > 500ms
            "response_time": {"high": 500.0},
            "cpu_usage": {"high": 80.0},  # Scale up if CPU > 80%
        },
        cooldown_seconds=60,
    )

    # Create Polaris instance
    polaris = Polaris(connectors=[swim], strategy=strategy)

    print("Starting Polaris...")
    print("Managing SWIM system with threshold strategy")
    print("Press Ctrl+C to stop\n")

    try:
        # Run Polaris
        await polaris.run()
    except KeyboardInterrupt:
        print("\nStopping Polaris...")
        await polaris.stop()
        print("Stopped.")


if __name__ == "__main__":
    asyncio.run(main())
