"""
Example: Using configuration file.

Shows how to use Polaris with a YAML configuration file.
"""

import asyncio

from polaris import Polaris


async def main():
    """Run Polaris with configuration file."""
    # Create Polaris from config file
    # This will:
    # - Load systems from config
    # - Create connectors automatically
    # - Set up strategy from config
    # - Apply observability settings
    polaris = Polaris(config_path="config/default.yaml")

    print("Starting Polaris from configuration...")
    print("Config: config/default.yaml")
    print("Press Ctrl+C to stop\n")

    try:
        await polaris.run()
    except KeyboardInterrupt:
        print("\nStopping...")
        await polaris.stop()
        print("Stopped.")


if __name__ == "__main__":
    asyncio.run(main())
