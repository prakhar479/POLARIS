#!/usr/bin/env python3
"""
Launch script for POLARIS Monitor Adapter with Switch system.

This script starts the Monitor Adapter that collects telemetry metrics
from the Switch YOLO system and publishes them to NATS.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add POLARIS src to path
polaris_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(polaris_src))

from polaris.adapters.monitor import MonitorAdapter


def setup_logging():
    """Configure logging for the adapter."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


async def main():
    """Main entry point for the monitor adapter."""
    setup_logging()
    logger = logging.getLogger("switch_monitor")
    
    # Configuration paths
    polaris_config = polaris_src / "config" / "polaris_config.yaml"
    plugin_dir = Path(__file__).parent
    
    logger.info("Starting Switch Monitor Adapter")
    logger.info(f"POLARIS config: {polaris_config}")
    logger.info(f"Plugin directory: {plugin_dir}")
    
    # Create and run the monitor adapter
    adapter = MonitorAdapter(
        polaris_config_path=str(polaris_config),
        plugin_dir=str(plugin_dir),
        logger=logger
    )
    
    try:
        async with adapter:
            logger.info("Monitor adapter started successfully")
            logger.info("Press Ctrl+C to stop")
            
            # Run until interrupted
            while True:
                await asyncio.sleep(1)
                
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Error running monitor adapter: {e}", exc_info=True)
        raise
    finally:
        logger.info("Monitor adapter stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete")
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
