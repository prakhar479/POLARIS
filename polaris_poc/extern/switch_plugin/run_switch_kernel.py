#!/usr/bin/env python3
"""
Launch script for SWITCH Kernel.

This script starts the SWITCH kernel which orchestrates adaptation
by routing telemetry to fast or slow controllers based on system state.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from polaris.kernel.switch_kernel import SwitchKernel


def setup_logging():
    """Configure logging for SWITCH kernel."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/switch_kernel.log')
        ]
    )
    return logging.getLogger("SwitchKernel")


async def main():
    """Main entry point for SWITCH kernel."""
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("Starting SWITCH Kernel for POLARIS Framework")
    logger.info("=" * 80)
    
    # NATS configuration
    nats_url = "nats://localhost:4222"
    logger.info(f"NATS URL: {nats_url}")
    
    # Create kernel
    kernel = SwitchKernel(nats_url=nats_url, logger=logger)
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, initiating shutdown...")
        kernel.running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start kernel
        await kernel.start()
        logger.info("SWITCH Kernel started successfully")
        logger.info("Subscribed to: polaris.telemetry.events.batch")
        logger.info("Publishing to: polaris.verification.requests (or polaris.execution.actions)")
        logger.info("Reasoner routing: polaris.reasoner.kernel.requests")
        logger.info("-" * 80)
        logger.info("Kernel is running. Press Ctrl+C to stop.")
        logger.info("=" * 80)
        
        # Run until interrupted
        while kernel.running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Error in kernel: {e}", exc_info=True)
    finally:
        logger.info("Shutting down SWITCH Kernel...")
        await kernel.stop()
        logger.info("SWITCH Kernel stopped")


if __name__ == "__main__":
    asyncio.run(main())
