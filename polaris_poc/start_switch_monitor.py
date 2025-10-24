#!/usr/bin/env python3
"""
Startup script for Switch System Monitor Adapter.

This script initializes and runs the Switch monitor adapter that collects
comprehensive metrics from the Switch YOLO system and publishes them to NATS.
"""

import asyncio
import logging
import sys
import signal
from pathlib import Path
from typing import Optional

# Add the extern directory to the path
sys.path.insert(0, str(Path(__file__).parent / "extern"))

from switch_monitor_adapter import SwitchMonitorAdapter


class SwitchMonitorService:
    """Service wrapper for the Switch monitor adapter."""
    
    def __init__(
        self,
        polaris_config_path: str = "config/switch_optimized_config.yaml",
        plugin_dir: str = "extern/switch_plugin",
        log_level: str = "INFO"
    ):
        """Initialize the Switch monitor service.
        
        Args:
            polaris_config_path: Path to POLARIS configuration file
            plugin_dir: Path to Switch plugin directory
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.polaris_config_path = polaris_config_path
        self.plugin_dir = plugin_dir
        self.log_level = log_level
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('logs/switch_monitor.log', mode='a')
            ]
        )
        
        self.logger = logging.getLogger("switch_monitor_service")
        self.adapter: Optional[SwitchMonitorAdapter] = None
        self.running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    async def start(self) -> None:
        """Start the Switch monitor service."""
        try:
            self.logger.info("Starting Switch Monitor Service")
            self.logger.info(f"POLARIS Config: {self.polaris_config_path}")
            self.logger.info(f"Plugin Directory: {self.plugin_dir}")
            self.logger.info(f"Log Level: {self.log_level}")
            
            # Create the monitor adapter
            self.adapter = SwitchMonitorAdapter(
                polaris_config_path=self.polaris_config_path,
                plugin_dir=self.plugin_dir,
                logger=self.logger
            )
            
            # Start the adapter
            await self.adapter.__aenter__()
            self.running = True
            
            self.logger.info("Switch Monitor Service started successfully")
            self.logger.info("Collecting metrics every 30 seconds...")
            self.logger.info("Press Ctrl+C to stop")
            
            # Run until stopped
            while self.running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Service interrupted by user")
        except Exception as e:
            self.logger.error(f"Service failed: {e}", exc_info=True)
            raise
        finally:
            await self.stop()
    
    async def stop(self) -> None:
        """Stop the Switch monitor service."""
        if self.adapter:
            try:
                self.logger.info("Stopping Switch Monitor Service...")
                await self.adapter.__aexit__(None, None, None)
                self.logger.info("Switch Monitor Service stopped")
            except Exception as e:
                self.logger.error(f"Error stopping service: {e}")
        
        self.running = False
    
    async def get_status(self) -> dict:
        """Get service status information.
        
        Returns:
            Dictionary containing service status
        """
        if not self.adapter:
            return {"status": "not_started"}
        
        try:
            # Get adapter status
            status = {
                "status": "running" if self.running else "stopped",
                "adapter_connected": self.adapter._connected if hasattr(self.adapter, '_connected') else False,
                "queue_size": self.adapter.telemetry_queue.qsize() if self.adapter.telemetry_queue else 0,
                "model_switch_count": getattr(self.adapter, 'model_switch_count', 0),
                "last_model": getattr(self.adapter, 'last_model', 'unknown'),
                "history_size": len(getattr(self.adapter, 'performance_history', []))
            }
            
            return status
            
        except Exception as e:
            return {"status": "error", "error": str(e)}


async def main():
    """Main entry point for the Switch monitor service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Switch System Monitor Adapter")
    parser.add_argument(
        "--config", 
        default="config/switch_optimized_config.yaml",
        help="Path to POLARIS configuration file"
    )
    parser.add_argument(
        "--plugin-dir",
        default="extern/switch_plugin", 
        help="Path to Switch plugin directory"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show service status and exit"
    )
    
    args = parser.parse_args()
    
    # Create service
    service = SwitchMonitorService(
        polaris_config_path=args.config,
        plugin_dir=args.plugin_dir,
        log_level=args.log_level
    )
    
    if args.status:
        # Show status and exit
        status = await service.get_status()
        print(f"Switch Monitor Service Status: {status}")
        return
    
    # Start service
    try:
        await service.start()
    except Exception as e:
        print(f"Failed to start Switch Monitor Service: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    # Run the service
    asyncio.run(main())