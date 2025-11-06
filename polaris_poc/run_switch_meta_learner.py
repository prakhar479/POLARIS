#!/usr/bin/env python3
"""
Startup script for SWITCH Meta Learner Agent.

This script initializes and runs the SWITCH-specific meta learner that continuously
optimizes the YOLO model switching system parameters for maximum utility.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from polaris.agents.switch_meta_learner import SwitchMetaLearnerAgent
from polaris.agents.meta_learner_agent import TriggerType


class SwitchMetaLearnerService:
    """Service wrapper for the SWITCH meta learner agent."""
    
    def __init__(
        self,
        config_path: str = "config/switch_optimized_config.yaml",
        log_level: str = "INFO"
    ):
        """Initialize the SWITCH meta learner service.
        
        Args:
            config_path: Path to POLARIS configuration file
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.config_path = config_path
        self.log_level = log_level
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('logs/switch_meta_learner.log', mode='a')
            ]
        )
        
        self.logger = logging.getLogger("switch_meta_learner_service")
        self.agent: Optional[SwitchMetaLearnerAgent] = None
        self.running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    async def start(self) -> None:
        """Start the SWITCH meta learner service."""
        try:
            self.logger.info("Starting SWITCH Meta Learner Service")
            self.logger.info(f"Config: {self.config_path}")
            self.logger.info(f"Log Level: {self.log_level}")
            
            # Create the meta learner agent
            self.agent = SwitchMetaLearnerAgent(
                agent_id="switch_meta_learner",
                config_path=self.config_path,
                logger=self.logger
            )
            
            # Start the agent
            await self.agent.start()
            self.running = True
            
            self.logger.info("SWITCH Meta Learner Service started successfully")
            self.logger.info("High-frequency learning enabled:")
            self.logger.info(f"  - Periodic analysis: every {self.agent.periodic_interval_minutes} minutes")
            self.logger.info(f"  - Calibration: every {self.agent.calibration_frequency_minutes} minutes")
            self.logger.info(f"  - Analysis window: {self.agent.analysis_window_hours} hours")
            self.logger.info("Press Ctrl+C to stop")
            
            # Start periodic trigger loop
            await self._run_periodic_triggers()
                
        except KeyboardInterrupt:
            self.logger.info("Service interrupted by user")
        except Exception as e:
            self.logger.error(f"Service failed: {e}", exc_info=True)
            raise
        finally:
            await self.stop()
    
    async def _run_periodic_triggers(self):
        """Run periodic meta-learning triggers at high frequency."""
        trigger_interval = self.agent.periodic_interval_minutes * 60  # Convert to seconds
        
        while self.running:
            try:
                # Trigger periodic meta-learning cycle
                self.logger.info("Triggering periodic SWITCH meta-learning cycle")
                
                success = await self.agent.handle_trigger(
                    TriggerType.PERIODIC,
                    {
                        "source": "periodic_scheduler",
                        "interval_minutes": self.agent.periodic_interval_minutes,
                        "timestamp": asyncio.get_event_loop().time()
                    }
                )
                
                if success:
                    self.logger.info("Periodic meta-learning cycle completed successfully")
                else:
                    self.logger.warning("Periodic meta-learning cycle failed")
                
                # Wait for next cycle
                await asyncio.sleep(trigger_interval)
                
            except Exception as e:
                self.logger.error(f"Error in periodic trigger loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def stop(self) -> None:
        """Stop the SWITCH meta learner service."""
        if self.agent:
            try:
                self.logger.info("Stopping SWITCH Meta Learner Service...")
                await self.agent.stop()
                self.logger.info("SWITCH Meta Learner Service stopped")
            except Exception as e:
                self.logger.error(f"Error stopping service: {e}")
        
        self.running = False
    
    async def get_status(self) -> dict:
        """Get service status information.
        
        Returns:
            Dictionary containing service status
        """
        if not self.agent:
            return {"status": "not_started"}
        
        try:
            status = {
                "status": "running" if self.running else "stopped",
                "agent_id": self.agent.agent_id,
                "periodic_interval_minutes": self.agent.periodic_interval_minutes,
                "calibration_frequency_minutes": self.agent.calibration_frequency_minutes,
                "analysis_window_hours": self.agent.analysis_window_hours,
                "learning_cycles_completed": getattr(self.agent, 'applied_updates_count', 0),
                "last_analysis_time": self.agent.last_analysis_time.isoformat() if self.agent.last_analysis_time else None,
                "last_calibration_time": self.agent.last_calibration_time.isoformat() if self.agent.last_calibration_time else None
            }
            
            return status
            
        except Exception as e:
            return {"status": "error", "error": str(e)}


async def main():
    """Main entry point for the SWITCH meta learner service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SWITCH Meta Learner Agent")
    parser.add_argument(
        "--config", 
        default="config/switch_optimized_config.yaml",
        help="Path to POLARIS configuration file"
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
    service = SwitchMetaLearnerService(
        config_path=args.config,
        log_level=args.log_level
    )
    
    if args.status:
        # Show status and exit
        status = await service.get_status()
        print(f"SWITCH Meta Learner Service Status: {status}")
        return
    
    # Start service
    try:
        await service.start()
    except Exception as e:
        print(f"Failed to start SWITCH Meta Learner Service: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    # Run the service
    asyncio.run(main())