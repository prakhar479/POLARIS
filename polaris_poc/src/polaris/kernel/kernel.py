import asyncio
import json
import logging
from abc import ABC, abstractmethod
from polaris.common.nats_client import NATSClient
from polaris.controllers.fast_controller import FastController
from polaris.controllers.slow_controller import SlowController
from polaris.controllers.controller_strategy import ControllerStrategy

# BaseKernel class
class BaseKernel(ABC):
    def __init__(self, nats_url: str, logger: logging.Logger, name: str):
        self.nats_client = NATSClient(nats_url=nats_url, logger=logger, name=name)
        self.logger = logger
        self.running = False

    async def start(self):
        """Start the kernel."""
        self.logger.info(f"Starting {self.__class__.__name__}")
        await self.nats_client.connect()
        self.running = True

        await self.nats_client.subscribe("polaris.telemetry.events.batch", self.process_telemetry_event)
        self.logger.info(f"Subscribed to 'polaris.telemetry.events.batch'")
        pass

    async def stop(self):
        self.logger.info(f"Stopping {self.__class__.__name__}")
        self.running = False
        await self.nats_client.close()

    @abstractmethod
    async def process_telemetry_event(self, msg):
        """Process telemetry events received from NATS."""
        pass

    @abstractmethod
    def generate_action(self, telemetry_data):
        """Generate an action based on telemetry data."""
        pass

# SWIMKernel class
class SWIMKernel(BaseKernel):
    def __init__(self, nats_url: str, logger: logging.Logger):
        super().__init__(nats_url=nats_url, logger=logger, name="SWIMKernel")
        self.strategy = ControllerStrategy()

    async def process_telemetry_event(self, msg):
        try:
            telemetry_data = json.loads(msg.data.decode())
            controller = self.strategy.select_controller(telemetry_data)
            self.logger.info(f"Selected controller: {controller.__class__.__name__}")
            action = self.generate_action(telemetry_data)
            if action is None:
                self.logger.warning("No action generated from telemetry data")
                return
            await self.nats_client.publish("polaris.execution.actions", json.dumps(action).encode())
            self.logger.info("Published action", extra={"action": action})
        except Exception as e:
            self.logger.error("Error processing telemetry event", extra={"error": str(e)})

    def generate_action(self, telemetry_data):
        return self.controller.decide_action(telemetry_data)
    
async def main():
    logger = logging.getLogger("SWIMKernel")
    logging.basicConfig(level=logging.INFO)

    kernel = SWIMKernel(nats_url="nats://localhost:4222", logger=logger)
    try:
        await kernel.start()
        while kernel.running:
            await asyncio.sleep(1)
    finally:
        await kernel.stop()

if __name__ == "__main__":
    asyncio.run(main())

