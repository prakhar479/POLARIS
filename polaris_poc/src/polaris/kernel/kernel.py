import asyncio
import json
import logging
from polaris.common.nats_client import NATSClient
from polaris.controllers.fast_controller import FastController

class SWIMKernel:
    def __init__(self, nats_url: str, logger: logging.Logger):
        self.nats_client = NATSClient(nats_url=nats_url, logger=logger, name="SWIMKernel")
        self.logger = logger
        self.running = False
        self.fast_controller = FastController(self)

    async def start(self):
        """Start the kernel."""
        self.logger.info("Starting SWIM Kernel")
        await self.nats_client.connect()
        self.running = True

        # Subscribe to telemetry events
        await self.nats_client.subscribe("polaris.telemetry.events.batch", self.process_telemetry_event)

    async def stop(self):
        """Stop the kernel."""
        self.logger.info("Stopping SWIM Kernel")
        self.running = False
        await self.nats_client.close()

    async def process_telemetry_event(self, msg):
        """Process telemetry events received from SWIM."""
        try:
            telemetry_data = json.loads(msg.data.decode())
            # self.logger.info(f"Received telemetry event - {telemetry_data}", extra={"data": telemetry_data})

            # Example: Generate an action based on telemetry data
            action = self.generate_action(telemetry_data)

            # Publish the action to NATS
            await self.nats_client.publish("polaris.execution.actions", json.dumps(action).encode())
            self.logger.info("Published action", extra={"action": action})
        except Exception as e:
            self.logger.error("Error processing telemetry event", extra={"error": str(e)})

    def generate_action(self, telemetry_data):
        """Generate an action based on telemetry data."""
        # Example logic: Create a dummy action
        action = self.fast_controller.decide_action(telemetry_data)
        return action
# Example usage
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