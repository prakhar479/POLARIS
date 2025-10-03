import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
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
        self.enable_verification = True  # Can be configured
        self.verification_timeout = 30.0  # Configurable timeout
        # Note: No longer tracking pending verifications - verification adapter handles execution directly

    async def start(self):
        """Start the kernel."""
        self.logger.info(f"Starting {self.__class__.__name__}")
        await self.nats_client.connect()
        self.running = True

        await self.nats_client.subscribe(
            "polaris.telemetry.events.batch", self.process_telemetry_event
        )
        self.logger.info(f"Subscribed to 'polaris.telemetry.events.batch'")

        # Note: Verification results are now handled directly by verification adapter
        # No need to subscribe to verification results

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

    async def send_action_for_verification(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        """Send action for verification (fire-and-forget - verification adapter handles execution)."""
        request_id = str(uuid.uuid4())

        verification_request = {
            "request_id": request_id,
            "action": action,
            "context": context,
            "verification_level": "basic",  # Can be configured
            "timeout_sec": self.verification_timeout,
            "requester": self.__class__.__name__,
        }

        # Send verification request (no need to track - verification adapter handles execution)
        await self.nats_client.publish_json("polaris.verification.requests", verification_request)

        self.logger.info(
            "Action sent for verification",
            extra={
                "request_id": request_id,
                "action_type": action.get("action_type"),
                "action_id": action.get("action_id"),
            },
        )

        return request_id

    # Note: handle_verification_result method removed - verification adapter now handles execution directly

    async def execute_action_with_verification(
        self, action: Dict[str, Any], context: Dict[str, Any] = None
    ):
        """Execute an action with optional verification (optimized flow)."""
        if self.enable_verification:
            # Send for verification - verification adapter will handle execution if approved
            await self.send_action_for_verification(action, context or {})
        else:
            # Send directly to execution
            await self.nats_client.publish_json("polaris.execution.actions", action)
            self.logger.info(
                "Action sent directly for execution (verification disabled)",
                extra={
                    "action_type": action.get("action_type"),
                    "action_id": action.get("action_id"),
                },
            )


# SWIMKernel class
class SWIMKernel(BaseKernel):
    def __init__(self, nats_url: str, logger: logging.Logger):
        super().__init__(nats_url=nats_url, logger=logger, name="SWIMKernel")
        self.strategy = ControllerStrategy()
        self.controller = None  # Will be set by strategy

    async def process_telemetry_event(self, msg):
        try:
            telemetry_data = json.loads(msg.data.decode())

            controller = self.strategy.select_controller(telemetry_data)
            self.logger.info(f"Selected controller: {controller.__class__.__name__}")

            if isinstance(controller, SlowController):
                # Send telemetry to reasoner instead of generating action here
                await self.nats_client.publish(
                    "polaris.reasoner.kernel.requests", json.dumps(telemetry_data).encode()
                )
                self.logger.info(
                    "Delegated telemetry to Reasoner via polaris.reasoner.kernel.requests"
                )
            else:
                # Normal fast path
                action = self.generate_action(telemetry_data)
                if action is None:
                    self.logger.warning("No action generated from telemetry data")
                    return

                # Extract context for verification
                context = self._extract_context_from_telemetry(telemetry_data)

                # Use verification-aware execution
                await self.execute_action_with_verification(action, context)

        except Exception as e:
            self.logger.error("Error processing telemetry event", extra={"error": str(e)})

    def generate_action(self, telemetry_data):
        if self.controller:
            return self.controller.decide_action(telemetry_data)
        return None

    def _extract_context_from_telemetry(self, telemetry_data) -> Dict[str, Any]:
        """Extract relevant context from telemetry for verification."""
        context = {}

        # Extract current system state from telemetry
        events = telemetry_data.get("events", [])

        for event in events:
            name = event.get("name", "")
            value = event.get("value")

            if "active.servers" in name:
                context["active_servers"] = value
            elif "max.servers" in name:
                context["max_servers"] = value
            elif "server.utilization" in name:
                context["utilization"] = value
            elif "response.time" in name:
                context["response_time"] = value
            elif "dimmer" in name:
                context["dimmer"] = value

        return context


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
