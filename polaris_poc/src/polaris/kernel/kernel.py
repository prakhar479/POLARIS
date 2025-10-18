import asyncio
import time
import json
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from collections import deque
from polaris.common.nats_client import NATSClient
from polaris.controllers.fast_controller import FastController
from polaris.controllers.slow_controller import SlowController
from polaris.controllers.controller_strategy import ControllerStrategy
from time import sleep

current_servers = -1


# BaseKernel class
class BaseKernel(ABC):
    def __init__(self, nats_url: str, logger: logging.Logger, name: str):
        self.nats_client = NATSClient(nats_url=nats_url, logger=logger, name=name)
        self.logger = logger
        self.running = False
        self.enable_verification = True
        self.verification_timeout = 30.0  # Configurable timeout

    async def start(self):
        """Start the kernel."""
        self.logger.info(f"Starting {self.__class__.__name__}")
        await self.nats_client.connect()
        self.running = True

        await self.nats_client.subscribe(
            "polaris.telemetry.events.batch", self.process_telemetry_event
        )
        await self.nats_client.subscribe(
            "polaris.telemetry.events.query", self.process_telemetry_event_stream
        )
        self.logger.info("Subscribed to telemetry streams.")

    async def stop(self):
        self.logger.info(f"Stopping {self.__class__.__name__}")
        self.running = False
        await self.nats_client.close()

    @abstractmethod
    async def process_telemetry_event(self, msg):
        pass

    @abstractmethod
    async def process_telemetry_event_stream(self, msg):
        pass

    @abstractmethod
    def generate_action(self, telemetry_data):
        pass

    async def send_action_for_verification(
        self, action: Dict[str, Any], context: Dict[str, Any]
    ) -> str:
        """Send action for verification."""
        request_id = str(uuid.uuid4())

        verification_request = {
            "request_id": request_id,
            "action": action,
            "context": context,
            "verification_level": "basic",
            "timeout_sec": self.verification_timeout,
            "requester": self.__class__.__name__,
        }

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

    async def execute_action_with_verification(
        self, action: Dict[str, Any], context: Dict[str, Any] = None
    ):
        """Execute an action with optional verification."""
        if self.enable_verification:
            await self.send_action_for_verification(action, context or {})
        else:
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
        self.strategy = ControllerStrategy(logger)
        self.fast_controller = FastController(logger)
        self.slow_controller = SlowController()

        self.controller = None  # Current controller
        # Maintain a rolling buffer of last 5 telemetry streams
        self.stream_buffer = deque(maxlen=1)
        self.batch_size = 1

        self.action_cooldown_sec = 60  # The desired cool-down period for fast controller
        self.slow_action_cooldown_sec = 30  # The desired cool-down period for slow controller
        self.last_action_time = None
        self.last_slow_action_time = None

    async def process_telemetry_event(self, msg):
        try:
            telemetry_data = json.loads(msg.data.decode())
            self.logger.debug(f"Telemetry data received: {telemetry_data}")
            self.logger.info(f"Selected controller: {self.controller.__class__.__name__}")

            if isinstance(self.controller, SlowController):
                await self.nats_client.publish(
                    "polaris.reasoner.kernel.requests",
                    json.dumps(telemetry_data).encode(),
                )
                self.logger.info(
                    "Delegated telemetry to Reasoner via polaris.reasoner.kernel.requests"
                )
            else:
                # Placeholder for normal fast path
                pass

        except Exception as e:
            self.logger.error(f"Error processing telemetry event batch: {str(e)}")

    async def process_telemetry_event_stream(self, msg):
        try:
            telemetry_data = json.loads(msg.data.decode())

            self.logger.info(f"Streaming telemetry data received: {telemetry_data}")

            # 1. Non-blocking check for cooldown

            # --- Continue processing only if cooldown has expired or no action has been sent ---

            self.stream_buffer.append(telemetry_data)

            # Wait until we have enough samples
            if len(self.stream_buffer) < self.batch_size:
                return

            # Compute average response time over last 5 snapshots
            avg_resp_time = sum(
                d.get("current_state", {}).get("average_response_time", 0)
                for d in self.stream_buffer
            ) / len(self.stream_buffer)

            self.logger.info(
                f"Avg response time (last {self.batch_size} streams): {avg_resp_time:.3f}"
            )

            # Decide which controller to use based on average
            if avg_resp_time > 1:
                # Switch to FastController

                current_time = time.time()
                if (
                    self.last_action_time is not None
                    and (current_time - self.last_action_time) < self.action_cooldown_sec
                ):
                    self.logger.info(
                        f"Action skipped due to cooldown. "
                        f"Next action allowed in {self.action_cooldown_sec - (current_time - self.last_action_time):.1f}s"
                    )
                    return

                self.controller = self.fast_controller
                self.logger.info("Switched to FastController")

                latest_data = self.stream_buffer[-1]

                # Check server condition before generating action
                active_servers = latest_data.get("current_state", {}).get("active_servers")
                total_servers = latest_data.get("current_state", {}).get("servers")

                if active_servers != total_servers:
                    self.logger.info("Active servers != Total servers, skipping action generation.")
                    return

                # --- Action Generation and Execution ---
                action = self.generate_action(latest_data)

                if action:
                    # Handle both single actions and lists of actions
                    actions_to_publish = action if isinstance(action, list) else [action]

                    for single_action in actions_to_publish:
                        await self.nats_client.publish_json(
                            "polaris.execution.actions", single_action
                        )
                        enriched_action = single_action.copy()
                        enriched_action["timestamp"] = time.time()
                        await self.nats_client.publish_json(
                            "polaris.execution.fast", enriched_action
                        )

                    # Update last action time to enforce the cool-down
                    self.last_action_time = current_time

                    # This replaces the blocking asyncio.sleep(60)
                    action_count = len(actions_to_publish)
                    self.logger.info(
                        f"Published {action_count} action(s) after 5-stream evaluation. "
                        f"Cooldown of {self.action_cooldown_sec}s initiated."
                    )
                else:
                    self.logger.warning("No action generated after 5-stream evaluation.")
            else:
                # Switch to SlowController (Delegates to Reasoner)
                current_time = time.time()
                if (
                    self.last_slow_action_time is not None
                    and (current_time - self.last_slow_action_time) < self.slow_action_cooldown_sec
                ):
                    self.logger.info(
                        f"Slow action skipped due to cooldown. "
                        f"Next slow action allowed in {self.slow_action_cooldown_sec - (current_time - self.last_slow_action_time):.1f}s"
                    )
                    return

                self.controller = self.slow_controller
                self.logger.info("Switched to SlowController")

                # Delegate the latest data to Reasoner
                latest_data = self.stream_buffer[-1]
                await self.nats_client.publish(
                    "polaris.reasoner.kernel.requests",
                    json.dumps(latest_data).encode(),
                )
                self.logger.info("Delegated streaming telemetry to Reasoner")

                # Update last slow action time to enforce the cool-down
                self.last_slow_action_time = current_time
                self.logger.info(
                    f"Slow controller action delegated. "
                    f"Cooldown of {self.slow_action_cooldown_sec}s initiated."
                )

        except Exception as e:
            self.logger.error(f"Error processing streaming telemetry event {e}")

    def generate_action(self, telemetry_data):
        if self.controller:
            return self.controller.decide_action(telemetry_data)
        return None

    def _extract_context_from_telemetry(self, telemetry_data) -> Dict[str, Any]:
        """Extract relevant context from telemetry for verification."""
        context = {}
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
