"""
Knowledge Base Service for POLARIS Framework.

This service runs continuously, subscribing to NATS telemetry events and
maintaining a shared knowledge base that can be queried by other components.
"""

import asyncio
import json
import logging
import signal
import uuid
import time
from typing import Optional, Dict, Any
from pathlib import Path

from polaris.knowledge_base.models import KBEntry, KBDataType, KBQuery, KBResponse
from polaris.models.knowledge_base_impl import InMemoryKnowledgeBase
from polaris.common.logging_setup import setup_logging

# Import NATS directly
try:
    from nats.aio.client import Client as NATS
except ImportError:
    print("Error: nats-py package required. Install with: pip install nats-py")
    raise


class KnowledgeBaseService:
    """
    Standalone Knowledge Base Service.

    This service:
    - Runs continuously as a daemon
    - Subscribes to NATS telemetry streams
    - Maintains a shared knowledge base
    - Provides query capabilities via NATS requests
    - Handles graceful shutdown
    """

    def __init__(
        self,
        nats_url: str = "nats://localhost:4222",
        telemetry_buffer_size: int = 50,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize the Knowledge Base Service.

        Args:
            nats_url: NATS server URL
            telemetry_buffer_size: Size of telemetry buffers before aggregation
            logger: Logger instance
        """
        self.nats_url = nats_url
        self.logger = logger or logging.getLogger(__name__)

        # Knowledge base instance
        self.kb = InMemoryKnowledgeBase(
            logger=self.logger, telemetry_buffer_size=telemetry_buffer_size
        )

        # NATS client
        self.nats_client: Optional[NATS] = None
        self.running = False

        # Statistics
        self.stats = {
            "start_time": time.time(),
            "events_processed": 0,
            "queries_served": 0,
            "last_event_time": None,
            "last_query_time": None,
        }

    async def start(self):
        """Start the Knowledge Base Service."""
        self.logger.info("🚀 Starting Knowledge Base Service")

        try:
            # Connect to NATS
            await self._connect_nats()

            # Subscribe to telemetry streams
            await self._subscribe_telemetry()

            # Subscribe to query requests
            await self._subscribe_queries()

            self.running = True
            self.logger.info("✅ Knowledge Base Service started successfully")

            # Run until shutdown
            await self._run_service()

        except Exception as e:
            self.logger.error(f"❌ Failed to start Knowledge Base Service: {e}")
            raise
        finally:
            await self._cleanup()

    async def _connect_nats(self):
        """Connect to NATS server."""
        self.nats_client = NATS()
        await self.nats_client.connect(self.nats_url)
        self.logger.info(f"✅ Connected to NATS at {self.nats_url}")

    async def _subscribe_telemetry(self):
        """Subscribe to telemetry streams."""
        # Subscribe to individual telemetry events
        await self.nats_client.subscribe(
            "polaris.telemetry.events.stream", cb=self._handle_telemetry_event
        )

        await self.nats_client.subscribe(
            "polaris.telemetry.events.snapshots", cb=self._handle_telemetry_snapshots
        )

        # Subscribe to batch telemetry events
        await self.nats_client.subscribe(
            "polaris.telemetry.events.batch", cb=self._handle_telemetry_batch
        )

        self.logger.info("📡 Subscribed to telemetry streams")

        # Subscribe to adaptation decisions
        await self.nats_client.subscribe(
            "polaris.execution.decisions", cb=self._handle_adaptation_decision
        )

        self.logger.info("🎯 Subscribed to adaptation decisions")

    async def _subscribe_queries(self):
        """Subscribe to query requests."""
        await self.nats_client.subscribe("polaris.knowledge.query", cb=self._handle_query_request)

        await self.nats_client.subscribe("polaris.knowledge.stats", cb=self._handle_stats_request)

        self.logger.info("🔍 Subscribed to query requests")

    async def _handle_telemetry_event(self, msg):
        """Handle individual telemetry event from NATS."""
        try:
            data = json.loads(msg.data.decode())

            # Create KBEntry for telemetry event
            entry = KBEntry(
                data_type=KBDataType.RAW_TELEMETRY_EVENT,
                metric_name=data.get("name"),
                metric_value=data.get("value"),
                source=data.get("source"),
                summary=f"Telemetry: {data.get('name')} = {data.get('value')} {data.get('unit', '')}",
                content=data,
                tags=["telemetry", data.get("source", "unknown")],
            )

            self.kb.store(entry)
            self.stats["events_processed"] += 1
            self.stats["last_event_time"] = time.time()

            if self.stats["events_processed"] % 100 == 0:
                self.logger.info(f"📊 Processed {self.stats['events_processed']} telemetry events")

        except Exception as e:
            self.logger.error(f"❌ Error processing telemetry event: {e}")

    async def _handle_telemetry_snapshots(self, msg):
        """Handle telemetry snapshot events from NATS."""
        try:
            snapshot_payload = json.loads(msg.data.decode())

            # Extract utilization (fallback to 0.0)
            current_util = snapshot_payload.get("current_state", {}).get("server_utilization", 0.0)

            # Generate a unique snapshot ID
            snapshot_id = f"snapshot_{uuid.uuid4()}"

            # Create a summary string
            summary = f"System Snapshot [{snapshot_id}]: Util={current_util}"

            # Build KBEntry from snapshot
            kb_entry = KBEntry(
                entry_id=snapshot_id,
                data_type=KBDataType.OBSERVATION,
                content=snapshot_payload,
                source=snapshot_payload.get("source", "unknown_snapshotter"),
                tags=["snapshot", "llm_context", "permanent"],
                metric_name="monitor.state.snapshot",
                metric_value=current_util,
                summary=summary,
            )

            # Store in knowledge base
            self.kb.store(kb_entry)
            self.stats["events_processed"] += 1
            self.stats["last_event_time"] = time.time()

            self.logger.debug(
                f"📸 Processed snapshot {snapshot_id} with utilization {current_util}"
            )

        except Exception as e:
            self.logger.error(f"❌ Error processing telemetry snapshot: {e}")

    async def _handle_telemetry_batch(self, msg):
        """Handle batch telemetry events from NATS."""
        try:
            data = json.loads(msg.data.decode())

            # Process each event in the batch
            events = data.get("events", [])
            for event_data in events:
                entry = KBEntry(
                    data_type=KBDataType.RAW_TELEMETRY_EVENT,
                    metric_name=event_data.get("name"),
                    metric_value=event_data.get("value"),
                    source=event_data.get("source"),
                    summary=f"Telemetry: {event_data.get('name')} = {event_data.get('value')} {event_data.get('unit', '')}",
                    content=event_data,
                    tags=["telemetry", event_data.get("source", "unknown")],
                )

                self.kb.store(entry)
                self.stats["events_processed"] += 1

            self.stats["last_event_time"] = time.time()

            self.logger.debug(f"📦 Processed batch of {len(events)} telemetry events")

        except Exception as e:
            self.logger.error(f"❌ Error processing telemetry batch: {e}")

    async def _handle_adaptation_decision(self, msg):
        """Handle adaptation decision from execution adapter.

        Stores the executed action and its result in the knowledge base
        for learning and analysis purposes.
        """
        try:
            decision_data = json.loads(msg.data.decode())

            # Create KBEntry from the decision data
            # The data is already in KBEntry format from the execution adapter
            entry = KBEntry(**decision_data)

            # Store in knowledge base
            self.kb.store(entry)
            self.stats["events_processed"] += 1
            self.stats["last_event_time"] = time.time()

            action_type = entry.content.get("action_type", "unknown")
            action_success = entry.content.get("execution_success", False)

            self.logger.info(
                f"🎯 Stored adaptation decision: {action_type} "
                f"({'✅ success' if action_success else '❌ failed'})"
            )

            if self.stats["events_processed"] % 10 == 0:
                self.logger.info(
                    f"📊 Total processed: {self.stats['events_processed']} events "
                    f"(including adaptation decisions)"
                )

        except Exception as e:
            self.logger.error(f"❌ Error processing adaptation decision: {e}", exc_info=True)

    async def _handle_query_request(self, msg):
        """Handle knowledge base query request."""
        try:
            # Parse query from message
            query_data = json.loads(msg.data.decode())
            query = KBQuery(**query_data)
            self.logger.debug(f"🔍 Received query: {query}")
            # Execute query
            response = self.kb.query(query)

            # Update stats
            self.stats["queries_served"] += 1
            self.stats["last_query_time"] = time.time()

            # Send response
            response_data = {
                "query_id": response.query_id,
                "success": response.success,
                "total_results": response.total_results,
                "processing_time_ms": response.processing_time_ms,
                "message": response.message,
                "results": [entry.dict() for entry in response.results],
            }

            await self.nats_client.publish(msg.reply, json.dumps(response_data).encode())

            self.logger.debug(
                f"🔍 Query processed: {response.total_results} results in {response.processing_time_ms:.2f}ms"
            )

        except Exception as e:
            self.logger.error(f"❌ Error processing query: {e}")

            # Send error response
            if msg.reply:
                error_response = {
                    "success": False,
                    "message": str(e),
                    "total_results": 0,
                    "results": [],
                }
                await self.nats_client.publish(msg.reply, json.dumps(error_response).encode())

    async def _handle_stats_request(self, msg):
        """Handle knowledge base statistics request."""
        try:
            # Get KB stats
            kb_stats = self.kb.get_stats()

            # Combine with service stats
            stats_data = {
                "service": self.stats,
                "knowledge_base": kb_stats,
                "uptime_seconds": time.time() - self.stats["start_time"],
            }

            # Send response
            await self.nats_client.publish(msg.reply, json.dumps(stats_data).encode())

            self.logger.debug("📊 Stats request served")

        except Exception as e:
            self.logger.error(f"❌ Error processing stats request: {e}")

    async def _run_service(self):
        """Main service loop."""
        self.logger.info("🔄 Knowledge Base Service running...")

        try:
            # Periodic status logging
            while self.running:
                await asyncio.sleep(30)  # Log status every 30 seconds

                kb_stats = self.kb.get_stats()
                uptime = time.time() - self.stats["start_time"]

                self.logger.info(
                    f"📊 Status: {self.stats['events_processed']} events, "
                    f"{self.stats['queries_served']} queries, "
                    f"{kb_stats['total_permanent_entries']} KB entries, "
                    f"uptime: {uptime:.0f}s"
                )

        except asyncio.CancelledError:
            self.logger.info("🛑 Service loop cancelled")
            raise

    async def shutdown(self):
        """Graceful shutdown."""
        self.logger.info("🛑 Shutting down Knowledge Base Service...")
        self.running = False
        await self._cleanup()

    async def _cleanup(self):
        """Clean up resources."""
        if self.nats_client:
            await self.nats_client.close()
            self.logger.info("🔌 NATS connection closed")


async def main():
    """Main entry point for standalone service."""
    import argparse

    parser = argparse.ArgumentParser(description="POLARIS Knowledge Base Service")
    parser.add_argument(
        "--nats-url",
        default="nats://localhost:4222",
        help="NATS server URL (default: nats://localhost:4222)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=50,
        help="Telemetry buffer size (default: 50)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()
    logger.setLevel(getattr(logging, args.log_level))

    # Create service
    service = KnowledgeBaseService(
        nats_url=args.nats_url, telemetry_buffer_size=args.buffer_size, logger=logger
    )

    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"🔔 Received signal {signum}")
        asyncio.create_task(service.shutdown())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start service
        await service.start()
    except KeyboardInterrupt:
        logger.info("🔔 Keyboard interrupt received")
    except Exception as e:
        logger.error(f"❌ Service failed: {e}")
        raise
    finally:
        logger.info("✅ Knowledge Base Service stopped")


if __name__ == "__main__":
    asyncio.run(main())
