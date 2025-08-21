"""
Generic Monitor Adapter for POLARIS Framework.

This adapter collects metrics from any managed system using the plugin
architecture and publishes telemetry events to NATS.
"""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from polaris.adapters.base import BaseAdapter
from polaris.models.telemetry import TelemetryEvent, TelemetryBatch
from polaris.common.utils import safe_eval
from polaris.knowledge_base.models import KBEntry, KBDataType
from polaris.models.knowledge_base_impl import InMemoryKnowledgeBase


class MonitorAdapter(BaseAdapter):
    """Generic monitor adapter that collects metrics from any managed system.
    
    This adapter provides a plugin-driven approach to metric collection, supporting:
    - Configurable metric collection from any managed system
    - Derived metric calculations based on collected data
    - Batch and streaming telemetry publishing to NATS
    - Flexible collection strategies (parallel/sequential)
    - Comprehensive error handling and retry logic
    
    The adapter uses the plugin configuration to determine:
    - Which metrics to collect and how often
    - How to connect to the managed system
    - What derived metrics to calculate
    - How to handle collection errors
    
    Example:
        adapter = MonitorAdapter(
            polaris_config_path="config/polaris.yaml",
            plugin_dir="plugins/my_system"
        )
        
        async with adapter:
            # Adapter runs continuously until stopped
            await asyncio.sleep(60)
    """

    def __init__(
        self,
        polaris_config_path: str,
        plugin_dir: str,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the monitor adapter.
        
        Args:
            polaris_config_path: Path to POLARIS framework configuration
            plugin_dir: Directory containing the managed system plugin
            logger: Logger instance (created if not provided)
        """
        super().__init__(polaris_config_path, plugin_dir, logger)
        
        # Get monitoring configuration from plugin
        self.monitoring_config = self.config_manager.get_monitoring_config()
        
        # Extract configuration values
        self.monitor_interval = self.monitoring_config.get("interval", 5.0)
        self.metrics_config = self.monitoring_config.get("metrics", [])
        self.derived_metrics_config = self.monitoring_config.get("derived_metrics", [])
        self.strategies = self.monitoring_config.get("strategies", {})
        
        # Telemetry configuration from framework
        telemetry_config = self.framework_config.get("telemetry", {})
        self.telemetry_stream_subject = telemetry_config.get(
            "stream_subject", "polaris.telemetry.events.stream"
        )
        self.telemetry_batch_subject = telemetry_config.get(
            "batch_subject", "polaris.telemetry.events.batch"
        )
        self.telemetry_batch_size = telemetry_config.get("batch_size", 100)
        self.telemetry_batch_max_wait = telemetry_config.get("batch_max_wait", 1.0)
        self.telemetry_stream = telemetry_config.get("stream", False)
        self.queue_maxsize = telemetry_config.get("queue_maxsize", 5000)
        
        # Collection strategies
        self.batch_collection = self.strategies.get("batch_collection", True)
        self.parallel_collection = self.strategies.get("parallel_collection", False)
        self.max_concurrent = self.strategies.get("max_concurrent", 1)
        self.error_handling = self.strategies.get("error_handling", "skip")
        
        # Runtime state
        self.telemetry_queue: asyncio.Queue[TelemetryEvent] = asyncio.Queue(
            maxsize=self.queue_maxsize if self.queue_maxsize > 0 else 0
        )
        self.collect_task: Optional[asyncio.Task] = None
        self.publish_task: Optional[asyncio.Task] = None
        self._immediate_publish_tasks: List[asyncio.Task] = []
        
        # Knowledge base for telemetry storage
        kb_config = telemetry_config.get("knowledge_base", {})
        if kb_config.get("enabled", False):
            buffer_size = kb_config.get("buffer_size", 100)
            self.knowledge_base = InMemoryKnowledgeBase(
                logger=self.logger, 
                telemetry_buffer_size=buffer_size
            )
        else:
            self.knowledge_base = None
        
        # Build metric name to unit mapping
        self.metric_units = {}
        for metric in self.metrics_config:
            self.metric_units[metric["name"]] = metric.get("unit", "unknown")
        for derived in self.derived_metrics_config:
            self.metric_units[derived["name"]] = derived.get("unit", "unknown")
        
        self.logger.info(
            "Monitor adapter initialized",
            extra={
                "system_name": self.plugin_config.get("system_name"),
                "metrics_count": len(self.metrics_config),
                "derived_metrics_count": len(self.derived_metrics_config),
                "monitor_interval": self.monitor_interval,
                "batch_collection": self.batch_collection,
                "parallel_collection": self.parallel_collection
            }
        )

    async def _query_metric(self, metric_config: Dict[str, Any], cycle_ctx: Dict[str, Any]) -> Optional[TelemetryEvent]:
        """Query a single metric from the managed system.
        
        Args:
            metric_config: Metric configuration from plugin
            cycle_ctx: Context for this collection cycle
            
        Returns:
            TelemetryEvent if successful, None otherwise
        """
        metric_name = metric_config["name"]
        command = metric_config["command"]
        unit = metric_config.get("unit", "unknown")
        metric_type = metric_config.get("type", "float")
        
        try:
            response = await self.connector.execute_command(command)
            
            # Parse response based on type
            if metric_type == "float":
                value = float(response)
            elif metric_type == "integer":
                value = int(response)
            elif metric_type == "boolean":
                value = response.lower() in ("true", "1", "yes", "on")
            else:  # string
                value = response
            
            # Create telemetry event
            event = TelemetryEvent(
                name=f"{self.plugin_config['system_name']}.{metric_name}",
                value=value,
                unit=unit,
                source=f"{self.plugin_config['system_name']}_monitor",
                tags={
                    "system": self.plugin_config["system_name"],
                    "category": metric_config.get("category", "unknown")
                }
            )
            
            self.logger.debug(
                "Metric collected",
                extra={
                    "metric": metric_name,
                    "value": value,
                    "unit": unit,
                    **cycle_ctx
                }
            )
            
            return event
            
        except Exception as e:
            if self.error_handling == "fail":
                raise
            elif self.error_handling == "default":
                # Return a default value event
                default_value = 0 if metric_type in ("float", "integer") else False if metric_type == "boolean" else ""
                return TelemetryEvent(
                    name=f"{self.plugin_config['system_name']}.{metric_name}",
                    value=default_value,
                    unit=unit,
                    source=f"{self.plugin_config['system_name']}_monitor",
                    tags={
                        "system": self.plugin_config["system_name"],
                        "category": metric_config.get("category", "unknown"),
                        "error": "default_value_used"
                    }
                )
            else:  # skip
                self.logger.warning(
                    "Metric collection failed",
                    extra={
                        "metric": metric_name,
                        "command": command,
                        "error": str(e),
                        **cycle_ctx
                    }
                )
                return None
    
    def _calculate_derived_metrics(self, base_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate derived metrics from base metrics.
        
        Args:
            base_metrics: Dictionary of base metric name -> value
            
        Returns:
            Dictionary of derived metric name -> value
        """
        derived = {}
        
        for derived_config in self.derived_metrics_config:
            name = derived_config["name"]
            formula = derived_config["formula"]
            
            try:
                result = safe_eval(formula, base_metrics)
                derived[name] = float(result)
                
                self.logger.debug(
                    "Derived metric calculated",
                    extra={
                        "metric": name,
                        "formula": formula,
                        "value": result
                    }
                )
                
            except Exception as e:
                self.logger.warning(
                    f"Derived metric calculation failed (metric: {name}, error: {str(e)})",
                    extra={
                        "metric": name,
                        "formula": formula,
                        "error": str(e)
                    }
                )
        
        return derived
    
    async def _publish_event(self, event: TelemetryEvent) -> None:
        """Publish a single telemetry event immediately."""
        try:
            await self.nats_client.publish_json(
                self.telemetry_stream_subject,
                event.to_dict()
            )
            self.logger.debug(
                "Telemetry event published",
                extra={
                    "metric": event.name,
                    "subject": self.telemetry_stream_subject
                }
            )
        except Exception as e:
            self.logger.warning(
                "Telemetry event publish failed",
                extra={
                    "metric": event.name,
                    "error": str(e)
                }
            )

    def _store_in_knowledge_base(self, event: TelemetryEvent) -> None:
        """Store telemetry event in knowledge base if enabled."""
        if not self.knowledge_base:
            return
            
        try:
            kb_entry = KBEntry(
                data_type=KBDataType.RAW_TELEMETRY_EVENT,
                metric_name=event.name,
                metric_value=event.value,
                source=event.source,
                summary=f"Telemetry: {event.name} = {event.value} {event.unit}",
                content=event.to_dict(),
                tags=["telemetry", event.source] + list(event.tags.values()) if event.tags else ["telemetry", event.source]
            )
            self.knowledge_base.store(kb_entry)
        except Exception as e:
            self.logger.warning(
                "Failed to store telemetry in knowledge base",
                extra={"metric": event.name, "error": str(e)}
            )

    async def _collect_once(self) -> int:
        """Collect metrics once and enqueue TelemetryEvents.
        
        Returns:
            Number of events enqueued
        """
        cycle_id = str(uuid.uuid4())
        cycle_ctx = {"cycle_id": cycle_id}
        start_time = time.perf_counter()
        
        events = []
        base_metrics = {}
        
        # Collect base metrics
        if self.parallel_collection and len(self.metrics_config) > 1:
            # Parallel collection with concurrency limit
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async def _collect_metric(metric_config):
                async with semaphore:
                    return await self._query_metric(metric_config, cycle_ctx)
            
            tasks = [
                asyncio.create_task(_collect_metric(metric_config))
                for metric_config in self.metrics_config
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            results_with_configs = list(zip(self.metrics_config, results))
            
            for metric_config, result in results_with_configs:
                if isinstance(result, Exception):
                    self.logger.warning(
                        "Metric collection task failed",
                        extra={"error": str(result), **cycle_ctx}
                    )
                elif result is not None:
                    events.append(result)
                    metric_name = metric_config["name"]
                    base_metrics[metric_name] = result.value
        else:
            # Sequential collection
            for metric_config in self.metrics_config:
                event = await self._query_metric(metric_config, cycle_ctx)
                if event is not None:
                    events.append(event)
                    metric_name = metric_config["name"]
                    base_metrics[metric_name] = event.value
        
        # Calculate derived metrics
        derived_metrics = self._calculate_derived_metrics(base_metrics)
        
        # Create events for derived metrics
        for name, value in derived_metrics.items():
            unit = self.metric_units.get(name, "unknown")
            event = TelemetryEvent(
                name=f"{self.plugin_config['system_name']}.{name}",
                value=value,
                unit=unit,
                source=f"{self.plugin_config['system_name']}_monitor",
                tags={
                    "system": self.plugin_config["system_name"],
                    "category": "derived"
                }
            )
            events.append(event)
        
        # Enqueue events and optionally publish immediately
        enqueued = 0
        for event in events:
            try:
                self.telemetry_queue.put_nowait(event)
                enqueued += 1
                
                # Store in knowledge base if enabled
                self._store_in_knowledge_base(event)
                
                # Immediate publish if streaming enabled
                if self.telemetry_stream:
                    task = asyncio.create_task(self._publish_event(event))
                    self._immediate_publish_tasks.append(task)
                    
                    # Clean up completed tasks
                    def _on_done(t: asyncio.Task):
                        try:
                            self._immediate_publish_tasks.remove(t)
                        except ValueError:
                            pass
                    task.add_done_callback(_on_done)
                    
            except asyncio.QueueFull:
                self.logger.error(
                    "Telemetry queue full",
                    extra={
                        "dropped_metric": event.name,
                        **cycle_ctx
                    }
                )
                break
        
        elapsed = time.perf_counter() - start_time
        self.logger.info(
            "Collection cycle completed",
            extra={
                "cycle_id": cycle_id,
                "metrics_collected": len(base_metrics),
                "derived_metrics": len(derived_metrics),
                "events_enqueued": enqueued,
                "elapsed_ms": round(elapsed * 1000, 3),
                "queue_size": self.telemetry_queue.qsize()
            }
        )
        
        return enqueued

    async def _collector_loop(self):
        """Main collection loop that runs continuously."""
        self.logger.info("Collector started")
        try:
            while self.running:
                try:
                    await self._collect_once()
                except Exception as e:
                    self.logger.exception(
                        "Collection cycle error",
                        extra={"error": str(e)}
                    )
                
                await asyncio.sleep(self.monitor_interval)
        except asyncio.CancelledError:
            self.logger.info("Collector cancelled")
        finally:
            self.logger.info("Collector stopped")
    
    async def _flush_batch(self, batch: List[TelemetryEvent]) -> None:
        """Publish a batch of telemetry events to NATS.
        
        Args:
            batch: List of telemetry events to publish
        """
        if not batch:
            return
        
        # Create batch payload
        batch_data = TelemetryBatch(
            events=batch
        )
        
        try:
            await self.nats_client.publish_json(
                self.telemetry_batch_subject,
                batch_data.model_dump()
            )
            
            self.logger.info(
                "Telemetry batch published",
                extra={
                    "count": len(batch),
                    "subject": self.telemetry_batch_subject,
                    "queue_size_after": self.telemetry_queue.qsize()
                }
            )
            
        except Exception as e:
            self.logger.warning(
                "Telemetry batch publish failed, trying individual events",
                extra={"error": str(e)}
            )
            
            # Fallback: publish events individually
            published = 0
            for event in batch:
                try:
                    await self.nats_client.publish_json(
                        self.telemetry_stream_subject,
                        event.to_dict()
                    )
                    published += 1
                except Exception as ex:
                    self.logger.error(
                        "Individual telemetry event publish failed",
                        extra={
                            "event": event.name,
                            "error": str(ex)
                        }
                    )
            
            self.logger.info(
                "Fallback publish completed",
                extra={
                    "published": published,
                    "attempted": len(batch)
                }
            )
    
    async def _publisher_loop(self):
        """Main publisher loop that batches and publishes telemetry events."""
        self.logger.info(
            "Publisher started",
            extra={
                "batch_size": self.telemetry_batch_size,
                "batch_max_wait": self.telemetry_batch_max_wait
            }
        )
        
        try:
            while self.running:
                batch: List[TelemetryEvent] = []
                
                # Wait for at least one event
                try:
                    first_event = await asyncio.wait_for(
                        self.telemetry_queue.get(),
                        timeout=self.telemetry_batch_max_wait
                    )
                    batch.append(first_event)
                except asyncio.TimeoutError:
                    continue  # No events available, try again
                
                # Collect additional events up to batch size
                batch_start_time = time.perf_counter()
                while len(batch) < self.telemetry_batch_size:
                    remaining_time = self.telemetry_batch_max_wait - (
                        time.perf_counter() - batch_start_time
                    )
                    
                    if remaining_time <= 0:
                        break
                    
                    try:
                        next_event = await asyncio.wait_for(
                            self.telemetry_queue.get(),
                            timeout=max(0.0, remaining_time)
                        )
                        batch.append(next_event)
                    except asyncio.TimeoutError:
                        break
                
                # Publish the batch
                try:
                    await self._flush_batch(batch)
                finally:
                    # Mark all events as processed
                    for _ in batch:
                        self.telemetry_queue.task_done()
        
        except asyncio.CancelledError:
            self.logger.info("Publisher cancelled")
        finally:
            self.logger.info("Publisher stopped")

    async def _start_processing(self) -> None:
        """Start monitoring-specific processing."""
        # Start collector and publisher tasks
        self.collect_task = asyncio.create_task(self._collector_loop())
        self.publish_task = asyncio.create_task(self._publisher_loop())
        
        # Track tasks for cleanup
        self._tasks.extend([self.collect_task, self.publish_task])
        
        self.logger.info(
            "Monitor processing started",
            extra={
                "telemetry_stream": self.telemetry_stream,
                "batch_collection": self.batch_collection
            }
        )
    
    async def _stop_processing(self) -> None:
        """Stop monitoring-specific processing."""
        # Cancel collector first (stops new events)
        if self.collect_task and not self.collect_task.done():
            self.collect_task.cancel()
        
        # Drain the queue with timeout
        try:
            await asyncio.wait_for(
                self.telemetry_queue.join(),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            self.logger.warning(
                "Telemetry queue drain timeout",
                extra={"remaining": self.telemetry_queue.qsize()}
            )
        
        # Cancel publisher
        if self.publish_task and not self.publish_task.done():
            self.publish_task.cancel()
        
        # Clean up immediate publish tasks
        if self._immediate_publish_tasks:
            self.logger.info(
                "Waiting for immediate publish tasks",
                extra={"count": len(self._immediate_publish_tasks)}
            )
            
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._immediate_publish_tasks, return_exceptions=True),
                    timeout=2.0
                )
            except asyncio.TimeoutError:
                # Cancel remaining tasks
                for task in self._immediate_publish_tasks:
                    if not task.done():
                        task.cancel()
            
            self._immediate_publish_tasks.clear()
        
        self.logger.info("Monitor processing stopped")
