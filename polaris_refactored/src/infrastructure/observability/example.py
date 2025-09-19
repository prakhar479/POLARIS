"""
Example usage of POLARIS Observability Framework

This module demonstrates how to use the logging, metrics, and tracing
components together in a typical POLARIS workflow.
"""

import asyncio
import time
from typing import Dict, Any

from .logging import get_logger, configure_default_logging, LogLevel
from .metrics import get_metrics_collector
from .tracing import get_tracer, configure_tracing, trace_polaris_method


# Configure observability
configure_default_logging(level=LogLevel.INFO, use_json=True)
configure_tracing(service_name="polaris-example", console_export=True)

# Get instances
logger = get_logger("polaris.example")
metrics = get_metrics_collector()
tracer = get_tracer()


class ExampleAdaptationService:
    """Example service demonstrating observability integration"""
    
    def __init__(self):
        self.logger = get_logger("polaris.adaptation")
        self.metrics = get_metrics_collector()
        self.tracer = get_tracer()
    
    @trace_polaris_method("process_telemetry", tags={"component": "adaptation"})
    async def process_telemetry(self, system_id: str, telemetry_data: Dict[str, Any]) -> None:
        """Process incoming telemetry with full observability"""
        
        # Use correlation context for request tracking
        with self.logger.correlation_context() as correlation_id:
            self.logger.info(
                f"Processing telemetry for system {system_id}",
                extra={"system_id": system_id, "data_size": len(telemetry_data)}
            )
            
            # Time the telemetry processing
            with self.metrics.time_telemetry_processing(system_id):
                # Simulate processing
                await asyncio.sleep(0.1)
                
                # Update metrics
                self.metrics.get_metric("polaris_telemetry_events_received_total").increment(
                    labels={"system_id": system_id, "event_type": "health_check"}
                )
                
                # Check if adaptation is needed
                health_score = telemetry_data.get("health_score", 1.0)
                self.metrics.set_system_health_score(system_id, health_score)
                
                if health_score < 0.8:
                    await self._trigger_adaptation(system_id, "low_health", correlation_id)
    
    @trace_polaris_method("trigger_adaptation", tags={"component": "adaptation"})
    async def _trigger_adaptation(self, system_id: str, reason: str, correlation_id: str) -> None:
        """Trigger system adaptation"""
        
        adaptation_id = f"adapt_{int(time.time())}"
        
        # Use adaptation context
        with self.logger.adaptation_context(adaptation_id, system_id):
            self.logger.info(
                f"Triggering adaptation for {system_id}",
                extra={"reason": reason, "adaptation_id": adaptation_id}
            )
            
            # Trace the complete adaptation flow
            with self.tracer.trace_adaptation_flow(adaptation_id, system_id) as span:
                try:
                    # Time the adaptation
                    with self.metrics.time_adaptation(system_id, "health_recovery"):
                        # Simulate adaptation work
                        span.add_event("adaptation_planning_started")
                        await asyncio.sleep(0.05)
                        
                        span.add_event("adaptation_execution_started")
                        await asyncio.sleep(0.1)
                        
                        span.add_event("adaptation_verification_started")
                        await asyncio.sleep(0.03)
                    
                    # Record successful adaptation
                    self.metrics.increment_adaptations_successful(system_id, "health_recovery")
                    self.logger.info(
                        f"Adaptation {adaptation_id} completed successfully",
                        extra={"duration_ms": span.duration_ms}
                    )
                    
                except Exception as e:
                    # Record failed adaptation
                    self.metrics.increment_adaptations_failed(system_id, "health_recovery", str(e))
                    self.logger.error(
                        f"Adaptation {adaptation_id} failed",
                        extra={"error": str(e)},
                        exc_info=e
                    )
                    raise


async def run_example():
    """Run the observability example"""
    
    print("=== POLARIS Observability Framework Example ===\n")
    
    # Create service
    service = ExampleAdaptationService()
    
    # Simulate telemetry processing
    systems = ["web-server-1", "database-1", "cache-1"]
    
    for system_id in systems:
        # Simulate different health scenarios
        if system_id == "database-1":
            telemetry = {"health_score": 0.7, "cpu_usage": 85, "memory_usage": 90}
        else:
            telemetry = {"health_score": 0.95, "cpu_usage": 45, "memory_usage": 60}
        
        await service.process_telemetry(system_id, telemetry)
        await asyncio.sleep(0.1)
    
    # Export metrics in Prometheus format
    print("\n=== Prometheus Metrics Export ===")
    from .metrics import PrometheusExporter
    exporter = PrometheusExporter()
    metrics_output = exporter.export(metrics.get_all_metrics())
    print(metrics_output)
    
    # Flush tracing data
    tracer.flush()
    
    print("\n=== Example completed ===")


if __name__ == "__main__":
    asyncio.run(run_example())