"""
Switch System Monitor Adapter for POLARIS Framework.

This module implements a specialized monitor adapter for the Switch YOLO system,
collecting comprehensive metrics and publishing telemetry events to NATS.
"""

import asyncio
import logging
import time
import uuid
import json
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path to import from polaris
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from polaris.adapters.monitor import MonitorAdapter
from polaris.models.telemetry import TelemetryEvent, TelemetryBatch
from polaris.knowledge_base.models import KBEntry, KBDataType
from switch_connector import SwitchSystemConnector


class SwitchMonitorAdapter(MonitorAdapter):
    """
    Specialized monitor adapter for Switch YOLO system.
    
    This adapter extends the generic MonitorAdapter to provide Switch-specific
    metric collection, including:
    - YOLO model performance metrics (response time, confidence, utility)
    - System resource metrics (CPU, memory, processing stats)
    - Model switching events and state changes
    - Comprehensive telemetry publishing with Switch-specific context
    """
    
    def __init__(
        self, 
        polaris_config_path: str, 
        plugin_dir: str, 
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the Switch monitor adapter.
        
        Args:
            polaris_config_path: Path to POLARIS framework configuration
            plugin_dir: Directory containing the Switch system plugin
            logger: Logger instance (created if not provided)
        """
        super().__init__(polaris_config_path, plugin_dir, logger)
        
        # Switch-specific configuration
        self.switch_config = self.plugin_config.get("switch_system", {})
        
        # Override monitoring interval for Switch system (more frequent)
        self.monitor_interval = self.switch_config.get("monitor_interval", 30.0)
        
        # Switch-specific telemetry subjects
        self.switch_metrics_subject = "polaris.switch.metrics"
        self.switch_model_events_subject = "polaris.switch.model_events"
        self.switch_performance_subject = "polaris.switch.performance"
        self.switch_utility_subject = "polaris.switch.utility"
        
        # Metric collection configuration
        self.collect_performance_metrics = True
        self.collect_system_metrics = True
        self.collect_model_state = True
        self.collect_utility_metrics = True
        
        # Performance tracking
        self.last_model = None
        self.model_switch_count = 0
        self.performance_history = []
        self.max_history_size = 100
        
        # Utility function parameters (from config)
        utility_config = self.switch_config.get("utility", {})
        self.utility_params = {
            "Rmin": utility_config.get("Rmin", 0.1),
            "Rmax": utility_config.get("Rmax", 1.0),
            "Cmin": utility_config.get("Cmin", 0.5),
            "Cmax": utility_config.get("Cmax", 1.0),
            "wd": utility_config.get("wd", 0.5),
            "we": utility_config.get("we", 0.5),
            "pdv": utility_config.get("pdv", 5.0),
            "pev": utility_config.get("pev", 5.0)
        }
        
        self.logger.info(
            "Switch monitor adapter initialized",
            extra={
                "monitor_interval": self.monitor_interval,
                "utility_params": self.utility_params,
                "switch_config": self.switch_config
            }
        )
    
    async def _collect_switch_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive Switch system metrics.
        
        Returns:
            Dictionary containing all collected metrics
        """
        metrics = {}
        
        try:
            # Get current system state
            system_state = await self.connector.get_system_state()
            
            # Extract basic metrics
            current_model = system_state.get("current_model", "unknown")
            raw_metrics = system_state.get("metrics", {})
            logs = system_state.get("logs", {})
            
            # Core performance metrics
            metrics.update({
                "current_model": current_model,
                "image_processing_time": float(raw_metrics.get("image_processing_time", 0.0)),
                "model_processing_time": float(raw_metrics.get("model_processing_time", 0.0)),
                "confidence": float(raw_metrics.get("confidence", 0.0)),
                "utility": float(raw_metrics.get("utility", 0.0)),
                "cpu_usage": float(raw_metrics.get("cpu_usage", 0.0)),
                "detection_boxes": int(raw_metrics.get("detection_boxes", 0)),
                "total_processed": int(raw_metrics.get("total_processed", 0)),
                "timestamp": time.time()
            })
            
            # Model switching tracking
            if self.last_model != current_model:
                if self.last_model is not None:
                    self.model_switch_count += 1
                    await self._publish_model_switch_event(self.last_model, current_model)
                self.last_model = current_model
            
            metrics["model_switch_count"] = self.model_switch_count
            
            # Additional system metrics from logs
            if logs:
                metrics.update({
                    "memory_usage": float(logs.get("memory_usage", 0.0)),
                    "disk_usage": float(logs.get("disk_usage", 0.0)),
                    "network_io": float(logs.get("network_io", 0.0)),
                    "process_count": int(logs.get("process_count", 0))
                })
            
            # Calculate derived metrics
            derived_metrics = await self._calculate_switch_derived_metrics(metrics)
            metrics.update(derived_metrics)
            
            # Update performance history
            self._update_performance_history(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect Switch metrics: {e}")
            return {
                "error": str(e),
                "timestamp": time.time(),
                "current_model": "unknown"
            }
    
    async def _calculate_switch_derived_metrics(self, base_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate Switch-specific derived metrics.
        
        Args:
            base_metrics: Base metrics collected from the system
            
        Returns:
            Dictionary of derived metrics
        """
        derived = {}
        
        try:
            # Utility function calculation (if we have the raw components)
            response_time = base_metrics.get("image_processing_time", 0.0)
            model_processing_time = base_metrics.get("model_processing_time", 0.0)
            confidence = base_metrics.get("confidence", 0.0)
            
            if response_time > 0 and confidence > 0:
                calculated_utility = self._calculate_utility(response_time, confidence)
                derived["calculated_utility"] = calculated_utility
                
                # Utility components
                derived["utility_response_score"] = self._normalize_response_time(response_time)
                derived["utility_confidence_score"] = self._normalize_confidence(confidence)
            
            # Model-specific processing metrics
            if model_processing_time > 0:
                derived["model_efficiency"] = 1.0 / model_processing_time
                
                # Compare model processing time vs total processing time
                if response_time > 0:
                    derived["model_overhead_ratio"] = model_processing_time / response_time
                    derived["non_model_processing_time"] = max(0.0, response_time - model_processing_time)
            
            # Performance efficiency metrics
            cpu_usage = base_metrics.get("cpu_usage", 0.0)
            if cpu_usage > 0 and response_time > 0:
                derived["cpu_efficiency"] = (1.0 / response_time) / (cpu_usage / 100.0)
            
            # Model-specific CPU efficiency
            if cpu_usage > 0 and model_processing_time > 0:
                derived["model_cpu_efficiency"] = (1.0 / model_processing_time) / (cpu_usage / 100.0)
            
            # Detection efficiency
            detection_boxes = base_metrics.get("detection_boxes", 0)
            if detection_boxes > 0 and response_time > 0:
                derived["detection_rate"] = detection_boxes / response_time
            
            # Model performance rating (based on current model characteristics)
            current_model = base_metrics.get("current_model", "unknown")
            derived["model_performance_rating"] = self._get_model_performance_rating(current_model)
            
            # Throughput metrics
            total_processed = base_metrics.get("total_processed", 0)
            if len(self.performance_history) > 1:
                time_diff = base_metrics["timestamp"] - self.performance_history[-1]["timestamp"]
                if time_diff > 0:
                    prev_processed = self.performance_history[-1].get("total_processed", 0)
                    derived["processing_rate"] = (total_processed - prev_processed) / time_diff
            
            # System load indicators
            if cpu_usage > 0:
                derived["system_load_factor"] = min(cpu_usage / 100.0, 1.0)
                derived["resource_pressure"] = max(0.0, (cpu_usage - 70.0) / 30.0)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate derived metrics: {e}")
        
        return derived
    
    def _calculate_utility(self, response_time: float, confidence: float) -> float:
        """Calculate utility using FIXED Switch system utility function.
        
        Args:
            response_time: Image processing time in seconds
            confidence: Detection confidence (0-1)
            
        Returns:
            Calculated utility value (bounded to prevent death spirals)
        """
        try:
            # Normalize response time
            rt_score = self._normalize_response_time(response_time)
            
            # Normalize confidence
            conf_score = self._normalize_confidence(confidence)
            
            # Calculate weighted utility
            base_utility = (self.utility_params["we"] * conf_score + 
                           self.utility_params["wd"] * rt_score)
            
            # FIXED: Apply bounded penalties for out-of-range values
            penalty = 0.0
            
            if response_time > self.utility_params["Rmax"]:
                # Bounded penalty to prevent exponential degradation
                excess_ratio = (response_time - self.utility_params["Rmax"]) / (self.utility_params["Rmax"] + 1e-6)
                penalty += min(self.utility_params["pdv"] * excess_ratio, 2.0)  # Max penalty of 2.0
            
            if confidence < self.utility_params["Cmin"]:
                # Bounded penalty for low confidence
                deficit_ratio = (self.utility_params["Cmin"] - confidence) / (self.utility_params["Cmin"] + 1e-6)
                penalty += min(self.utility_params["pev"] * deficit_ratio, 1.0)  # Max penalty of 1.0
            
            # FIXED: Subtract penalty from base utility (penalties reduce utility)
            final_utility = base_utility - penalty
            
            # Ensure utility doesn't go below -1.0 to prevent death spirals
            return max(final_utility, -1.0)
            
        except Exception as e:
            self.logger.warning(f"Utility calculation failed: {e}")
            return 0.1  # Return low but not catastrophic utility
    
    def _normalize_response_time(self, response_time: float) -> float:
        """Normalize response time to 0-1 scale."""
        if response_time <= self.utility_params["Rmin"]:
            return 1.0
        elif response_time >= self.utility_params["Rmax"]:
            return 0.0
        else:
            return 1.0 - ((response_time - self.utility_params["Rmin"]) / 
                         (self.utility_params["Rmax"] - self.utility_params["Rmin"]))
    
    def _normalize_confidence(self, confidence: float) -> float:
        """Normalize confidence to 0-1 scale."""
        if confidence >= self.utility_params["Cmax"]:
            return 1.0
        elif confidence <= self.utility_params["Cmin"]:
            return 0.0
        else:
            return (confidence - self.utility_params["Cmin"]) / (
                self.utility_params["Cmax"] - self.utility_params["Cmin"])
    
    def _get_model_performance_rating(self, model_name: str) -> float:
        """Get performance rating for a YOLO model.
        
        Args:
            model_name: Name of the YOLO model
            
        Returns:
            Performance rating (0-1, higher is better overall)
        """
        model_ratings = {
            "yolov5n": 0.6,  # Fast but low accuracy
            "yolov5s": 0.7,  # Balanced
            "yolov5m": 0.8,  # Good balance
            "yolov5l": 0.85, # High accuracy, slower
            "yolov5x": 0.9   # Highest accuracy, slowest
        }
        return model_ratings.get(model_name, 0.5)
    
    def _update_performance_history(self, metrics: Dict[str, Any]) -> None:
        """Update performance history for trend analysis.
        
        Args:
            metrics: Current metrics to add to history
        """
        # Add to history
        self.performance_history.append(metrics.copy())
        
        # Trim history if too large
        if len(self.performance_history) > self.max_history_size:
            self.performance_history = self.performance_history[-self.max_history_size:]
    
    async def _publish_model_switch_event(self, old_model: str, new_model: str) -> None:
        """Publish a model switch event.
        
        Args:
            old_model: Previous model name
            new_model: New model name
        """
        try:
            event_data = {
                "event_type": "model_switch",
                "old_model": old_model,
                "new_model": new_model,
                "timestamp": time.time(),
                "switch_count": self.model_switch_count,
                "system": "switch_yolo"
            }
            
            await self.nats_client.publish_json(
                self.switch_model_events_subject, 
                event_data
            )
            
            self.logger.info(
                f"Model switch event published: {old_model} -> {new_model}",
                extra=event_data
            )
            
        except Exception as e:
            self.logger.error(f"Failed to publish model switch event: {e}")
    
    async def _publish_switch_telemetry(self, metrics: Dict[str, Any]) -> None:
        """Publish Switch-specific telemetry events.
        
        Args:
            metrics: Collected metrics to publish
        """
        timestamp = time.time()
        system_name = self.plugin_config.get("system_name", "switch_yolo")
        
        # Create telemetry events for key metrics
        events = []
        
        # Core performance metrics
        performance_metrics = [
            ("image_processing_time", "seconds", "performance"),
            ("model_processing_time", "seconds", "performance"),
            ("confidence", "ratio", "performance"),
            ("utility", "score", "performance"),
            ("cpu_usage", "percent", "system"),
            ("detection_boxes", "count", "performance"),
            ("total_processed", "count", "performance")
        ]
        
        for metric_name, unit, category in performance_metrics:
            if metric_name in metrics:
                event = TelemetryEvent(
                    name=f"{system_name}.{metric_name}",
                    value=metrics[metric_name],
                    unit=unit,
                    source=f"{system_name}_monitor",
                    tags={
                        "system": system_name,
                        "category": category,
                        "model": metrics.get("current_model", "unknown")
                    }
                )
                events.append(event)
        
        # Derived metrics
        derived_metrics = [
            ("calculated_utility", "score", "derived"),
            ("cpu_efficiency", "ratio", "derived"),
            ("model_efficiency", "inferences_per_second", "derived"),
            ("model_overhead_ratio", "ratio", "derived"),
            ("model_cpu_efficiency", "efficiency_ratio", "derived"),
            ("non_model_processing_time", "seconds", "derived"),
            ("detection_rate", "per_second", "derived"),
            ("model_performance_rating", "score", "derived"),
            ("processing_rate", "per_second", "derived"),
            ("system_load_factor", "ratio", "derived"),
            ("resource_pressure", "ratio", "derived")
        ]
        
        for metric_name, unit, category in derived_metrics:
            if metric_name in metrics:
                event = TelemetryEvent(
                    name=f"{system_name}.{metric_name}",
                    value=metrics[metric_name],
                    unit=unit,
                    source=f"{system_name}_monitor",
                    tags={
                        "system": system_name,
                        "category": category,
                        "model": metrics.get("current_model", "unknown")
                    }
                )
                events.append(event)
        
        # Publish events
        for event in events:
            try:
                # Add to queue for batch processing
                await self.telemetry_queue.put(event)
                
                # Also publish immediately to Switch-specific subjects
                if event.tags.get("category") == "performance":
                    await self.nats_client.publish_json(
                        self.switch_performance_subject, 
                        event.to_dict()
                    )
                elif "utility" in event.name:
                    await self.nats_client.publish_json(
                        self.switch_utility_subject, 
                        event.to_dict()
                    )
                
                # Publish to general Switch metrics subject
                await self.nats_client.publish_json(
                    self.switch_metrics_subject, 
                    event.to_dict()
                )
                
            except Exception as e:
                self.logger.error(f"Failed to publish telemetry event {event.name}: {e}")
    
    async def _collect_once(self) -> int:
        """Override the base collect_once to use Switch-specific collection.
        
        Returns:
            Number of events collected and published
        """
        cycle_id = str(uuid.uuid4())
        start_time = time.perf_counter()
        
        try:
            # Collect Switch-specific metrics
            metrics = await self._collect_switch_metrics()
            
            # Publish telemetry
            await self._publish_switch_telemetry(metrics)
            
            # Store system snapshot
            snapshot_payload = {
                "current_state": metrics,
                "historical_trends": self._calculate_trends(),
                "controller_state": {
                    "model_switch_count": self.model_switch_count,
                    "last_model": self.last_model
                },
                "snapshot_source": "SwitchMonitorAdapter",
                "cycle_id": cycle_id,
                "system": "switch_yolo"
            }
            
            await self.store_system_snapshot(snapshot_payload)
            
            elapsed = time.perf_counter() - start_time
            
            self.logger.info(
                "Switch collection cycle completed",
                extra={
                    "cycle_id": cycle_id,
                    "metrics_collected": len(metrics),
                    "current_model": metrics.get("current_model"),
                    "utility": metrics.get("utility", 0.0),
                    "elapsed_ms": round(elapsed * 1000, 3),
                    "queue_size": self.telemetry_queue.qsize()
                }
            )
            
            return len(metrics)
            
        except Exception as e:
            self.logger.error(f"Switch collection cycle failed: {e}")
            return 0
    
    def _calculate_trends(self) -> Dict[str, Any]:
        """Calculate performance trends from history.
        
        Returns:
            Dictionary containing trend analysis
        """
        if len(self.performance_history) < 2:
            return {}
        
        trends = {}
        
        try:
            # Calculate trends for key metrics
            metrics_to_trend = ["utility", "cpu_usage", "image_processing_time", "model_processing_time", "confidence"]
            
            for metric in metrics_to_trend:
                values = [h.get(metric, 0) for h in self.performance_history[-10:] if metric in h]
                if len(values) >= 2:
                    # Simple trend calculation (slope)
                    if len(values) > 1:
                        trend = (values[-1] - values[0]) / len(values)
                        trends[f"{metric}_trend"] = trend
                        trends[f"{metric}_avg"] = sum(values) / len(values)
                        trends[f"{metric}_min"] = min(values)
                        trends[f"{metric}_max"] = max(values)
            
            # Model stability (how often model changes)
            recent_models = [h.get("current_model") for h in self.performance_history[-10:]]
            unique_models = len(set(recent_models))
            trends["model_stability"] = 1.0 - (unique_models / len(recent_models)) if recent_models else 1.0
            
        except Exception as e:
            self.logger.warning(f"Trend calculation failed: {e}")
        
        return trends
    
    async def store_system_snapshot(self, snapshot_payload: Dict[str, Any]) -> None:
        """Store Switch system snapshot in knowledge base.
        
        Args:
            snapshot_payload: Complete system snapshot data
        """
        try:
            # Extract key metrics for the snapshot
            current_state = snapshot_payload.get("current_state", {})
            utility = current_state.get("utility", 0.0)
            current_model = current_state.get("current_model", "unknown")
            
            snapshot_id = f"switch_snapshot_{uuid.uuid4()}"
            
            summary = (
                f"Switch System Snapshot [{snapshot_id}]: "
                f"Model={current_model}, Utility={utility:.3f}, "
                f"CPU={current_state.get('cpu_usage', 0):.1f}%, "
                f"ModelTime={current_state.get('model_processing_time', 0):.3f}s"
            )
            
            kb_entry = KBEntry(
                entry_id=snapshot_id,
                data_type=KBDataType.OBSERVATION,
                content=snapshot_payload,
                source="switch_monitor_adapter",
                tags=["switch_snapshot", "yolo_system", "performance", "permanent"],
                metric_name="switch.system.utility",
                metric_value=utility,
                summary=summary
            )
            
            # Publish to knowledge base
            await self.nats_client.publish_json(
                self.telemetry_snapshot_subject, 
                kb_entry.model_dump()
            )
            
            self.logger.debug(
                f"Stored Switch snapshot {snapshot_id} with utility {utility:.3f}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to store Switch snapshot: {e}")


# Convenience function to create and run the Switch monitor adapter
async def run_switch_monitor(
    polaris_config_path: str = "config/switch_optimized_config.yaml",
    plugin_dir: str = "extern/switch_plugin",
    log_level: str = "INFO"
) -> None:
    """Run the Switch monitor adapter.
    
    Args:
        polaris_config_path: Path to POLARIS configuration
        plugin_dir: Path to Switch plugin directory
        log_level: Logging level
    """
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger("switch_monitor")
    
    # Create and run adapter
    adapter = SwitchMonitorAdapter(polaris_config_path, plugin_dir, logger)
    
    try:
        async with adapter:
            logger.info("Switch monitor adapter started - press Ctrl+C to stop")
            # Run indefinitely until interrupted
            while True:
                await asyncio.sleep(1)
                
    except KeyboardInterrupt:
        logger.info("Switch monitor adapter stopped by user")
    except Exception as e:
        logger.error(f"Switch monitor adapter failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(run_switch_monitor())