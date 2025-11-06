"""
SWITCH System Kernel for POLARIS Framework.

This module implements the kernel orchestration for the SWITCH ML-enabled
adaptive system, managing the interaction between fast and slow controllers
for YOLO model switching optimization.
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, Any, Optional

from polaris.common.nats_client import NATSClient
from polaris.kernel.kernel import BaseKernel
from polaris.controllers.switch_fast_controller import SwitchFastController
from polaris.controllers.switch_slow_controller import SwitchSlowController
from polaris.controllers.controller_strategy import ControllerStrategy


class SwitchControllerStrategy:
    """
    Controller selection strategy for SWITCH system.
    
    Determines whether to use fast (reactive) or slow (reasoner) controller
    based on system state and adaptation requirements.
    """
    
    def __init__(self):
        self.controllers = {
            "fast": SwitchFastController(),
            "slow": SwitchSlowController()
        }
        self.current = self.controllers["fast"]  # Default to fast
        
        # Strategy parameters
        self.utility_trend_window = 5
        self.recent_utilities = []
        self.action_count = 0
        self.optimization_interval = 15  # Trigger slow controller every N adaptations (increased for stability)
        
        # Add logger
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def select_controller(self, telemetry_data: Dict[str, Any]):
        """
        Select appropriate controller based on system state.
        
        FLIPPED STRATEGY:
        1. Use FAST controller for CRITICAL situations (low utility, declining performance, high processing time)
        2. Use SLOW controller (reasoner) for STABLE situations where proactive optimization is safe
        
        Args:
            telemetry_data: Current telemetry batch
            
        Returns:
            Selected controller instance
        """
        events = telemetry_data.get("events", [])
        
        # Extract key metrics
        utility = None
        processing_time = None
        cpu_usage = None
        
        for event in events:
            name = event.get("name", "")
            if "utility" in name:
                utility = event.get("value")
            elif "image.processing.time" in name or "processing.time" in name:
                processing_time = event.get("value")
            elif "cpu.usage" in name:
                cpu_usage = event.get("value")
        
        # Track utility trend (fix duplicate tracking issue)
        if utility is not None and utility not in self.recent_utilities[-1:]:
            self.recent_utilities.append(utility)
            if len(self.recent_utilities) > self.utility_trend_window:
                self.recent_utilities.pop(0)
        
        # Increment action counter
        self.action_count += 1
        
        # Decision logic with detailed logging
        self.logger.info(
            "Evaluating controller selection criteria (FLIPPED STRATEGY)",
            extra={
                "action_count": self.action_count,
                "current_utility": utility,
                "processing_time": processing_time,
                "cpu_usage": cpu_usage,
                "recent_utilities": self.recent_utilities[-3:] if self.recent_utilities else []
            }
        )
        
        # CRITICAL SITUATIONS → FAST CONTROLLER (immediate reactive response)
        
        # 1. Critically low utility - need immediate action
        if utility is not None and utility < 0.3:
            self.logger.warning(
                "CRITICAL: Selecting FAST controller for immediate response to low utility",
                extra={
                    "reason": f"Critical utility {utility:.3f} < 0.3 threshold",
                    "current_utility": utility,
                    "action": "immediate_reactive_response"
                }
            )
            self.current = self.controllers["fast"]
            return self.current
        
        # 2. High processing time - system under stress
        if processing_time is not None and processing_time > 2.0:
            self.logger.warning(
                "CRITICAL: Selecting FAST controller for high processing time",
                extra={
                    "reason": f"Processing time {processing_time:.2f}s > 2.0s threshold",
                    "processing_time": processing_time,
                    "action": "immediate_performance_response"
                }
            )
            self.current = self.controllers["fast"]
            return self.current
        
        # 3. Rapidly declining utility trend - system degrading
        if len(self.recent_utilities) >= self.utility_trend_window:
            mid_point = len(self.recent_utilities) // 2
            first_half_avg = sum(self.recent_utilities[:mid_point]) / mid_point
            second_half_avg = sum(self.recent_utilities[mid_point:]) / (len(self.recent_utilities) - mid_point)
            
            if second_half_avg < first_half_avg * 0.85:  # 15% rapid decline
                decline_pct = ((first_half_avg - second_half_avg) / first_half_avg) * 100
                self.logger.warning(
                    "CRITICAL: Selecting FAST controller for rapid utility decline",
                    extra={
                        "reason": f"Rapid utility decline {decline_pct:.1f}% over last {self.utility_trend_window} measurements",
                        "first_half_avg": first_half_avg,
                        "second_half_avg": second_half_avg,
                        "action": "immediate_degradation_response"
                    }
                )
                self.current = self.controllers["fast"]
                return self.current
        
        # 4. High CPU usage - resource constraint
        if cpu_usage is not None and cpu_usage > 85.0:
            self.logger.warning(
                "CRITICAL: Selecting FAST controller for high CPU usage",
                extra={
                    "reason": f"CPU usage {cpu_usage:.1f}% > 85% threshold",
                    "cpu_usage": cpu_usage,
                    "action": "immediate_resource_response"
                }
            )
            self.current = self.controllers["fast"]
            return self.current
        
        # STABLE SITUATIONS → SLOW CONTROLLER (proactive optimization)
        
        # 5. Periodic optimization when system is stable
        if (self.action_count % self.optimization_interval == 0 and 
            utility is not None and utility >= 0.5):
            self.logger.info(
                "STABLE: Selecting SLOW controller for proactive optimization",
                extra={
                    "reason": f"Periodic optimization (action #{self.action_count}) with stable utility {utility:.3f}",
                    "recent_utilities": self.recent_utilities[-3:] if self.recent_utilities else [],
                    "action": "proactive_optimization"
                }
            )
            self.current = self.controllers["slow"]
            return self.current
        
        # 6. Good utility with room for improvement
        if (utility is not None and 0.5 <= utility < 0.8 and 
            processing_time is not None and processing_time < 1.5):
            self.logger.info(
                "STABLE: Selecting SLOW controller for utility improvement",
                extra={
                    "reason": f"Good utility {utility:.3f} with optimization potential",
                    "current_utility": utility,
                    "processing_time": processing_time,
                    "action": "proactive_improvement"
                }
            )
            self.current = self.controllers["slow"]
            return self.current
        
        # DEFAULT: Fast controller for uncertain situations
        self.logger.info(
            "DEFAULT: Selecting FAST controller for reactive adaptation",
            extra={
                "reason": "No clear stable/critical classification - defaulting to reactive",
                "current_utility": utility,
                "processing_time": processing_time,
                "action": "default_reactive"
            }
        )
        self.current = self.controllers["fast"]
        return self.current


class SwitchKernel(BaseKernel):
    """
    Kernel for SWITCH ML-enabled adaptive system.
    
    Orchestrates adaptation decisions by routing telemetry to appropriate
    controllers (fast reactive or slow optimization) and managing action
    execution through verification.
    """
    
    def __init__(self, nats_url: str, logger: logging.Logger):
        """Initialize SWITCH kernel.
        
        Args:
            nats_url: NATS server URL
            logger: Logger instance
        """
        super().__init__(nats_url=nats_url, logger=logger, name="SwitchKernel")
        self.strategy = SwitchControllerStrategy()
        self.controller = None  # Will be set by strategy
        
        # SWITCH-specific parameters
        self.last_action_time = 0
        self.min_action_interval = 1.0  # Minimum seconds between actions (reduced for critical responses)
        self.last_reasoner_time = 0
        self.min_reasoner_interval = 5.0  # Minimum seconds between reasoner requests
        
        self.logger.info(
            "SwitchKernel initialized",
            extra={
                "nats_url": nats_url,
                "enable_verification": self.enable_verification,
                "verification_timeout": self.verification_timeout
            }
        )
    
    async def process_telemetry_event(self, msg):
        """Process telemetry events from SWITCH system.
        
        Routes telemetry to appropriate controller based on system state.
        
        Args:
            msg: NATS message containing telemetry batch
        """
        try:
            telemetry_data = json.loads(msg.data.decode())
            
            # Log telemetry receipt with detailed event information
            events = telemetry_data.get("events", [])
            event_count = len(events)
            
            # Extract key metrics for logging
            key_metrics = {}
            for event in events:
                name = event.get("name", "")
                value = event.get("value")
                if any(metric in name for metric in ["image.processing.time", "confidence", "utility", "cpu.usage", "current.model"]):
                    key_metrics[name] = value
            
            self.logger.info(
                "Processing telemetry batch for adaptation decision",
                extra={
                    "event_count": event_count,
                    "source": telemetry_data.get("source", "unknown"),
                    "key_metrics": key_metrics
                }
            )
            
            # Select controller
            controller = self.strategy.select_controller(telemetry_data)
            self.controller = controller
            
            controller_name = controller.__class__.__name__
            
            # Log controller selection reasoning
            selection_reason = self._get_controller_selection_reason(telemetry_data)
            
            self.logger.info(
                "Selected controller for adaptation decision",
                extra={
                    "controller": controller_name,
                    "action_count": self.strategy.action_count,
                    "recent_utilities": self.strategy.recent_utilities[-3:] if self.strategy.recent_utilities else [],
                    "selection_reason": selection_reason
                }
            )
            
            # Handle slow controller (reasoner) path with rate limiting
            if isinstance(controller, SwitchSlowController):
                # Check reasoner rate limiting
                import time
                current_time = time.time()
                if current_time - self.last_reasoner_time < self.min_reasoner_interval:
                    self.logger.info(
                        "Skipping reasoner delegation due to rate limiting",
                        extra={
                            "time_since_last_reasoner": current_time - self.last_reasoner_time,
                            "min_interval_required": self.min_reasoner_interval,
                            "fallback": "using_fast_controller"
                        }
                    )
                    # Fallback to fast controller
                    controller = self.strategy.controllers["fast"]
                    self.controller = controller
                else:
                    # Delegate to reasoner for proactive optimization
                    await self.nats_client.publish(
                        "polaris.reasoner.kernel.requests",
                        json.dumps(telemetry_data).encode()
                    )
                    self.logger.info(
                        "Delegated to reasoner for proactive optimization",
                        extra={
                            "reason": "stable_system_optimization",
                            "recent_utilities": self.strategy.recent_utilities[-3:] if len(self.strategy.recent_utilities) >= 3 else []
                        }
                    )
                    self.last_reasoner_time = current_time
                    return
            
            # Fast controller path - generate action immediately
            action = self.generate_action(telemetry_data)
            
            # Extract utility for logging
            utility = None
            for event in events:
                if event.get("name") == "switch.yolo.switch.yolo.utility":
                    utility = event.get("value")
                    break

            if action is None:
                self.logger.info(
                    "No adaptation action generated",
                    extra={
                        "reason": "No action needed based on current system state",
                        "controller": controller_name,
                        "current_utility": utility,
                        "recent_utilities": self.strategy.recent_utilities[-3:] if self.strategy.recent_utilities else []
                    }
                )
                return
            
            # Log action generation details
            self.logger.info(
                "Adaptation action generated",
                extra={
                    "action_type": action.get("action_type"),
                    "target_model": action.get("params", {}).get("model"),
                    "reason": action.get("metadata", {}).get("reason"),
                    "priority": action.get("priority"),
                    "controller": controller_name
                }
            )
            
            # Check minimum action interval
            import time
            current_time = time.time()
            if current_time - self.last_action_time < self.min_action_interval:
                self.logger.info(
                    "Skipping adaptation action due to timing constraint",
                    extra={
                        "time_since_last_action": current_time - self.last_action_time,
                        "min_interval_required": self.min_action_interval,
                        "action_type": action.get("action_type"),
                        "target_model": action.get("params", {}).get("model"),
                        "reason": action.get("metadata", {}).get("reason")
                    }
                )
                return
            
            # Extract context for verification
            context = self._extract_context_from_telemetry(telemetry_data)
            
            # Log action execution attempt
            self.logger.info(
                "Executing adaptation action with verification",
                extra={
                    "action_type": action.get("action_type"),
                    "target_model": action.get("params", {}).get("model"),
                    "priority": action.get("priority"),
                    "current_context": context,
                    "controller": controller_name
                }
            )
            
            # Execute action with verification
            await self.execute_action_with_verification(action, context)
            
            self.last_action_time = current_time
            
        except Exception as e:
            self.logger.error(
                "Error processing telemetry event",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
    
    def generate_action(self, telemetry_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate adaptation action from telemetry.
        
        Args:
            telemetry_data: Telemetry batch data
            
        Returns:
            Action dictionary or None if no action needed
        """
        if self.controller:
            return self.controller.decide_action(telemetry_data)
        return None
    
    def _extract_context_from_telemetry(self, telemetry_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant context from telemetry for verification.
        
        Args:
            telemetry_data: Telemetry batch
            
        Returns:
            Context dictionary with current system state
        """
        context = {}
        
        # Extract current system state from telemetry
        events = telemetry_data.get("events", [])
        
        for event in events:
            name = event.get("name", "")
            value = event.get("value")
            
            # Map SWITCH metrics to context
            if "current_model" in name:
                context["current_model"] = value
            elif "utility" in name:
                context["utility"] = value
            elif "image_processing_time" in name:
                context["response_time"] = value
            elif "confidence" in name:
                context["confidence"] = value
            elif "cpu_usage" in name:
                context["cpu_usage"] = value
            elif "detection_boxes" in name:
                context["detection_boxes"] = value
            elif "total_processed" in name:
                context["total_processed"] = value
        
    def _get_controller_selection_reason(self, telemetry_data: Dict[str, Any]) -> str:
        """Analyze telemetry data to determine controller selection reasoning.
        
        Args:
            telemetry_data: Current telemetry batch
            
        Returns:
            Human-readable explanation of controller selection
        """
        events = telemetry_data.get("events", [])
        
        # Extract key metrics
        utility = None
        processing_time = None
        cpu_usage = None
        
        for event in events:
            name = event.get("name", "")
            if "utility" in name:
                utility = event.get("value")
            elif "image.processing.time" in name or "processing.time" in name:
                processing_time = event.get("value")
            elif "cpu.usage" in name:
                cpu_usage = event.get("value")
        
        # Analyze selection criteria (FLIPPED STRATEGY)
        reasons = []
        
        # Critical situations (FAST controller)
        if utility is not None and utility < 0.3:
            reasons.append(f"CRITICAL: Low utility {utility:.3f} < 0.3")
        
        if processing_time is not None and processing_time > 2.0:
            reasons.append(f"CRITICAL: High processing time {processing_time:.2f}s > 2.0s")
        
        if cpu_usage is not None and cpu_usage > 85.0:
            reasons.append(f"CRITICAL: High CPU usage {cpu_usage:.1f}% > 85%")
        
        # Check rapid utility decline
        if len(self.strategy.recent_utilities) >= self.strategy.utility_trend_window:
            mid_point = len(self.strategy.recent_utilities) // 2
            first_half_avg = sum(self.strategy.recent_utilities[:mid_point]) / mid_point
            second_half_avg = sum(self.strategy.recent_utilities[mid_point:]) / (len(self.strategy.recent_utilities) - mid_point)
            
            if second_half_avg < first_half_avg * 0.85:
                decline_pct = ((first_half_avg - second_half_avg) / first_half_avg) * 100
                reasons.append(f"CRITICAL: Rapid utility decline {decline_pct:.1f}%")
        
        # Stable situations (SLOW controller)
        if (self.strategy.action_count % self.strategy.optimization_interval == 0 and 
            utility is not None and utility >= 0.5):
            reasons.append(f"STABLE: Periodic optimization with stable utility {utility:.3f}")
        
        if (utility is not None and 0.5 <= utility < 0.8 and 
            processing_time is not None and processing_time < 1.5):
            reasons.append(f"STABLE: Good utility {utility:.3f} with optimization potential")
        
        # Default case
        if not reasons:
            reasons.append("DEFAULT: Reactive adaptation (fast controller)")
            
        return "; ".join(reasons)


async def main():
    """Main entry point for SWITCH kernel."""
    # Setup logging
    logger = logging.getLogger("SwitchKernel")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start kernel
    kernel = SwitchKernel(
        nats_url="nats://localhost:4222",
        logger=logger
    )
    
    try:
        await kernel.start()
        logger.info("SwitchKernel started successfully")
        
        # Run until interrupted
        while kernel.running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        await kernel.stop()
        logger.info("SwitchKernel stopped")


if __name__ == "__main__":
    asyncio.run(main())
