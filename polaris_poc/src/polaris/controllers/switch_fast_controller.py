"""
Fast Controller for SWITCH System.

Implements reactive, rule-based adaptation for YOLO model switching
to maximize utility function based on response time and confidence.
"""

import uuid
import time
import logging
from typing import Dict, Any, Optional, List
from polaris.controllers.controller import BaseController


class SwitchFastController(BaseController):
    """
    Fast reactive controller for SWITCH ML system.
    
    Uses rule-based logic to switch YOLO models based on utility function
    optimization considering response time and confidence tradeoffs.
    """
    
    def __init__(self):
        """Initialize SWITCH fast controller with thresholds and model info."""
        super().__init__()
        
        # OPTIMIZED: Utility function parameters for switch exemplar
        self.Rmin = 0.05  # Best response time (yolov5n baseline)
        self.Rmax = 2.0   # More lenient max response time (prevent death spirals)
        self.Cmin = 0.3   # More lenient min confidence (prevent death spirals)
        self.Cmax = 1.0   # Best confidence
        self.wd = 0.6     # Slightly favor response time (60/40 split)
        self.we = 0.4     # Confidence weight
        self.pdv = 2.0    # Reduced penalty scale (prevent exponential degradation)
        self.pev = 2.0    # Reduced penalty scale
        
        # OPTIMIZED: More aggressive utility thresholds for switch exemplar
        self.UTILITY_LOW_THRESHOLD = 0.3   # Below this, consider switching (more lenient)
        self.UTILITY_CRITICAL_THRESHOLD = 0.1  # Below this, immediate action
        self.UTILITY_TARGET = 0.7           # Target utility level (more realistic)
        
        # OPTIMIZED: Model characteristics based on switch exemplar analysis
        self.model_profiles = {
            "yolov5n": {
                "response_time": 0.05,  # ~50ms
                "confidence": 0.65,
                "utility_estimate": None,  # Calculated dynamically
                "cpu_factor": 1.0,
                "order": 1,
                "stability_score": 0.9  # Most stable (least resource intensive)
            },
            "yolov5s": {
                "response_time": 0.10,  # ~100ms
                "confidence": 0.75,
                "utility_estimate": None,
                "cpu_factor": 1.5,
                "order": 2,
                "stability_score": 0.8  # Good balance
            },
            "yolov5m": {
                "response_time": 0.20,  # ~200ms
                "confidence": 0.82,
                "utility_estimate": None,
                "cpu_factor": 2.5,
                "order": 3,
                "stability_score": 0.6  # Moderate stability
            },
            "yolov5l": {
                "response_time": 0.40,  # ~400ms
                "confidence": 0.88,
                "utility_estimate": None,
                "cpu_factor": 4.0,
                "order": 4,
                "stability_score": 0.4  # Less stable (high resource usage)
            },
            "yolov5x": {
                "response_time": 0.80,  # ~800ms
                "confidence": 0.92,
                "utility_estimate": None,
                "cpu_factor": 6.0,
                "order": 5,
                "stability_score": 0.2  # Least stable (highest resource usage)
            }
        }
        
        # OPTIMIZED: More conservative thresholds to prevent cascading failures
        self.CPU_HIGH_THRESHOLD = 80.0   # Above this, consider lighter model (more conservative)
        self.CPU_CRITICAL_THRESHOLD = 90.0  # Above this, immediate downgrade
        self.CPU_LOW_THRESHOLD = 40.0     # Below this, consider heavier model (more conservative)
        
        # Response time thresholds
        self.RT_CRITICAL_THRESHOLD = 1.5  # Critical response time (more lenient)
        self.RT_HIGH_THRESHOLD = 0.8      # High response time
        self.RT_GOOD_THRESHOLD = 0.3      # Good response time (more lenient)
        
        # Confidence thresholds
        self.CONF_LOW_THRESHOLD = 0.4     # Low confidence (more lenient)
        self.CONF_HIGH_THRESHOLD = 0.8    # High confidence (more realistic)
        
        # Stability tracking
        self.recent_switches = []
        self.max_switches_per_hour = 20   # Prevent thrashing
        self.min_switch_interval = 3.0    # Minimum seconds between switches
        
        # Spiral detection
        self.recent_utilities = []
        self.utility_history_size = 5
        self.spiral_detection_enabled = True
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def decide_action(self, telemetry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Decide adaptation action based on telemetry.
        
        Args:
            telemetry: Telemetry batch with system metrics
            
        Returns:
            Action dictionary or None if no action needed
        """
        # Extract telemetry values
        events = telemetry.get("events", [])
        telemetry_values = self._extract_telemetry_values(events)

        self.logger.debug("Extracted telemetry values", extra={"telemetry": telemetry_values})
        self.logger.debug("Raw events received", extra={"event_count": len(events), "events": events})
        
        # Validate required metrics
        if not self._validate_telemetry(telemetry_values):
            self.logger.warning("Missing required metrics for decision")
            self.logger.info(f"Available metrics:{list(telemetry_values.keys())}")
            return None
        
        # Calculate current utility
        current_utility = self._calculate_utility(telemetry_values)
        telemetry_values["calculated_utility"] = current_utility
        
        self.logger.debug(
            "Current system state",
            extra={
                "utility": current_utility,
                "response_time": telemetry_values.get("switch.yolo.switch.yolo.image.processing.time"),
                "confidence": telemetry_values.get("switch.yolo.switch.yolo.confidence"),
                "model": telemetry_values.get("switch.yolo.switch.yolo.current.model"),
                "cpu": telemetry_values.get("switch.yolo.switch.yolo.cpu.usage")
            }
        )
        
        # Decide on model switch
        return self._decide_model_switch(telemetry_values, current_utility)
    
    def _extract_telemetry_values(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract and average telemetry values from events.
        
        Args:
            events: List of telemetry events
            
        Returns:
            Dictionary mapping metric names to average values
        """
        values = {}
        for event in events:
            name = event.get("name")
            value = event.get("value")
            if name not in values:
                values[name] = []
            values[name].append(value)
        
        # Average multiple readings
        telemetry_data = {}
        for key, value_list in values.items():
            # For strings (like model name), take the last value
            if isinstance(value_list[0], str):
                telemetry_data[key] = value_list[-1]
            else:
                # For numbers, calculate average
                telemetry_data[key] = sum(value_list) / len(value_list)
        
        return telemetry_data
    
    def _validate_telemetry(self, telemetry_values: Dict[str, Any]) -> bool:
        """Validate that required metrics are present.
        
        Args:
            telemetry_values: Extracted telemetry values
            
        Returns:
            True if all required metrics present, False otherwise
        """
        required_metrics = [
            "switch.yolo.switch.yolo.image.processing.time",
            "switch.yolo.switch.yolo.confidence",
            "switch.yolo.switch.yolo.current.model",
            "switch.yolo.switch.yolo.cpu.usage"
        ]
        return all(metric in telemetry_values for metric in required_metrics)
    
    def _calculate_utility(self, telemetry_values: Dict[str, Any]) -> float:
        """Calculate utility function from telemetry using FIXED penalty math.
        
        Uses the corrected utility function:
        base_utility = we * conf_score + wd * rt_score - bounded_penalties
        
        Args:
            telemetry_values: Current metrics
            
        Returns:
            Utility score (higher is better, range [-1.0, 1.0])
        """
        r = telemetry_values.get("switch.yolo.switch.yolo.image.processing.time", 0.5)
        C = telemetry_values.get("switch.yolo.switch.yolo.confidence", 0.5)
        
        # Input sanitization
        try:
            r = max(0.001, float(r))  # Ensure positive response time
            C = max(0.0, min(1.0, float(C)))  # Clamp confidence to [0,1]
        except:
            return 0.1  # Return low but not catastrophic utility
        
        import math
        if math.isnan(r) or math.isnan(C) or not math.isfinite(r) or not math.isfinite(C):
            return 0.1
        
        # Normalize confidence: map C in [Cmin, Cmax] to [0, 1]
        if C <= self.Cmin:
            conf_score = 0.0
        elif C >= self.Cmax:
            conf_score = 1.0
        else:
            conf_score = (C - self.Cmin) / (self.Cmax - self.Cmin)
        
        # Normalize response time: map r in [Rmin, Rmax] so Rmin->1.0, Rmax->0.0
        if r <= self.Rmin:
            rt_score = 1.0
        elif r >= self.Rmax:
            rt_score = 0.0
        else:
            rt_score = (self.Rmax - r) / (self.Rmax - self.Rmin)
        
        # Base utility
        base_utility = self.we * conf_score + self.wd * rt_score
        
        # FIXED: Bounded penalties for out-of-bounds (prevent death spirals)
        penalty = 0.0
        
        if r > self.Rmax:
            # Bounded penalty to prevent exponential degradation
            excess_ratio = (r - self.Rmax) / (self.Rmax + 1e-6)
            penalty += min(self.pdv * excess_ratio * 0.2, 1.0)  # Max penalty of 1.0, reduced factor
        
        if C < self.Cmin:
            # Bounded penalty for low confidence
            deficit_ratio = (self.Cmin - C) / (self.Cmin + 1e-6)
            penalty += min(self.pev * deficit_ratio * 0.2, 0.5)  # Max penalty of 0.5, reduced factor
        
        # FIXED: Subtract penalty from base utility (penalties reduce utility)
        final_utility = base_utility - penalty
        
        # Ensure utility doesn't go below -1.0 to prevent death spirals
        return max(final_utility, -1.0)
    
    def _decide_model_switch(
        self,
        telemetry_values: Dict[str, Any],
        current_utility: float
    ) -> Optional[Dict[str, Any]]:
        """OPTIMIZED: Decide whether to switch models with stability checks.
        
        Decision Logic (Priority Order):
        1. Check switching rate limits (prevent thrashing)
        2. Critical safety constraints (CPU, RT)
        3. Utility optimization with stability consideration
        4. Opportunistic improvements (conservative)
        
        Args:
            telemetry_values: Current metrics
            current_utility: Calculated utility score
            
        Returns:
            Action dictionary or None
        """
        current_model = telemetry_values.get("switch.yolo.switch.yolo.current.model", "yolov5s")
        response_time = telemetry_values.get("switch.yolo.switch.yolo.image.processing.time", 0.5)
        confidence = telemetry_values.get("switch.yolo.switch.yolo.confidence", 0.5)
        cpu_usage = telemetry_values.get("switch.yolo.switch.yolo.cpu.usage", 50.0)
        
        # Get current model order and stability
        current_order = self.model_profiles.get(current_model, {}).get("order", 2)
        current_stability = self.model_profiles.get(current_model, {}).get("stability_score", 0.5)
        
        # 1. STABILITY CHECK: Prevent thrashing
        if not self._can_switch_model():
            self.logger.info(
                "RATE_LIMITED: Model switch blocked to prevent thrashing",
                extra={
                    "recent_switches": len(self.recent_switches),
                    "time_since_last": time.time() - self.recent_switches[-1] if self.recent_switches else 0,
                    "min_interval": self.min_switch_interval,
                    "decision_log": "RATE_LIMITED: Too many recent switches, maintaining current model"
                }
            )
            return None
        
        # 1.5. SPIRAL DETECTION: Check for utility death spiral
        spiral_detected = self._detect_utility_spiral(current_utility, response_time)
        if spiral_detected and current_order > 1:
            # Emergency recovery - go to most stable model regardless of other metrics
            target_model = "yolov5n"
            reason = f"SPIRAL_RECOVERY: Utility spiral detected (U={current_utility:.3f}, RT={response_time:.3f}s)"
            
            self.logger.error(
                "SPIRAL_RECOVERY: Utility death spiral detected - emergency recovery to most stable model",
                extra={
                    "utility": current_utility,
                    "response_time": response_time,
                    "current_model": current_model,
                    "target_model": target_model,
                    "recent_utilities": self.recent_utilities,
                    "decision_log": f"SPIRAL_RECOVERY: Death spiral detected → {target_model} for stability (ignore utility temporarily)"
                }
            )
            return self._create_switch_action(target_model, reason, "high")
        
        # 2. CRITICAL SAFETY: CPU overload (immediate action)
        if cpu_usage > self.CPU_CRITICAL_THRESHOLD and current_order > 1:
            # Emergency downgrade - go to most stable model
            target_model = "yolov5n"  # Most stable
            reason = f"CRITICAL: CPU overload ({cpu_usage:.1f}%) - emergency downgrade"
            
            self.logger.warning(
                "EMERGENCY_CPU: Critical CPU overload detected",
                extra={
                    "cpu_usage": cpu_usage,
                    "threshold": self.CPU_CRITICAL_THRESHOLD,
                    "current_model": current_model,
                    "target_model": target_model,
                    "utility": current_utility,
                    "decision_log": f"EMERGENCY_CPU: CPU {cpu_usage:.1f}% > {self.CPU_CRITICAL_THRESHOLD}% → {target_model} for stability"
                }
            )
            return self._create_switch_action(target_model, reason, "high")
        
        # 3. CRITICAL SAFETY: Response time explosion (immediate action)
        if response_time > self.RT_CRITICAL_THRESHOLD and current_order > 1:
            # Aggressive but controlled downgrade
            target_order = max(1, current_order - 2)
            target_model = self._get_model_by_order(target_order)
            reason = f"CRITICAL: Response time explosion ({response_time:.3f}s)"
            
            self.logger.warning(
                "EMERGENCY_RT: Critical response time explosion detected",
                extra={
                    "response_time": response_time,
                    "threshold": self.RT_CRITICAL_THRESHOLD,
                    "current_model": current_model,
                    "target_model": target_model,
                    "utility": current_utility,
                    "downgrade_levels": current_order - target_order,
                    "decision_log": f"EMERGENCY_RT: RT {response_time:.3f}s > {self.RT_CRITICAL_THRESHOLD}s → {target_model} (skip {current_order - target_order} levels)"
                }
            )
            return self._create_switch_action(target_model, reason, "high")
        
        # 4. CRITICAL SAFETY: Utility death spiral
        if current_utility < self.UTILITY_CRITICAL_THRESHOLD and current_order > 1:
            # Go to most stable model to recover
            target_model = "yolov5n"
            reason = f"CRITICAL: Utility death spiral ({current_utility:.3f}) - recovery mode"
            
            self.logger.error(
                "SPIRAL_DETECTED: Utility death spiral detected - emergency recovery",
                extra={
                    "utility": current_utility,
                    "threshold": self.UTILITY_CRITICAL_THRESHOLD,
                    "current_model": current_model,
                    "target_model": target_model,
                    "response_time": response_time,
                    "confidence": confidence,
                    "cpu_usage": cpu_usage,
                    "decision_log": f"SPIRAL_RECOVERY: Utility {current_utility:.3f} < {self.UTILITY_CRITICAL_THRESHOLD} → {target_model} for stability recovery"
                }
            )
            return self._create_switch_action(target_model, reason, "high")
        
        # 5. HIGH PRIORITY: CPU pressure (proactive)
        if cpu_usage > self.CPU_HIGH_THRESHOLD and current_order > 1:
            target_model = self._get_model_by_order(current_order - 1)
            reason = f"High CPU pressure ({cpu_usage:.1f}%) - proactive downgrade"
            
            self.logger.info(
                "HIGH_PRIORITY_CPU: Proactive CPU pressure management",
                extra={
                    "cpu_usage": cpu_usage,
                    "threshold": self.CPU_HIGH_THRESHOLD,
                    "current_model": current_model,
                    "target_model": target_model,
                    "utility": current_utility,
                    "decision_log": f"HIGH_PRIORITY_CPU: CPU {cpu_usage:.1f}% > {self.CPU_HIGH_THRESHOLD}% → {target_model} (proactive)"
                }
            )
            return self._create_switch_action(target_model, reason, "medium")
        
        # 6. HIGH PRIORITY: High response time (proactive)
        if response_time > self.RT_HIGH_THRESHOLD and current_order > 1:
            target_model = self._get_model_by_order(current_order - 1)
            reason = f"High response time ({response_time:.3f}s) - proactive optimization"
            
            self.logger.info(
                "HIGH_PRIORITY_RT: Proactive response time optimization",
                extra={
                    "response_time": response_time,
                    "threshold": self.RT_HIGH_THRESHOLD,
                    "current_model": current_model,
                    "target_model": target_model,
                    "utility": current_utility,
                    "decision_log": f"HIGH_PRIORITY_RT: RT {response_time:.3f}s > {self.RT_HIGH_THRESHOLD}s → {target_model} (proactive)"
                }
            )
            return self._create_switch_action(target_model, reason, "medium")
        
        # 7. MEDIUM PRIORITY: Utility optimization
        if current_utility < self.UTILITY_LOW_THRESHOLD:
            best_model = self._find_best_model_for_utility_stable(
                telemetry_values,
                current_model,
                current_order,
                current_stability
            )
            
            if best_model != current_model:
                reason = f"Utility optimization (current: {current_utility:.3f})"
                
                self.logger.info(
                    "OPTIMIZATION: Utility-based model optimization",
                    extra={
                        "current_utility": current_utility,
                        "threshold": self.UTILITY_LOW_THRESHOLD,
                        "current_model": current_model,
                        "target_model": best_model,
                        "response_time": response_time,
                        "confidence": confidence,
                        "cpu_usage": cpu_usage,
                        "decision_log": f"OPTIMIZATION: Utility {current_utility:.3f} < {self.UTILITY_LOW_THRESHOLD} → {best_model} for better performance"
                    }
                )
                return self._create_switch_action(best_model, reason, "medium")
        
        # 8. LOW PRIORITY: Quality improvement (very conservative)
        if (confidence < self.CONF_LOW_THRESHOLD and 
            response_time < self.RT_GOOD_THRESHOLD and 
            cpu_usage < self.CPU_LOW_THRESHOLD and
            current_order < 5 and
            current_utility > self.UTILITY_LOW_THRESHOLD):  # Only if utility is acceptable
            
            target_model = self._get_model_by_order(current_order + 1)
            reason = f"Conservative quality improvement (conf: {confidence:.3f})"
            
            self.logger.info(
                "QUALITY_UPGRADE: Conservative quality improvement with spare resources",
                extra={
                    "confidence": confidence,
                    "conf_threshold": self.CONF_LOW_THRESHOLD,
                    "response_time": response_time,
                    "cpu_usage": cpu_usage,
                    "current_model": current_model,
                    "target_model": target_model,
                    "utility": current_utility,
                    "decision_log": f"QUALITY_UPGRADE: Low confidence {confidence:.3f} with spare resources → {target_model} for better accuracy"
                }
            )
            return self._create_switch_action(target_model, reason, "low")
        
        # No action needed - system is stable
        self.logger.info(
            "MAINTAIN: System stable - no model switch needed",
            extra={
                "utility": current_utility,
                "current_model": current_model,
                "stability_score": current_stability,
                "response_time": response_time,
                "confidence": confidence,
                "cpu_usage": cpu_usage,
                "decision_log": f"MAINTAIN: System stable (U={current_utility:.3f}, RT={response_time:.3f}s, CPU={cpu_usage:.1f}%) → keep {current_model}"
            }
        )
        return None
    
    def _can_switch_model(self) -> bool:
        """Check if model switching is allowed based on rate limits.
        
        Returns:
            True if switching is allowed, False if rate limited
        """
        import time
        current_time = time.time()
        
        # Clean old switches (older than 1 hour)
        self.recent_switches = [
            switch_time for switch_time in self.recent_switches
            if current_time - switch_time < 3600  # 1 hour
        ]
        
        # Check minimum interval since last switch
        if self.recent_switches:
            time_since_last = current_time - self.recent_switches[-1]
            if time_since_last < self.min_switch_interval:
                return False
        
        # Check maximum switches per hour
        if len(self.recent_switches) >= self.max_switches_per_hour:
            return False
        
        return True
    
    def _record_switch(self):
        """Record a model switch for rate limiting."""
        import time
        self.recent_switches.append(time.time())
    
    def _detect_utility_spiral(self, current_utility: float, response_time: float) -> bool:
        """Detect if system is in a utility death spiral.
        
        Args:
            current_utility: Current utility value
            response_time: Current response time
            
        Returns:
            True if spiral detected, False otherwise
        """
        if not self.spiral_detection_enabled:
            return False
        
        # Update utility history
        self.recent_utilities.append(current_utility)
        if len(self.recent_utilities) > self.utility_history_size:
            self.recent_utilities.pop(0)
        
        # Spiral indicators:
        # 1. Utility is very low (< 0.2)
        # 2. Response time is high (> 1.0s) 
        # 3. Utility trend is declining over recent history
        
        if current_utility < 0.2 and response_time > 1.0:
            # Check if utility has been consistently declining
            if len(self.recent_utilities) >= 3:
                recent_avg = sum(self.recent_utilities[-3:]) / 3
                if recent_avg < 0.3:  # Consistently low
                    self.logger.warning(
                        "SPIRAL_DETECTED: Utility death spiral identified",
                        extra={
                            "current_utility": current_utility,
                            "response_time": response_time,
                            "recent_utilities": self.recent_utilities,
                            "recent_avg": recent_avg,
                            "spiral_indicators": "Low utility + High RT + Declining trend"
                        }
                    )
                    return True
        
        return False
    
    def _find_best_model_for_utility_stable(
        self,
        telemetry_values: Dict[str, Any],
        current_model: str,
        current_order: int,
        current_stability: float
    ) -> str:
        """Find model that maximizes expected utility with stability consideration.
        
        Args:
            telemetry_values: Current metrics
            current_model: Current YOLO model
            current_order: Current model order (1-5)
            current_stability: Current model stability score
            
        Returns:
            Model name with best expected utility considering stability
        """
        cpu_usage = telemetry_values.get("switch.yolo.switch.yolo.cpu.usage", 50.0)
        
        best_model = current_model
        best_score = self._calculate_utility(telemetry_values) + (current_stability * 0.1)  # Stability bonus
        
        # Check adjacent models (one lighter, one heavier)
        for delta in [-1, +1]:
            target_order = current_order + delta
            if target_order < 1 or target_order > 5:
                continue
            
            target_model = self._get_model_by_order(target_order)
            profile = self.model_profiles[target_model]
            
            # Check CPU constraint with safety margin
            estimated_cpu = cpu_usage * (profile["cpu_factor"] / self.model_profiles[current_model]["cpu_factor"])
            if estimated_cpu > 85.0:  # Conservative CPU limit
                continue
            
            # Estimate utility with this model
            estimated_r = profile["response_time"]
            estimated_C = profile["confidence"]
            
            # Create temporary telemetry for calculation
            temp_values = telemetry_values.copy()
            temp_values["switch.yolo.switch.yolo.image.processing.time"] = estimated_r
            temp_values["switch.yolo.switch.yolo.confidence"] = estimated_C
            
            estimated_utility = self._calculate_utility(temp_values)
            
            # Add stability bonus (favor more stable models)
            stability_bonus = profile["stability_score"] * 0.1
            total_score = estimated_utility + stability_bonus
            
            # Require significant improvement to justify switch (hysteresis)
            improvement_threshold = 0.05  # 5% improvement required
            if total_score > best_score + improvement_threshold:
                best_score = total_score
                best_model = target_model
        
        return best_model
    
    def _get_model_by_order(self, order: int) -> str:
        """Get model name by order number.
        
        Args:
            order: Model order (1=yolov5n, 2=yolov5s, ... 5=yolov5x)
            
        Returns:
            Model name
        """
        for model_name, profile in self.model_profiles.items():
            if profile["order"] == order:
                return model_name
        return "yolov5s"  # Default fallback
    
    def _create_switch_action(
        self,
        target_model: str,
        reason: str,
        priority: str = "normal"
    ) -> Dict[str, Any]:
        """Create a model switch action with stability tracking.
        
        Args:
            target_model: Target YOLO model
            reason: Reason for switch
            priority: Action priority (low/medium/high)
            
        Returns:
            Action dictionary
        """
        # Record the switch for rate limiting
        self._record_switch()
        
        action = {
            "action_type": f"SWITCH_MODEL_{target_model.upper()}",
            "source": "switch_fast_controller_optimized",
            "action_id": str(uuid.uuid4()),
            "params": {"model": target_model},
            "priority": priority,
            "metadata": {
                "reason": reason,
                "controller": "fast_optimized",
                "switch_count": len(self.recent_switches),
                "stability_aware": True
            }
        }
        
        self.logger.info(
            "Generated optimized model switch action",
            extra={
                "target_model": target_model,
                "reason": reason,
                "priority": priority,
                "recent_switches": len(self.recent_switches),
                "stability_aware": True
            }
        )
        
        return action
