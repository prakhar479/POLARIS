# polaris_poc/src/controllers/fast_controller.py

# from polaris.kernel.kernel import SWIMKernel
from polaris.models.telemetry import TelemetryBatch, TelemetryEvent
import json
from polaris.controllers.controller import BaseController

# FastController class
import uuid
import time
class FastController(BaseController):
    def __init__(self, logger):
        self.logger = logger
        
        # --- NEW: Configuration parameters for tuning ---
        self.RT_THRESHOLD = 0.75
        self.DIMMER_STEP = 0.1
        self.LOW_UTIL_THRESHOLD = 0.4  # Consider scaling down if avg server util is below 40%
        self.PERSISTENCE_COUNT = 3      # Require 3 consecutive violations before acting

        # --- NEW: State tracking for persistence ---
        self.violation_counter = 0
        self.normal_counter = 0

    def decide_action(self, telemetry_snapshot: dict):
        """
        Decide action based on a snapshot, now with persistence checks.
        """
        self.logger.info("FastController: deciding action...")

        state = telemetry_snapshot.get("current_state", {})
        telemetry_values = self._normalize_telemetry(state)

        weighted_response_time = self._calculate_weighted_response_time(telemetry_values)

        if weighted_response_time > self.RT_THRESHOLD:
            self.normal_counter = 0  # Reset normal counter
            self.violation_counter += 1
            self.logger.info(f"High RT detected. Violation count: {self.violation_counter}/{self.PERSISTENCE_COUNT}")
            
            # --- MODIFIED: Act only after persistence threshold is met ---
            if self.violation_counter >= self.PERSISTENCE_COUNT:
                self.logger.warning(f"RT threshold violated for {self.violation_counter} consecutive checks. Taking action.")
                self.violation_counter = 0  # Reset after acting
                return self._handle_high_response_time(telemetry_values)
            else:
                # Still in violation, but haven't met persistence. Do nothing.
                return self._no_op("Waiting for sustained RT violation.")
                
        else:
            self.violation_counter = 0 # Reset violation counter
            self.normal_counter += 1
            self.logger.info(f"Normal RT detected. Normal count: {self.normal_counter}")

            # We can also add persistence to scale-down actions if needed,
            # but for now, we act more readily to improve efficiency.
            return self._handle_normal_response_time(telemetry_values)

    def _handle_high_response_time(self, telemetry_values):
        active_servers = telemetry_values.get("swim.active.servers", 0)
        max_servers = telemetry_values.get("swim.max.servers", 0)

        if active_servers < max_servers:
            self.logger.warning("High response time action: ADD_SERVER")
            return self._create_action("ADD_SERVER", {"server_type": "compute", "count": 1}, "high")
        else:
            current_dimmer = telemetry_values.get("swim.dimmer", 1.0)
            new_dimmer = max(0, current_dimmer - self.DIMMER_STEP)
            self.logger.warning(f"High response time action: SET_DIMMER to {new_dimmer}")
            return self._create_action("SET_DIMMER", {"value": new_dimmer}, "normal")

    def _handle_normal_response_time(self, telemetry_values):
        active_servers = telemetry_values.get("swim.active.servers", 0)
        # --- MODIFIED: Using average utilization for a more stable metric ---
        # The old 'spare_util' was too sensitive.
        avg_utilization = telemetry_values.get("swim.server.utilization", 0) / active_servers if active_servers > 0 else 0

        # Condition 1: System is underutilized, prioritize increasing quality (dimmer)
        current_dimmer = telemetry_values.get("swim.dimmer", 1.0)
        if avg_utilization < 0.8 and current_dimmer < 1.0: # Leave some headroom
            new_dimmer = min(1.0, current_dimmer + self.DIMMER_STEP)
            self.logger.info(f"Normal RT, underutilized. Action: SET_DIMMER to {new_dimmer}")
            return self._create_action("SET_DIMMER", {"value": new_dimmer}, "low")

        # Condition 2: Quality is maxed out, and system is significantly underutilized
        # --- MODIFIED: Less aggressive scale-down logic ---
        elif current_dimmer == 1.0 and active_servers > 1 and avg_utilization < self.LOW_UTIL_THRESHOLD:
            self.logger.info(f"Normal RT, low utilization ({avg_utilization:.2f} < {self.LOW_UTIL_THRESHOLD}). Action: REMOVE_SERVER")
            return self._create_action("REMOVE_SERVER", {"server_type": "compute", "count": 1}, "low")
        
        # Otherwise, the system is stable. Do nothing.
        return self._no_op("System is stable, no action needed.")

    # --- HELPER METHODS to keep code clean ---
    def _normalize_telemetry(self, state: dict) -> dict:
        return {
            "swim.dimmer": state.get("dimmer", 1.0),
            "swim.active.servers": state.get("active_servers", 0),
            "swim.max.servers": state.get("max_servers", 0),
            "swim.basic.response.time": state.get("basic_response_time", 0.0),
            "swim.optional.response.time": state.get("optional_response_time", 0.0),
            "swim.basic.throughput": state.get("basic_throughput", 0.0),
            "swim.optional.throughput": state.get("optional_throughput", 0.0),
            "swim.server.utilization": state.get("server_utilization", 0.0),
        }

    def _calculate_weighted_response_time(self, telemetry_values):
        basic_rt = telemetry_values.get("swim.basic.response.time", 0)
        opt_rt = telemetry_values.get("swim.optional.response.time", 0)
        basic_tp = telemetry_values.get("swim.basic.throughput", 0)
        opt_tp = telemetry_values.get("swim.optional.throughput", 0)
        total_tp = basic_tp + opt_tp
        return (basic_tp * basic_rt + opt_tp * opt_rt) / total_tp if total_tp > 0 else 0

    def _create_action(self, action_type: str, params: dict, priority: str):
        return {
            "action_type": action_type,
            "source": "fast_controller",
            "action_id": str(uuid.uuid4()),
            "params": params,
            "priority": priority,
        }
        
    def _no_op(self, reason: str):
        self.logger.info(f"NO_OP: {reason}")
        return None # Return None for no action
# TestController class
class TestController(BaseController):
    def decide_action(self, telemetry: TelemetryBatch):
        """
        Simple implementation for testing purposes.
        """
        return {
            "action_type": "TEST_ACTION",
            "source": "test_controller",
            "action_id": "test-1234",
            "params": {"test_param": "value"},
            "priority": "low",
        }
        

        


# The log message `Active servers != Total servers, skipping
# action generation.` is indicating that the number of active
# servers in the system is not equal to the total number of
# servers. In this context, it means that there is a discrepancy
# between the expected number of servers that should be active
# and the total number of servers available.
