# polaris_poc/src/controllers/fast_controller.py

# from polaris.kernel.kernel import SWIMKernel
from polaris.models.telemetry import TelemetryBatch, TelemetryEvent
import json
from polaris.controllers.controller import BaseController

# FastController class
import uuid
import time
class FastController(BaseController):
    def __init__(self):
        self.RT_THRESHOLD = 0.75
        self.DIMMER_STEP = 0.1
    
    def decide_action(self, telemetry: TelemetryBatch):
        events = telemetry.get("events", [])
        telemetry_values = self._extract_telemetry_values(events)

        if not self._validate_telemetry(telemetry_values):
            return None

        weighted_response_time = self._calculate_weighted_response_time(telemetry_values)

        if weighted_response_time > self.RT_THRESHOLD:
            return self._handle_high_response_time(telemetry_values)
        else:
            return self._handle_normal_response_time(telemetry_values)

    def _extract_telemetry_values(self, events):
        values = {}
        for event in events:
            name = event.get("name")
            value = event.get("value")
            if name not in values:
                values[name] = []
            values[name].append(value)

        telemetry_data = {}
        for key, value_list in values.items():
            if value_list:
                telemetry_data[key] = sum(value_list) / len(value_list)
        return telemetry_data

    def _validate_telemetry(self, telemetry_values):
        required_metrics = [
            "swim.active.servers",
            "swim.max.servers",
            "swim.basic.response.time",
            "swim.optional.response.time",
            "swim.basic.throughput",
            "swim.optional.throughput",
            "swim.dimmer",
            "swim.server.utilization",
        ]
        return all(metric in telemetry_values for metric in required_metrics)

    def _calculate_weighted_response_time(self, telemetry_values):
        basic_rt = telemetry_values.get("swim.basic.response.time", 0)
        opt_rt = telemetry_values.get("swim.optional.response.time", 0)
        basic_tp = telemetry_values.get("swim.basic.throughput", 0)
        opt_tp = telemetry_values.get("swim.optional.throughput", 0)

        total_tp = basic_tp + opt_tp
        if total_tp == 0:
            return 0
        return (basic_tp * basic_rt + opt_tp * opt_rt) / total_tp

    def _handle_high_response_time(self, telemetry_values):
        active_servers = telemetry_values.get("swim.active.servers", 0)
        max_servers = telemetry_values.get("swim.max.servers", 0)
        current_dimmer = telemetry_values.get("swim.dimmer", 1.0)

        if active_servers < max_servers:
            return {
                "action_type": "ADD_SERVER",
                "source": "fast_controller",
                "action_id": str(uuid.uuid4()),
                "params": {"server_type": "compute", "count": 1},
                "priority": "high",
            }
        else:
            new_dimmer = max(0, current_dimmer - self.DIMMER_STEP)
            return {
                "action_type": "SET_DIMMER",
                "source": "fast_controller",
                "action_id": str(uuid.uuid4()),
                "params": {"value": new_dimmer},
                "priority": "normal",
            }

    def _handle_normal_response_time(self, telemetry_values):
        active_servers = telemetry_values.get("swim.active.servers", 0)
        total_utilization = telemetry_values.get("swim.server.utilization", 0)
        current_dimmer = telemetry_values.get("swim.dimmer", 1.0)

        spare_util = active_servers - total_utilization

        if spare_util > 1:
            if current_dimmer < 1.0:
                new_dimmer = min(1.0, current_dimmer + self.DIMMER_STEP)
                return {
                    "action_type": "SET_DIMMER",
                    "source": "fast_controller",
                    "action_id": str(uuid.uuid4()),
                    "params": {"value": new_dimmer},
                    "priority": "low",
                }
            else:
                return {
                    "action_type": "REMOVE_SERVER",
                    "source": "fast_controller",
                    "action_id": str(uuid.uuid4()),
                    "params": {"server_type": "compute", "count": 1},
                    "priority": "low",
                }
        return None

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
        

        


