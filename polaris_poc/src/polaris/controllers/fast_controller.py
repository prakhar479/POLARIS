# polaris_poc/src/controllers/fast_controller.py

# from polaris.kernel.kernel import SWIMKernel
from polaris.models.telemetry import TelemetryBatch, TelemetryEvent
import json

class FastController:
    def __init__(self, kernel):
        """
        Initialize the FastController with a reference to the kernel.
        """
        self.kernel = kernel


    def decide_action(self, telemetry: TelemetryBatch):
        """
        Decide an action based on the telemetry data.
        """
        ##get the latest telemetry event
        # self.kernel.logger.info(f"Deciding action based on telemetry data {telemetry}")
        # telemetry_data = json.loads(telemetry.data.decode())
        events = telemetry.get("events", [])
        # self.kernel.logger.info(f"Received telemetry events: {events}")
        # iterate 'name' and 'value' key in each, get avg for all names
        avg_values = {}
        for event in events:
            name = event.get("name")
            value = event.get("value")
            if name not in avg_values:
                avg_values[name] = []
            avg_values[name].append(value)

        if avg_values["swim.server.utilization"][0]>0.5:
            return {"action_type":"REMOVE_SERVER"
                    ,"source":"fast_controller",
                    "action_id":"123e4567-e89b-12d3-a456-426614174000",
                    "params":{"server_type":"compute",
                              "count":1},
                    "priority":"normal",}
        else:
            return {"action_type":"ADD_SERVER","source":"fast_controller",
                    "action_id":"123e4567-e89b-12d3-a456-426614174000",
                    "params":{"server_type":"compute",
                              "count":1},
                    "priority":"normal",}
        

        


