# strategy/controller_strategy.py
from .fast_controller import FastController
from .slow_controller import SlowController

class ControllerStrategy:
    def __init__(self):
        # Initialize all strategies
        self.controllers = {
            "fast": FastController(),
            "slow": SlowController()
        }
        self.current = self.controllers["fast"]  # default

    def select_controller(self, telemetry_data):
        # Example heuristic: select based on CPU trend
        cpu = telemetry_data.get("cpu", 0)

        if cpu > 80:
            self.current = self.controllers["fast"]
        else:
            self.current = self.controllers["slow"]

        return self.current
