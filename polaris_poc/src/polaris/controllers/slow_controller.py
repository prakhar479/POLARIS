# controllers/slow_controller.py
from polaris.controllers.controller import BaseController

class SlowController(BaseController):
    def __init__(self):
        pass

    def decide_action(self, telemetry):
        # slower, stable reaction
        return {"type": "slow", "decision": "scale_down"}
