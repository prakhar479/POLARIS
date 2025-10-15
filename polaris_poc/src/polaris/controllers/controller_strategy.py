# strategy/controller_strategy.py
from .fast_controller import FastController
from .slow_controller import SlowController


class ControllerStrategy:
    def __init__(self, logger):
        # Initialize all strategies
        self.controllers = {"fast": FastController(logger), "slow": SlowController()}
        self.logger = logger
        self.current = self.controllers["fast"]  # default

    def select_controller(self, telemetry_data):
        # Example heuristic: select based on CPU trend
        response_times = []
        for event in telemetry_data["events"]:
            if event["name"] == "swim.average.response.time":
                response_times.append(event["value"])

        average_response_time = sum(response_times) / len(response_times)

        print(average_response_time)

        self.logger.info(
            f"Response time: {average_response_time}"
        )  # Log the response time for debugging
        RESP_TIME_THRESHOLD = 0.1

        if average_response_time > RESP_TIME_THRESHOLD:
            self.current = self.controllers["fast"]
        else:
            self.current = self.controllers["slow"]

        return self.current
