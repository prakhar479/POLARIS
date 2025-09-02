# Base class
class BaseController:
    def __init__(self):
        """
        Initialize the controller with a reference to the kernel.
        """
        pass

    def decide_action(self, telemetry):
        """
        Decide an action based on the telemetry data.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")
