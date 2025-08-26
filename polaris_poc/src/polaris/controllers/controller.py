# Base class
class BaseController:
    def __init__(self, kernel):
        """
        Initialize the controller with a reference to the kernel.
        """
        self.kernel = kernel

    def decide_action(self, telemetry):
        """
        Decide an action based on the telemetry data.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")
