from abc import ABC, abstractmethod

class ReasonerAgent(ABC):
    @abstractmethod
    def publish(self, topic: str, message: dict) -> None:
        """
        Publish a message to a specific topic.
        
        Args:
            topic (str): The topic to publish to.
            message (dict): The message to publish.
        """
        pass

    @abstractmethod
    def listen(self, topic: str) -> dict:
        """
        Listen for messages on a specific topic.
        
        Args:
            topic (str): The topic to listen to.
        
        Returns:
            dict: The received message.
        """
        pass

    @abstractmethod
    def predict(self, input_data: dict) -> dict:
        """
        Make a prediction based on the input data.
        
        Args:
            input_data (dict): The input data for prediction.
        
        Returns:
            dict: The prediction result.
        """
        pass

class TestReasoner(ReasonerAgent):
    def publish(self, topic: str, message: dict) -> None:
        # Implementation for publishing a message
        pass

    def listen(self, topic: str) -> dict:
        # Implementation for listening to a topic
        pass

    def predict(self, input_data: dict) -> dict:
        # Implementation for making predictions
        pass 