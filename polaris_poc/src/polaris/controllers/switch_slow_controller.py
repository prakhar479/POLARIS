"""
Slow Controller for SWITCH System.

Placeholder for LLM-based reasoner integration. The actual reasoning
is performed by the reasoner agent when telemetry is delegated via
the 'polaris.reasoner.kernel.requests' NATS subject.
"""

from polaris.controllers.controller import BaseController


class SwitchSlowController(BaseController):
    """
    Slow controller for SWITCH ML system - Reasoner placeholder.
    
    This controller serves as a marker to indicate that telemetry should be
    routed to the reasoner agent for deep LLM-based optimization analysis.
    The actual reasoning logic is implemented in the reasoner agent with
    SWITCH-specific prompts and knowledge base integration.
    """
    
    def __init__(self):
        """Initialize SWITCH slow controller."""
        super().__init__()
    
    def decide_action(self, telemetry):
        """
        This method should not be called directly.
        
        When SlowController is selected, the kernel delegates telemetry
        to the reasoner agent via NATS for processing.
        
        Args:
            telemetry: Telemetry data (not used here)
            
        Returns:
            None (reasoner handles action generation)
        """
        return None
