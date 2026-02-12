"""
Example: Creating a custom adaptation strategy.

Shows how to implement custom adaptation logic.
"""

import asyncio
from typing import Optional

from polaris import AdaptationContext, AdaptationStrategy, ParameterSpec, Polaris
from polaris.connectors import SWIMConnector
from polaris.core.models import AdaptationAction, SystemState


class CustomStrategy(AdaptationStrategy):
    """
    Custom strategy that adapts based on custom logic.

    Example: Proactive scaling based on time of day.
    """

    def __init__(self):
        """Initialize the custom strategy."""
        self.peak_hours = [9, 10, 11, 12, 13, 14, 15, 16, 17]  # 9 AM - 5 PM
        self.adaptation_count = 0

    async def assess(
        self, state: SystemState, context: AdaptationContext
    ) -> Optional[AdaptationAction]:
        """Assess system state and recommend adaptation."""
        # Get current hour
        current_hour = state.timestamp.hour

        # Check if response time metric exists
        if "response_time" not in state.metrics:
            return None

        response_time = float(state.metrics["response_time"].value)

        # During peak hours, be more aggressive
        if current_hour in self.peak_hours:
            if response_time > 300:  # Lower threshold during peak
                return AdaptationAction(
                    action_id=f"custom_{self.adaptation_count}",
                    action_type="scale_up",
                    target_system=state.system_id,
                    parameters={"reason": "peak_hour_high_response"},
                )
        else:
            # Off-peak hours, higher threshold
            if response_time > 600:
                return AdaptationAction(
                    action_id=f"custom_{self.adaptation_count}",
                    action_type="scale_up",
                    target_system=state.system_id,
                    parameters={"reason": "off_peak_high_response"},
                )

        return None

    def get_tunable_parameters(self):
        """Define tunable parameters."""
        return {
            "peak_response_threshold": ParameterSpec(
                current_value=300,
                type=float,
                min_value=100,
                max_value=1000,
                description="Response time threshold during peak hours",
            )
        }

    async def update_parameter(self, parameter_path: str, new_value) -> bool:
        """Update parameters (simplified for example)."""
        return True


async def main():
    """Run with custom strategy."""
    swim = SWIMConnector()
    custom_strategy = CustomStrategy()

    polaris = Polaris(connectors=[swim], strategy=custom_strategy)

    print("Running Polaris with custom strategy")
    print("Strategy: Time-aware adaptive thresholds\n")

    try:
        await polaris.run()
    except KeyboardInterrupt:
        await polaris.stop()


if __name__ == "__main__":
    asyncio.run(main())
