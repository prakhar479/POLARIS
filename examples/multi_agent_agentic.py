"""Example: Advanced Multi-Agent Committee with Iterative Tool-Use.

Demonstrates how each agent in the committee (Diagnostician, Planner, Validator) can
autonomously use tools like the Knowledge Store and World Model to perform deep
investigation, simulation, and safety verification.
"""

import asyncio
import os
from datetime import datetime, timezone

from polaris import Polaris
from polaris.connectors import SWIMConnector
from polaris.infrastructure.llm import create_llm_client
from polaris.infrastructure.observability import StructuredLogger
from polaris.strategies.multi_agent import AgentConfig, MultiAgentStrategy


async def main():
    """Run Polaris with iterative multi-agent committee."""
    # Check for API key (requires LLM for this example)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set")
        print("Set it with: export GOOGLE_API_KEY='your-key-here'")
        return

    # 1. Create a high-performance shared LLM client
    shared_llm = create_llm_client(
        "google", api_key=api_key, model="gemini-2.0-flash"  # Fast and capable
    )

    # 2. Configure specialized agents with unique tool access and reasoning depths

    # Diagnostician: Allowed many steps to investigate root causes using metric trends
    diagnostician_cfg = AgentConfig(
        temperature=0.0,
        steps_limit=5,
        allowed_tools=["get_recent_states", "summarize_metric_trends", "get_world_model_insights"],
    )

    # Planner: Allowed moderate steps to simulate outcomes of candidate plans
    planner_cfg = AgentConfig(
        temperature=0.4,
        steps_limit=3,
        allowed_tools=["predict_outcome", "list_supported_actions", "get_action_history"],
    )

    # Safety Validator: Fast check, focused on predicting stability impact
    validator_cfg = AgentConfig(
        temperature=0.0, steps_limit=2, allowed_tools=["predict_outcome", "get_action_history"]
    )

    # 3. Initialize the Multi-Agent Strategy
    strategy = MultiAgentStrategy(
        llm_client=shared_llm,
        system_description="Production e-commerce microservices (SWIM Simulation)",
        steps_limit=3,  # Strategy-level default
        diagnostician_config=diagnostician_cfg,
        planner_config=planner_cfg,
        validator_config=validator_cfg,
        temperature=0.1,
    )

    # 4. Create standard components
    logger = StructuredLogger(name="polaris", level="INFO")
    swim = SWIMConnector(host="localhost", port=4242)

    # 5. Assemble Polaris
    polaris = Polaris(connectors=[swim], strategy=strategy, logger=logger)

    print("Starting Advanced Multi-Agent Committee...")
    print(f"[{datetime.now(timezone.utc).isoformat()}] Initialization complete.")
    print("- Committee Strategy: 3 specialized agents")
    print("- Agentic Loops: Each agent can iteratively call tools")
    print("- Role Isolation: Agents have restricted toolsets for better focus")
    print("\nPress Ctrl+C to stop\n")

    try:
        await polaris.run()
    except KeyboardInterrupt:
        print("\nGracefully stopping committee...")
        await polaris.stop()


if __name__ == "__main__":
    asyncio.run(main())
