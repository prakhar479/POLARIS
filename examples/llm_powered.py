"""
Example: Using LLM-powered meta-learner and strategy.

Shows advanced AI-powered adaptation using Google Gemini or OpenAI.
"""

import asyncio
import os

from polaris import Polaris
from polaris.connectors import SWIMConnector
from polaris.infrastructure.llm import create_llm_client
from polaris.infrastructure.observability import StructuredLogger
from polaris.meta_learner import LLMMetaLearner
from polaris.strategies import LLMReasoningStrategy


async def main():
    """Run Polaris with LLM-powered intelligence."""
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set")
        print("Set it with: export GOOGLE_API_KEY='your-key-here'")
        return

    # Create LLM client
    llm_client = create_llm_client("google", api_key=api_key, model="gemini-2.5-flash")

    # Create LLM-powered strategy
    strategy = LLMReasoningStrategy(
        llm_client=llm_client,
        system_description="SWIM web application server pool",
        adaptation_goals="Maintain performance with minimal resource usage",
        temperature=0.1,
    )

    # Create LLM meta-learner for autonomous optimization
    logger = StructuredLogger(name="polaris", level="INFO")

    meta_learner = LLMMetaLearner(
        llm_client=llm_client,
        knowledge_store=None,  # Will use default
        logger=logger,
        auto_apply=False,  # Require approval for safety
        temperature=0.1,
    )

    # Create SWIM connector
    swim = SWIMConnector(host="localhost", port=4242)

    # Create Polaris with AI components
    polaris = Polaris(
        connectors=[swim], strategy=strategy, meta_learner=meta_learner, logger=logger
    )

    print("Starting Polaris with AI-Powered Adaptation...")
    print("- LLM Strategy: Natural language reasoning")
    print("- LLM Meta-Learner: Intelligent parameter optimization")
    print("Press Ctrl+C to stop\n")

    try:
        await polaris.run()
    except KeyboardInterrupt:
        print("\nStopping...")
        await polaris.stop()


if __name__ == "__main__":
    asyncio.run(main())
