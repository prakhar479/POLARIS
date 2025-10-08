#!/usr/bin/env python3
"""
Test script for the Agentic Reasoner

This script tests the core functionality of the agentic reasoner
without requiring the full POLARIS infrastructure.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from polaris.agents.agentic_reasoner import (
    AgenticLLMReasoner,
    KnowledgeBaseTool,
    DigitalTwinTool,
)
from polaris.agents.reasoner_core import ReasoningContext, ReasoningType


class MockKnowledgeQuery:
    """Mock knowledge base query interface."""
    
    async def query_structured(self, data_types, filters=None, limit=10, min_score=0.0):
        """Mock structured query."""
        return [
            {
                "timestamp": "2024-01-01T12:00:00Z",
                "content": {
                    "content": {
                        "current_state": {
                            "average_response_time": 0.8,
                            "utilization": 0.65,
                            "active_servers": 3,
                            "dimmer": 0.9
                        }
                    }
                },
                "source": "swim_snapshotter",
                "tags": ["snapshot"]
            }
        ]
    
    async def query_natural_language(self, query_text, limit=10):
        """Mock natural language query."""
        return [{"result": f"Mock result for: {query_text}"}]
    
    async def query_recent_observations(self, limit=10):
        """Mock recent observations query."""
        return [{"observation": "System running normally"}]
    
    async def query_raw_telemetry(self, limit=10):
        """Mock raw telemetry query."""
        return [{"metric": "cpu_usage", "value": 65.0}]


class MockDigitalTwin:
    """Mock digital twin interface."""
    
    async def connect(self):
        """Mock connect."""
        pass
    
    async def disconnect(self):
        """Mock disconnect."""
        pass
    
    async def query(self, query):
        """Mock query."""
        from polaris.agents.reasoner_agent import DTResponse
        return DTResponse(
            success=True,
            result={"system_state": "healthy", "prediction": "stable"},
            confidence=0.85,
            explanation="System is operating within normal parameters"
        )
    
    async def simulate(self, simulation):
        """Mock simulation."""
        from polaris.agents.reasoner_agent import DTResponse
        return DTResponse(
            success=True,
            confidence=0.9,
            explanation="Simulation completed successfully",
            future_states=[
                {"time": "+5min", "response_time": 0.7, "utilization": 0.6},
                {"time": "+10min", "response_time": 0.65, "utilization": 0.58}
            ]
        )
    
    async def diagnose(self, diagnosis):
        """Mock diagnosis."""
        from polaris.agents.reasoner_agent import DTResponse
        return DTResponse(
            success=True,
            confidence=0.8,
            explanation="Diagnostic analysis completed",
            hypotheses=[
                {"hypothesis": "Normal operation", "probability": 0.8},
                {"hypothesis": "Minor performance degradation", "probability": 0.2}
            ]
        )


class MockLLMReasoner(AgenticLLMReasoner):
    """Mock LLM reasoner for testing."""
    
    def __init__(self, *args, **kwargs):
        # Override the LLM call to avoid actual API calls
        super().__init__(*args, **kwargs)
        self.mock_responses = []
        self.call_count = 0
    
    def set_mock_responses(self, responses):
        """Set mock responses for LLM calls."""
        self.mock_responses = responses
        self.call_count = 0
    
    async def _call_llm_with_history(self, conversation_history):
        """Mock LLM call."""
        if self.call_count < len(self.mock_responses):
            response = self.mock_responses[self.call_count]
            self.call_count += 1
            return response
        else:
            # Default response if no more mock responses
            return """
Analysis: The system appears to be operating normally based on the telemetry data.

Decision: No immediate action is required as all metrics are within acceptable ranges.

Action:
{
  "action_type": "NO_ACTION",
  "source": "agentic_reasoner",
  "action_id": "test-uuid",
  "params": {},
  "priority": "low",
  "reasoning": "System operating within normal parameters"
}
"""


async def test_knowledge_base_tool():
    """Test the knowledge base tool."""
    print("🧪 Testing Knowledge Base Tool...")
    
    logger = logging.getLogger("test")
    kb_query = MockKnowledgeQuery()
    tool = KnowledgeBaseTool(kb_query, logger)
    
    # Test structured query
    result = await tool.execute(
        query_type="structured",
        data_types=["observation"],
        filters={"source": "test"},
        limit=5
    )
    
    assert result["success"] == True
    assert len(result["results"]) > 0
    print("✅ Knowledge Base Tool test passed")


async def test_digital_twin_tool():
    """Test the digital twin tool."""
    print("🧪 Testing Digital Twin Tool...")
    
    logger = logging.getLogger("test")
    dt_interface = MockDigitalTwin()
    tool = DigitalTwinTool(dt_interface, logger)
    
    # Test query operation
    result = await tool.execute(
        operation="query",
        query_type="current_state",
        query_content="Get system overview"
    )
    
    assert result["success"] == True
    assert result["confidence"] > 0
    print("✅ Digital Twin Tool test passed")


async def test_agentic_reasoner_simple():
    """Test the agentic reasoner with a simple scenario."""
    print("🧪 Testing Agentic Reasoner - Simple Scenario...")
    
    logger = logging.getLogger("test")
    kb_query = MockKnowledgeQuery()
    dt_interface = MockDigitalTwin()
    
    # Create mock reasoner
    reasoner = MockLLMReasoner(
        api_key="test-key",
        reasoning_type=ReasoningType.DECISION,
        kb_query_interface=kb_query,
        dt_interface=dt_interface,
        logger=logger
    )
    
    # Set mock LLM response
    reasoner.set_mock_responses([
        """
Analysis: System metrics show normal operation with good response time and utilization.

Decision: No action needed as system is performing optimally.

{
  "action_type": "NO_ACTION",
  "source": "agentic_reasoner",
  "action_id": "test-uuid-1",
  "params": {},
  "priority": "low",
  "reasoning": "System operating optimally"
}
"""
    ])
    
    # Create reasoning context
    context = ReasoningContext(
        session_id="test-session-1",
        reasoning_type=ReasoningType.DECISION,
        input_data={
            "events": [
                {"name": "average_response_time", "value": 0.8},
                {"name": "utilization", "value": 0.65},
                {"name": "active_servers", "value": 3}
            ]
        }
    )
    
    # Execute reasoning
    result = await reasoner.reason(context)
    
    assert result.result["action_type"] == "NO_ACTION"
    assert result.confidence > 0
    print("✅ Simple Agentic Reasoner test passed")


async def test_agentic_reasoner_with_tools():
    """Test the agentic reasoner with tool usage."""
    print("🧪 Testing Agentic Reasoner - With Tool Usage...")
    
    logger = logging.getLogger("test")
    kb_query = MockKnowledgeQuery()
    dt_interface = MockDigitalTwin()
    
    # Create mock reasoner
    reasoner = MockLLMReasoner(
        api_key="test-key",
        reasoning_type=ReasoningType.DECISION,
        kb_query_interface=kb_query,
        dt_interface=dt_interface,
        logger=logger
    )
    
    # Set mock LLM responses for multi-step reasoning
    reasoner.set_mock_responses([
        # First response - requests tool usage
        """
Analysis: High response time detected. Need to check historical trends.

TOOL_CALL: query_knowledge_base
PARAMETERS: {"query_type": "structured", "data_types": ["observation"], "limit": 10}
""",
        # Second response - after tool results
        """
Analysis: Based on historical data, this is a significant increase in response time.

Decision: Need to add a server to handle the increased load.

{
  "action_type": "ADD_SERVER",
  "source": "agentic_reasoner",
  "action_id": "test-uuid-2",
  "params": {"server_type": "compute", "count": 1},
  "priority": "high",
  "reasoning": "High response time requires additional capacity"
}
"""
    ])
    
    # Create reasoning context with high response time
    context = ReasoningContext(
        session_id="test-session-2",
        reasoning_type=ReasoningType.DECISION,
        input_data={
            "events": [
                {"name": "average_response_time", "value": 2.0},  # High response time
                {"name": "utilization", "value": 0.95},
                {"name": "active_servers", "value": 2}
            ]
        }
    )
    
    # Execute reasoning
    result = await reasoner.reason(context)
    
    assert result.result["action_type"] == "ADD_SERVER"
    assert result.result["priority"] == "high"
    assert result.kb_queries_made > 0  # Should have made tool calls
    print("✅ Agentic Reasoner with tools test passed")


async def test_tool_parsing():
    """Test the tool call parsing functionality."""
    print("🧪 Testing Tool Call Parsing...")
    
    logger = logging.getLogger("test")
    reasoner = MockLLMReasoner(
        api_key="test-key",
        reasoning_type=ReasoningType.DECISION,
        logger=logger
    )
    
    # Test parsing tool calls
    response = """
Analysis: Need more information about the system.

TOOL_CALL: query_knowledge_base
PARAMETERS: {"query_type": "structured", "data_types": ["observation"]}

TOOL_CALL: query_digital_twin
PARAMETERS: {"operation": "query", "query_type": "current_state"}
"""
    
    tool_calls, action = reasoner._parse_llm_response(response)
    
    assert len(tool_calls) == 2
    assert tool_calls[0]["tool_name"] == "query_knowledge_base"
    assert tool_calls[1]["tool_name"] == "query_digital_twin"
    assert action is None  # No action in this response
    
    # Test parsing action
    response_with_action = """
Decision: Add a server.

{
  "action_type": "ADD_SERVER",
  "source": "agentic_reasoner",
  "params": {"count": 1}
}
"""
    
    tool_calls, action = reasoner._parse_llm_response(response_with_action)
    
    assert len(tool_calls) == 0
    assert action is not None
    assert action["action_type"] == "ADD_SERVER"
    
    print("✅ Tool call parsing test passed")


async def run_all_tests():
    """Run all tests."""
    print("🚀 Starting Agentic Reasoner Tests")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests
    
    tests = [
        test_knowledge_base_tool,
        test_digital_twin_tool,
        test_tool_parsing,
        test_agentic_reasoner_simple,
        test_agentic_reasoner_with_tools,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("💥 Some tests failed!")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)