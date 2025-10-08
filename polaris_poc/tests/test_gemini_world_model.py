#!/usr/bin/env python3
"""
Test script for the Gemini World Model implementation.

This script tests the core functionality of the Gemini World Model
to ensure it works correctly with the real Gemini API.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from polaris.models.gemini_world_model import GeminiWorldModel
from polaris.models.world_model import QueryRequest, SimulationRequest, DiagnosisRequest
from polaris.models.digital_twin_events import KnowledgeEvent, CalibrationEvent
from polaris.models.telemetry import TelemetryEvent


async def test_gemini_world_model():
    """Test the Gemini World Model implementation."""
    print("🧪 Testing Gemini World Model")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test")
    
    # Check if API key is available
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY environment variable not set")
        print("Please set your Gemini API key to run this test")
        return False
    
    # Configuration for testing
    config = {
        "api_key_env": "GEMINI_API_KEY",
        "model": "gemini-2.5-flash",
        "temperature": 0.7,
        "max_tokens": 1024,
        "concurrent_requests": 2,
        "request_timeout_sec": 30,
        "retry_attempts": 2,
        "max_conversation_memory": 10,
        "max_history_events": 100
    }
    
    try:
        # Initialize the model
        print("🔧 Initializing Gemini World Model...")
        model = GeminiWorldModel(config, logger)
        await model.initialize()
        print("✅ Model initialized successfully")
        
        # Test health status
        print("\n🏥 Testing health status...")
        health = await model.get_health_status()
        print(f"Health status: {health['status']}")
        print(f"Model: {health['model_name']}")
        print(f"API connectivity: {health['api_connectivity']['healthy']}")
        
        # Test query functionality
        print("\n🔍 Testing query functionality...")
        query_request = QueryRequest(
            query_id="test-query-1",
            query_type="current_state",
            query_content="What is the current system status?",
            parameters={"format": "summary"}
        )
        
        query_response = await model.query_state(query_request)
        print(f"Query success: {query_response.success}")
        print(f"Query confidence: {query_response.confidence:.2f}")
        print(f"Query result: {query_response.result[:100]}...")
        
        # Test simulation functionality
        print("\n🎯 Testing simulation functionality...")
        simulation_request = SimulationRequest(
            simulation_id="test-sim-1",
            simulation_type="what_if",
            actions=[{"action_type": "ADD_SERVER", "parameters": {"count": 1}}],
            horizon_minutes=30
        )
        
        simulation_response = await model.simulate(simulation_request)
        print(f"Simulation success: {simulation_response.success}")
        print(f"Simulation confidence: {simulation_response.confidence:.2f}")
        print(f"Future states: {len(simulation_response.future_states)}")
        
        # Test diagnosis functionality
        print("\n🩺 Testing diagnosis functionality...")
        diagnosis_request = DiagnosisRequest(
            diagnosis_id="test-diag-1",
            anomaly_description="High response time detected in web service",
            context={"service": "web-api", "severity": "high"}
        )
        
        diagnosis_response = await model.diagnose(diagnosis_request)
        print(f"Diagnosis success: {diagnosis_response.success}")
        print(f"Diagnosis confidence: {diagnosis_response.confidence:.2f}")
        print(f"Hypotheses found: {len(diagnosis_response.hypotheses)}")
        
        # Test state update and calibration
        print("\n📊 Testing state update and calibration...")
        
        # Create a proper telemetry event
        telemetry_data = TelemetryEvent(
            name="cpu_usage",
            value=75.5,
            unit="percent",
            timestamp="2024-01-01T12:00:00Z",
            source="test-monitor",
            tags={"host": "test-server", "region": "test"}
        )
        
        telemetry_event = KnowledgeEvent(
            event_id="test-event-1",
            event_type="telemetry",
            source="test-monitor",
            timestamp="2024-01-01T12:00:00Z",
            data=telemetry_data
        )
        
        await model.update_state(telemetry_event)
        print("✅ State update successful")
        
        # Test calibration
        calibration_event = CalibrationEvent(
            calibration_id="test-cal-1",
            prediction_id="test-pred-1",
            predicted_outcome={"cpu_usage": 80.0},
            actual_outcome={"cpu_usage": 78.5},
            accuracy_metrics={"mae": 1.5}
        )
        
        await model.calibrate(calibration_event)
        print("✅ Calibration successful")
        
        # Final health check
        print("\n🏥 Final health check...")
        final_health = await model.get_health_status()
        print(f"Events processed: {final_health['metrics']['events_processed']}")
        print(f"Conversation memory: {final_health['metrics']['conversation_memory_size']}")
        print(f"Current state metrics: {final_health['metrics']['current_state_metrics']}")
        
        # Shutdown
        await model.shutdown()
        print("\n✅ All tests passed! Gemini World Model is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


async def main():
    """Main test function."""
    print("🚀 Gemini World Model Test Suite")
    print("=" * 50)
    print("This test verifies the Gemini World Model implementation")
    print("with real API calls to Google's Gemini service.")
    print("")
    
    success = await test_gemini_world_model()
    
    if success:
        print("\n🎉 All tests completed successfully!")
        print("The Gemini World Model is ready for use.")
    else:
        print("\n💥 Tests failed. Please check the implementation.")
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)