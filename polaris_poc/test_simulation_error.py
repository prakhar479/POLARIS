#!/usr/bin/env python3
"""
Test script to reproduce the "bad argument type for built-in operation" error
in the Bayesian World Model simulation.

This test recreates the exact scenario from the error logs to identify the root cause.
"""

import asyncio
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Add the src directory to the path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from polaris.models.bayesian_world_model import BayesianWorldModel
from polaris.models.world_model import SimulationRequest


async def test_simulation_error():
    """Test the exact simulation scenario that causes the error."""
    
    # Setup logging to see detailed debug info
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s'
    )
    logger = logging.getLogger("test_simulation_error")
    
    logger.info("Starting simulation error reproduction test")
    
    # Create a minimal config for the Bayesian World Model
    config = {
        "prediction_horizon_minutes": 60,
        "max_history_points": 1000,
        "correlation_threshold": 0.7,
        "anomaly_threshold": 4.0,
        "update_interval_seconds": 30,
        "process_noise": 0.01,
        "measurement_noise": 0.1,
        "initial_uncertainty": 1.0,
        "prior_confidence": 0.5,
        "learning_rate": 0.1,
        "switch_context": {
            "yolo_models": {
                "yolov5s": {
                    "expected_response_time": 0.2,
                    "expected_confidence": 0.75,
                    "expected_cpu_factor": 1.0
                },
                "yolov5m": {
                    "expected_response_time": 0.3,
                    "expected_confidence": 0.85,
                    "expected_cpu_factor": 1.5
                }
            }
        }
    }
    
    try:
        # Initialize the Bayesian World Model
        logger.info("Initializing Bayesian World Model")
        world_model = BayesianWorldModel(config, logger)
        await world_model.initialize()
        
        # Add some initial telemetry data to establish baseline
        logger.info("Adding initial telemetry data")
        from polaris.models.digital_twin_events import KnowledgeEvent
        
        # Add some metrics to the system state
        telemetry_events = [
            KnowledgeEvent(
                event_id="test-1",
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="test",
                event_type="telemetry",
                data={"name": "image_processing_time", "value": 0.25}
            ),
            KnowledgeEvent(
                event_id="test-2", 
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="test",
                event_type="telemetry",
                data={"name": "confidence", "value": 0.80}
            ),
            KnowledgeEvent(
                event_id="test-3",
                timestamp=datetime.now(timezone.utc).isoformat(), 
                source="test",
                event_type="telemetry",
                data={"name": "cpu_usage", "value": 50.0}
            )
        ]
        
        for event in telemetry_events:
            await world_model.update_state(event)
        
        logger.info("Initial telemetry data added successfully")
        
        # Create the exact simulation request from the error logs
        logger.info("Creating simulation request that matches the error scenario")
        
        # These are the exact actions from the error logs that cause the issue
        actions = [
            {
                'action_type': 'SWITCH_MODEL_YOLOV5S',
                'params': {'model': 'yolov5s'}  # This will be converted to string by gRPC
            },
            {
                'action_type': 'SWITCH_MODEL_YOLOV5M', 
                'params': {'model': 'yolov5m'}  # This will be converted to string by gRPC
            }
        ]
        
        # Simulate the gRPC string conversion that happens in the improved_grpc_client
        logger.info("Simulating gRPC string conversion of parameters")
        converted_actions = []
        for action in actions:
            converted_action = action.copy()
            if 'params' in converted_action:
                # Convert all params to strings (as gRPC does)
                converted_params = {}
                for k, v in converted_action['params'].items():
                    converted_params[k] = str(v)
                    logger.debug(f"Parameter conversion: {k}={v} ({type(v).__name__}) -> {str(v)} (str)")
                converted_action['params'] = converted_params
            converted_actions.append(converted_action)
        
        simulation_request = SimulationRequest(
            simulation_id="test-simulation-error",
            simulation_type="model_comparison",
            actions=converted_actions,
            horizon_minutes=15,
            parameters={}
        )
        
        logger.info(f"Simulation request created: {simulation_request.simulation_id}")
        logger.info(f"Actions: {simulation_request.actions}")
        
        # Run the simulation that should trigger the error
        logger.info("Running simulation that should trigger the error...")
        
        try:
            response = await world_model.simulate(simulation_request)
            
            if response.success:
                logger.info(f"Simulation completed successfully: {response.explanation}")
                logger.info(f"Confidence: {response.confidence}")
                logger.info(f"Future states count: {len(response.future_states)}")
            else:
                logger.error(f"Simulation failed: {response.explanation}")
                logger.error(f"Metadata: {response.metadata}")
                
        except Exception as e:
            logger.error(f"Simulation raised exception: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error("Full traceback:")
            traceback.print_exc()
            
            # Check if this is the specific error we're looking for
            if "bad argument type for built-in operation" in str(e):
                logger.error("*** FOUND THE TARGET ERROR! ***")
                logger.error("This is the 'bad argument type for built-in operation' error we're investigating")
                
                # Analyze the error context
                logger.error("Error analysis:")
                logger.error(f"- Error message: {str(e)}")
                logger.error(f"- Error type: {type(e).__name__}")
                
                # Try to identify where the error occurred
                tb = traceback.format_exc()
                logger.error("Traceback analysis:")
                for line in tb.split('\n'):
                    if 'bayesian_world_model.py' in line or 'bad argument type' in line:
                        logger.error(f"  -> {line}")
                
                return True  # Found the error
        
        await world_model.shutdown()
        return False  # Did not reproduce the error
        
    except Exception as e:
        logger.error(f"Test setup failed: {e}")
        traceback.print_exc()
        return False


async def test_type_conversion_issue():
    """Test specific type conversion scenarios that might cause the error."""
    
    logger = logging.getLogger("test_type_conversion")
    logger.info("Testing type conversion scenarios")
    
    # Test scenarios that might cause "bad argument type for built-in operation"
    test_cases = [
        # String arithmetic operations
        ("String + float", lambda: "1.5" + 2.0),
        ("String * int", lambda: "2" * 3.5),
        ("String - float", lambda: "5.0" - 1.0),
        ("String / int", lambda: "10" / 2),
        
        # NumPy operations with strings
        ("NumPy array with string", lambda: __import__('numpy').array(["1.0", "2.0"]) + 1.0),
        ("NumPy mean with mixed types", lambda: __import__('numpy').mean(["1.0", 2.0, 3.0])),
        
        # Mathematical functions with strings
        ("Math sqrt with string", lambda: __import__('math').sqrt("4.0")),
        ("Math pow with string", lambda: pow("2.0", 2)),
        
        # Comparison operations
        ("String comparison", lambda: "1.5" > 1.0),
        ("String min/max", lambda: min("1.0", 2.0, 3.0)),
    ]
    
    for test_name, test_func in test_cases:
        try:
            result = test_func()
            logger.info(f"✓ {test_name}: {result}")
        except Exception as e:
            logger.error(f"✗ {test_name}: {type(e).__name__}: {e}")
            if "bad argument type for built-in operation" in str(e):
                logger.error(f"*** FOUND TARGET ERROR in {test_name} ***")


async def main():
    """Main test function."""
    print("=" * 80)
    print("POLARIS Bayesian World Model Simulation Error Reproduction Test")
    print("=" * 80)
    
    # Test type conversion issues first
    print("\n1. Testing type conversion scenarios...")
    await test_type_conversion_issue()
    
    print("\n2. Testing simulation error reproduction...")
    error_reproduced = await test_simulation_error()
    
    print("\n" + "=" * 80)
    if error_reproduced:
        print("✓ SUCCESS: Reproduced the 'bad argument type for built-in operation' error")
        print("The error has been identified and can now be fixed.")
    else:
        print("✗ Could not reproduce the specific error")
        print("The error might be intermittent or require different conditions.")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())