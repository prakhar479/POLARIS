#!/usr/bin/env python3
"""
Test script to verify the Bayesian World Model type error fixes.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the src directory to the path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import the fixed model
from polaris.models.world_model import SimulationRequest

# Import the fixed model by temporarily modifying the module
import importlib.util
fixed_model_path = Path(__file__).parent / "src" / "polaris" / "models" / "bayesian_world_model_fixed.py"
spec = importlib.util.spec_from_file_location("bayesian_world_model_fixed", fixed_model_path)
fixed_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixed_module)
BayesianWorldModel = fixed_module.BayesianWorldModel


async def test_fixed_model():
    """Test the fixed Bayesian World Model with string parameters."""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test_fixed_model")
    
    logger.info("Testing fixed Bayesian World Model with gRPC string parameters")
    
    # Create config
    config = {
        "prediction_horizon_minutes": 60,
        "switch_context": {
            "yolo_models": {
                "yolov5s": {"expected_response_time": 0.2, "expected_confidence": 0.75, "expected_cpu_factor": 1.0},
                "yolov5m": {"expected_response_time": 0.3, "expected_confidence": 0.85, "expected_cpu_factor": 1.5}
            }
        }
    }
    
    # Initialize model
    model = BayesianWorldModel(config, logger)
    await model.initialize()
    
    # Test with string parameters (simulating gRPC conversion)
    actions = [
        {
            'action_type': 'SWITCH_MODEL_YOLOV5M',
            'params': {'model': 'yolov5m'}  # String parameter
        },
        {
            'action_type': 'SET_DIMMER',
            'params': {'value': '0.8'}  # String parameter that should be float
        }
    ]
    
    # Create simulation request with string horizon_minutes (simulating gRPC)
    class MockSimulationRequest:
        def __init__(self):
            self.simulation_id = "test-fixed-model"
            self.simulation_type = "forecast"
            self.actions = actions
            self.horizon_minutes = "15"  # String instead of int (gRPC conversion)
            self.parameters = {}
    
    request = MockSimulationRequest()
    
    try:
        logger.info(f"Testing simulation with string horizon_minutes: {request.horizon_minutes} ({type(request.horizon_minutes).__name__})")
        
        response = await model.simulate(request)
        
        if response.success:
            logger.info("✓ SUCCESS: Fixed model handled string parameters correctly!")
            logger.info(f"  - Simulation completed: {response.explanation}")
            logger.info(f"  - Confidence: {response.confidence}")
            logger.info(f"  - Future states: {len(response.future_states)}")
            return True
        else:
            logger.error(f"✗ FAILED: Simulation failed: {response.explanation}")
            return False
            
    except Exception as e:
        logger.error(f"✗ FAILED: Exception occurred: {e}")
        return False
    
    finally:
        await model.shutdown()


async def main():
    """Main test function."""
    print("=" * 80)
    print("TESTING FIXED BAYESIAN WORLD MODEL")
    print("=" * 80)
    
    success = await test_fixed_model()
    
    print("\n" + "=" * 80)
    if success:
        print("✓ ALL TESTS PASSED: The type error fixes are working correctly!")
        print("The 'bad argument type for built-in operation' error should be resolved.")
    else:
        print("✗ TESTS FAILED: The fixes need further refinement.")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
