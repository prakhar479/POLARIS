#!/usr/bin/env python3
"""
Test script to verify the type error fixes work by directly testing the problematic functions.
"""

import asyncio
import logging
import sys
import traceback
from pathlib import Path

# Add the src directory to the path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from polaris.models.bayesian_world_model import BayesianWorldModel
from polaris.models.world_model import SimulationRequest
from polaris.models.digital_twin_events import KnowledgeEvent
from datetime import datetime, timezone


async def test_original_model_with_string_params():
    """Test the original model to see if it fails with string parameters."""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test_original_model")
    
    logger.info("Testing original Bayesian World Model with string parameters (should fail)")
    
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
    
    # Add some telemetry data
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
        await model.update_state(event)
    
    # Test with string parameters (simulating gRPC conversion)
    actions = [
        {
            'action_type': 'SWITCH_MODEL_YOLOV5M',
            'params': {'model': 'yolov5m'}
        },
        {
            'action_type': 'SET_DIMMER',
            'params': {'value': '0.8'}  # String parameter that should be float
        }
    ]
    
    # Create a mock simulation request that simulates the gRPC string conversion issue
    class MockSimulationRequest:
        def __init__(self):
            self.simulation_id = "test-string-params"
            self.simulation_type = "forecast"
            self.actions = actions
            self.horizon_minutes = "15"  # String instead of int (this causes the error!)
            self.parameters = {}
    
    request = MockSimulationRequest()
    
    try:
        logger.info(f"Testing with string horizon_minutes: '{request.horizon_minutes}' ({type(request.horizon_minutes).__name__})")
        
        response = await model.simulate(request)
        
        if response.success:
            logger.info("✓ Model handled string parameters correctly (unexpected)")
            return True
        else:
            logger.error(f"✗ Simulation failed: {response.explanation}")
            if "bad argument type for built-in operation" in response.explanation:
                logger.error("*** CONFIRMED: Found the target error! ***")
                return "error_found"
            return False
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"✗ Exception occurred: {error_msg}")
        
        if "bad argument type for built-in operation" in error_msg:
            logger.error("*** CONFIRMED: Found the target error in exception! ***")
            return "error_found"
        elif any(keyword in error_msg.lower() for keyword in ['unsupported operand', 'can only concatenate', "can't multiply"]):
            logger.error("*** CONFIRMED: Found related type error! ***")
            return "error_found"
        
        traceback.print_exc()
        return False
    
    finally:
        await model.shutdown()


def test_safe_conversion_functions():
    """Test the safe conversion functions that should fix the issue."""
    
    logger = logging.getLogger("test_safe_conversion")
    logger.info("Testing safe conversion functions")
    
    # Create a minimal model instance to test the conversion methods
    config = {"prediction_horizon_minutes": 60}
    model = BayesianWorldModel(config, logger)
    
    # Test _safe_float_conversion
    test_cases = [
        ("String float", "1.5", 1.5),
        ("String int", "42", 42.0),
        ("Integer", 10, 10.0),
        ("Float", 3.14, 3.14),
        ("Invalid string", "not_a_number", 0.0),
        ("None", None, 0.0),
    ]
    
    logger.info("Testing _safe_float_conversion:")
    all_passed = True
    
    for name, input_val, expected in test_cases:
        try:
            result = model._safe_float_conversion(input_val, 0.0, name)
            if abs(result - expected) < 0.001:  # Float comparison with tolerance
                logger.info(f"  ✓ {name}: {input_val} -> {result}")
            else:
                logger.error(f"  ✗ {name}: {input_val} -> {result} (expected {expected})")
                all_passed = False
        except Exception as e:
            logger.error(f"  ✗ {name}: {input_val} -> ERROR: {e}")
            all_passed = False
    
    return all_passed


def apply_inline_fixes():
    """Apply the fixes directly to the original model for testing."""
    
    logger = logging.getLogger("apply_fixes")
    logger.info("Applying inline fixes to test the solution")
    
    # Monkey patch the BayesianWorldModel to add the missing _safe_int_conversion method
    def _safe_int_conversion(self, value, default=0, param_name="unknown"):
        """Safely convert a parameter value to int, handling string conversions from gRPC."""
        try:
            if isinstance(value, int):
                return value
            elif isinstance(value, (float, str)):
                return int(float(value))  # Convert via float to handle "15.0" strings
            else:
                self.logger.warning(f"Unexpected type for parameter '{param_name}': {type(value).__name__}, using default {default}")
                return default
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Failed to convert parameter '{param_name}' value '{value}' to int: {e}, using default {default}")
            return default
    
    # Add the method to the class
    BayesianWorldModel._safe_int_conversion = _safe_int_conversion
    
    # Monkey patch the simulate method to use safe conversion
    original_simulate = BayesianWorldModel.simulate
    
    async def patched_simulate(self, request):
        """Patched simulate method with safe horizon_minutes conversion."""
        # Safely convert horizon_minutes from potential string (gRPC conversion)
        if hasattr(request, 'horizon_minutes'):
            original_horizon = request.horizon_minutes
            request.horizon_minutes = self._safe_int_conversion(request.horizon_minutes, 60, "horizon_minutes")
            self.logger.debug(f"Converted horizon_minutes: {original_horizon} ({type(original_horizon).__name__}) -> {request.horizon_minutes} (int)")
        
        return await original_simulate(self, request)
    
    BayesianWorldModel.simulate = patched_simulate
    
    logger.info("✓ Applied inline fixes to BayesianWorldModel")
    return True


async def test_fixed_model():
    """Test the model after applying fixes."""
    
    logger = logging.getLogger("test_fixed_model")
    logger.info("Testing Bayesian World Model after applying fixes")
    
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
    
    # Add some telemetry data
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
        )
    ]
    
    for event in telemetry_events:
        await model.update_state(event)
    
    # Test with string parameters (simulating gRPC conversion)
    actions = [
        {
            'action_type': 'SWITCH_MODEL_YOLOV5M',
            'params': {'model': 'yolov5m'}
        }
    ]
    
    # Create simulation request with string horizon_minutes
    class MockSimulationRequest:
        def __init__(self):
            self.simulation_id = "test-fixed-model"
            self.simulation_type = "forecast"
            self.actions = actions
            self.horizon_minutes = "15"  # String instead of int
            self.parameters = {}
    
    request = MockSimulationRequest()
    
    try:
        logger.info(f"Testing fixed model with string horizon_minutes: '{request.horizon_minutes}' ({type(request.horizon_minutes).__name__})")
        
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
        traceback.print_exc()
        return False
    
    finally:
        await model.shutdown()


async def main():
    """Main test function."""
    print("=" * 80)
    print("BAYESIAN WORLD MODEL TYPE ERROR FIX VERIFICATION")
    print("=" * 80)
    
    # Test 1: Verify the safe conversion functions work
    print("\n1. Testing safe conversion functions...")
    conversion_works = test_safe_conversion_functions()
    
    # Test 2: Test original model to confirm the error
    print("\n2. Testing original model with string parameters...")
    original_result = await test_original_model_with_string_params()
    
    # Test 3: Apply fixes and test again
    print("\n3. Applying fixes and testing...")
    apply_inline_fixes()
    fixed_result = await test_fixed_model()
    
    print("\n" + "=" * 80)
    print("TEST RESULTS:")
    print("=" * 80)
    
    if conversion_works:
        print("✓ Safe conversion functions work correctly")
    else:
        print("✗ Safe conversion functions have issues")
    
    if original_result == "error_found":
        print("✓ Confirmed the original error exists")
    elif original_result:
        print("? Original model worked unexpectedly")
    else:
        print("✗ Could not reproduce the original error")
    
    if fixed_result:
        print("✓ Fixed model works correctly with string parameters")
        print("\n🎉 SUCCESS: The type error fix is working!")
        print("The 'bad argument type for built-in operation' error should be resolved.")
    else:
        print("✗ Fixed model still has issues")
        print("\n❌ The fix needs more work.")
    
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())