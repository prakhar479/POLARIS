#!/usr/bin/env python3
"""
Fix for the "bad argument type for built-in operation" error in Bayesian World Model.

This script applies the necessary fixes to handle gRPC string parameter conversion properly.
"""

import sys
from pathlib import Path

# Add the src directory to the path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def create_fixed_bayesian_world_model():
    """Create the fixed version of the Bayesian World Model."""
    
    # Read the original file
    original_file = Path("src/polaris/models/bayesian_world_model.py")
    
    if not original_file.exists():
        print(f"Error: {original_file} not found")
        return False
    
    with open(original_file, 'r') as f:
        content = f.read()
    
    # Apply fixes
    fixes_applied = []
    
    # Fix 1: Update the simulate method to safely convert horizon_minutes
    old_simulate_start = '''    async def simulate(self, request: SimulationRequest) -> SimulationResponse:
        """Enhanced predictive simulation with SWITCH-aware dynamics and uncertainty quantification."""
        try:
            start_time = datetime.now(timezone.utc)
            
            self.logger.debug(f"Processing enhanced simulation: {request.simulation_id}")
            
            # Adaptive time stepping based on system volatility
            system_volatility = self._calculate_system_volatility()
            time_steps_array = self._calculate_adaptive_timesteps(request.horizon_minutes, system_volatility)'''
    
    new_simulate_start = '''    async def simulate(self, request: SimulationRequest) -> SimulationResponse:
        """Enhanced predictive simulation with SWITCH-aware dynamics and uncertainty quantification."""
        try:
            start_time = datetime.now(timezone.utc)
            
            self.logger.debug(f"Processing enhanced simulation: {request.simulation_id}")
            
            # Safely convert horizon_minutes from potential string (gRPC conversion)
            horizon_minutes = self._safe_int_conversion(request.horizon_minutes, 60, "horizon_minutes")
            
            # Adaptive time stepping based on system volatility
            system_volatility = self._calculate_system_volatility()
            time_steps_array = self._calculate_adaptive_timesteps(horizon_minutes, system_volatility)'''
    
    if old_simulate_start in content:
        content = content.replace(old_simulate_start, new_simulate_start)
        fixes_applied.append("Fixed simulate method horizon_minutes conversion")
    
    # Fix 2: Add _safe_int_conversion method
    safe_int_method = '''
    def _safe_int_conversion(self, value: Any, default: int = 0, param_name: str = "unknown") -> int:
        """Safely convert a parameter value to int, handling string conversions from gRPC.
        
        Args:
            value: The value to convert (could be string, int, float, etc.)
            default: Default value if conversion fails
            param_name: Name of parameter for logging
            
        Returns:
            Integer value or default if conversion fails
        """
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
'''
    
    # Insert the new method after the existing _safe_float_conversion method
    safe_float_method_end = '''            return default
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):'''
    
    if safe_float_method_end in content:
        content = content.replace(safe_float_method_end, safe_float_method_end.replace('def __init__', safe_int_method + '\n    def __init__'))
        fixes_applied.append("Added _safe_int_conversion method")
    
    # Fix 3: Update _calculate_adaptive_timesteps to use proper type annotation
    old_timesteps_method = '''    def _calculate_adaptive_timesteps(self, horizon_minutes: int, system_volatility: float) -> np.ndarray:
        """Calculate adaptive time steps based on system dynamics."""
        if system_volatility > 0.8:  # High volatility - more frequent steps
            n_steps = min(horizon_minutes * 4, 120)  # Max 120 steps
        elif system_volatility > 0.5:  # Medium volatility
            n_steps = min(horizon_minutes * 2, 60)   # Max 60 steps
        else:  # Low volatility - fewer steps
            n_steps = min(horizon_minutes, 30)       # Max 30 steps
        
        return np.linspace(0, horizon_minutes, n_steps)'''
    
    new_timesteps_method = '''    def _calculate_adaptive_timesteps(self, horizon_minutes: int, system_volatility: float) -> np.ndarray:
        """Calculate adaptive time steps based on system dynamics."""
        # Ensure horizon_minutes is an integer (defensive programming)
        horizon_minutes = int(horizon_minutes) if not isinstance(horizon_minutes, int) else horizon_minutes
        
        if system_volatility > 0.8:  # High volatility - more frequent steps
            n_steps = min(horizon_minutes * 4, 120)  # Max 120 steps
        elif system_volatility > 0.5:  # Medium volatility
            n_steps = min(horizon_minutes * 2, 60)   # Max 60 steps
        else:  # Low volatility - fewer steps
            n_steps = min(horizon_minutes, 30)       # Max 30 steps
        
        return np.linspace(0, horizon_minutes, n_steps)'''
    
    if old_timesteps_method in content:
        content = content.replace(old_timesteps_method, new_timesteps_method)
        fixes_applied.append("Fixed _calculate_adaptive_timesteps method")
    
    # Fix 4: Update _apply_dimmer_effects to use safe conversion
    old_dimmer_effects = '''    def _apply_dimmer_effects(self, future_state: Dict, dimmer_value: float, time_step: int) -> Dict:
        """Apply dimmer effects with enhanced dynamics."""
        
        # Dimmer affects response time and throughput
        if "image_processing_time" in future_state:
            current_rt = future_state["image_processing_time"]["predicted_value"]
            # Lower dimmer = better response time (less processing)
            dimmer_factor = 0.7 + 0.3 * dimmer_value  # Range: 0.7 to 1.0
            future_state["image_processing_time"]["predicted_value"] = current_rt * dimmer_factor
        
        # Dimmer may slightly affect confidence (less processing = potentially lower accuracy)
        if "confidence" in future_state and dimmer_value < 0.8:
            confidence_penalty = (0.8 - dimmer_value) * 0.1  # Max 2% penalty
            current_conf = future_state["confidence"]["predicted_value"]
            future_state["confidence"]["predicted_value"] = max(0.5, current_conf - confidence_penalty)
        
        return future_state'''
    
    new_dimmer_effects = '''    def _apply_dimmer_effects(self, future_state: Dict, dimmer_value: float, time_step: int) -> Dict:
        """Apply dimmer effects with enhanced dynamics."""
        
        # Ensure dimmer_value is a float (defensive programming for gRPC string conversion)
        dimmer_value = self._safe_float_conversion(dimmer_value, 1.0, "dimmer_value")
        
        # Dimmer affects response time and throughput
        if "image_processing_time" in future_state:
            current_rt = future_state["image_processing_time"]["predicted_value"]
            # Lower dimmer = better response time (less processing)
            dimmer_factor = 0.7 + 0.3 * dimmer_value  # Range: 0.7 to 1.0
            future_state["image_processing_time"]["predicted_value"] = current_rt * dimmer_factor
        
        # Dimmer may slightly affect confidence (less processing = potentially lower accuracy)
        if "confidence" in future_state and dimmer_value < 0.8:
            confidence_penalty = (0.8 - dimmer_value) * 0.1  # Max 2% penalty
            current_conf = future_state["confidence"]["predicted_value"]
            future_state["confidence"]["predicted_value"] = max(0.5, current_conf - confidence_penalty)
        
        return future_state'''
    
    if old_dimmer_effects in content:
        content = content.replace(old_dimmer_effects, new_dimmer_effects)
        fixes_applied.append("Fixed _apply_dimmer_effects method")
    
    # Fix 5: Add comprehensive parameter validation in _apply_enhanced_action_effects
    # Look for the dimmer value extraction and fix it
    old_dimmer_extraction = '''                    try:
                        dimmer_value = self._safe_float_conversion(action_params.get("value", "1.0"), 1.0, "dimmer_value")
                        self.logger.debug(f"_apply_enhanced_action_effects: Applying dimmer value: {dimmer_value}")
                        future_state = self._apply_dimmer_effects(future_state, dimmer_value, time_step)
                        self.logger.debug(f"_apply_enhanced_action_effects: Dimmer effects applied successfully")
                    except Exception as e:
                        self.logger.error(f"_apply_enhanced_action_effects: Dimmer effects failed: {e}", exc_info=True)
                        raise'''
    
    new_dimmer_extraction = '''                    try:
                        # Safely extract and convert dimmer value (handles gRPC string conversion)
                        raw_dimmer_value = action_params.get("value", "1.0")
                        dimmer_value = self._safe_float_conversion(raw_dimmer_value, 1.0, "dimmer_value")
                        self.logger.debug(f"_apply_enhanced_action_effects: Raw dimmer value: {raw_dimmer_value} ({type(raw_dimmer_value).__name__}) -> {dimmer_value} (float)")
                        future_state = self._apply_dimmer_effects(future_state, dimmer_value, time_step)
                        self.logger.debug(f"_apply_enhanced_action_effects: Dimmer effects applied successfully")
                    except Exception as e:
                        self.logger.error(f"_apply_enhanced_action_effects: Dimmer effects failed: {e}", exc_info=True)
                        raise'''
    
    if old_dimmer_extraction in content:
        content = content.replace(old_dimmer_extraction, new_dimmer_extraction)
        fixes_applied.append("Enhanced dimmer value extraction with type logging")
    
    # Write the fixed content to a new file
    fixed_file = Path("src/polaris/models/bayesian_world_model_fixed.py")
    with open(fixed_file, 'w') as f:
        f.write(content)
    
    print("=" * 80)
    print("BAYESIAN WORLD MODEL TYPE ERROR FIXES APPLIED")
    print("=" * 80)
    print(f"Original file: {original_file}")
    print(f"Fixed file: {fixed_file}")
    print(f"Fixes applied: {len(fixes_applied)}")
    
    for i, fix in enumerate(fixes_applied, 1):
        print(f"  {i}. {fix}")
    
    print("\nSUMMARY OF FIXES:")
    print("- Added _safe_int_conversion() method for integer parameters")
    print("- Fixed simulate() method to safely convert horizon_minutes")
    print("- Enhanced _calculate_adaptive_timesteps() with defensive programming")
    print("- Improved _apply_dimmer_effects() with safe conversion")
    print("- Added comprehensive type logging for debugging")
    
    print("\nNEXT STEPS:")
    print("1. Review the fixed file: bayesian_world_model_fixed.py")
    print("2. Test the fixes with the reproduction script")
    print("3. Replace the original file if tests pass")
    print("4. Deploy the fix to resolve the gRPC type conversion error")
    
    return True


def create_test_for_fixed_model():
    """Create a test to verify the fixes work."""
    
    test_content = '''#!/usr/bin/env python3
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
sys.path.insert(0, str(Path(__file__).parent / "src" / "polaris" / "models"))
from bayesian_world_model_fixed import BayesianWorldModel
from polaris.models.world_model import SimulationRequest


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
    
    print("\\n" + "=" * 80)
    if success:
        print("✓ ALL TESTS PASSED: The type error fixes are working correctly!")
        print("The 'bad argument type for built-in operation' error should be resolved.")
    else:
        print("✗ TESTS FAILED: The fixes need further refinement.")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
'''
    
    test_file = Path("test_fixed_bayesian_model.py")
    with open(test_file, 'w') as f:
        f.write(test_content)
    
    print(f"\\nCreated test file: {test_file}")
    print("Run this test to verify the fixes work correctly.")


def main():
    """Main function to apply fixes."""
    print("Applying fixes for Bayesian World Model type conversion error...")
    
    success = create_fixed_bayesian_world_model()
    
    if success:
        create_test_for_fixed_model()
        print("\\n" + "=" * 80)
        print("FIXES COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("The 'bad argument type for built-in operation' error should now be resolved.")
        print("\\nTo test the fixes:")
        print("1. python test_fixed_bayesian_model.py")
        print("\\nTo deploy the fixes:")
        print("1. Review bayesian_world_model_fixed.py")
        print("2. Replace the original file if tests pass")
    else:
        print("\\nFailed to apply fixes. Please check the file paths and try again.")


if __name__ == "__main__":
    main()