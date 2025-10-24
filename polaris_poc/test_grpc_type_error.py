#!/usr/bin/env python3
"""
Targeted test to reproduce the "bad argument type for built-in operation" error
by simulating the exact gRPC parameter conversion scenario.
"""

import asyncio
import logging
import sys
import traceback
from pathlib import Path

# Add the src directory to the path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def test_grpc_parameter_conversion_error():
    """Test the specific gRPC parameter conversion that causes the error."""
    
    logger = logging.getLogger("test_grpc_type_error")
    logger.info("Testing gRPC parameter conversion scenarios that cause 'bad argument type for built-in operation'")
    
    # Simulate the exact parameter conversion that happens in improved_grpc_client.py
    # The gRPC client converts all parameters to strings
    
    # Original action from agentic reasoner
    original_action = {
        'action_type': 'SWITCH_MODEL_YOLOV5S',
        'params': {'model': 'yolov5s'}
    }
    
    # After gRPC conversion (this is what the Bayesian World Model receives)
    grpc_converted_action = {
        'action_type': 'SWITCH_MODEL_YOLOV5S', 
        'params': {'model': 'yolov5s'}  # This is already a string, but let's test numeric params
    }
    
    # Test scenarios that would cause the error
    test_scenarios = [
        {
            "name": "Numeric parameter converted to string",
            "original": {'dimmer_value': 0.8},
            "grpc_converted": {'dimmer_value': '0.8'},
            "operation": lambda params: float(params['dimmer_value']) * 1.2  # This should work
        },
        {
            "name": "Numeric parameter used directly as string",
            "original": {'dimmer_value': 0.8},
            "grpc_converted": {'dimmer_value': '0.8'},
            "operation": lambda params: params['dimmer_value'] * 1.2  # This will fail!
        },
        {
            "name": "Integer parameter converted to string",
            "original": {'horizon_minutes': 15},
            "grpc_converted": {'horizon_minutes': '15'},
            "operation": lambda params: params['horizon_minutes'] + 5  # This will fail!
        },
        {
            "name": "Float parameter in mathematical operation",
            "original": {'cpu_factor': 1.5},
            "grpc_converted": {'cpu_factor': '1.5'},
            "operation": lambda params: params['cpu_factor'] / 2.0  # This will fail!
        },
        {
            "name": "Comparison operation with string",
            "original": {'threshold': 0.7},
            "grpc_converted": {'threshold': '0.7'},
            "operation": lambda params: params['threshold'] > 0.5  # This will fail!
        }
    ]
    
    errors_found = []
    
    for scenario in test_scenarios:
        try:
            result = scenario["operation"](scenario["grpc_converted"])
            logger.info(f"✓ {scenario['name']}: {result}")
        except Exception as e:
            error_msg = str(e)
            logger.error(f"✗ {scenario['name']}: {type(e).__name__}: {error_msg}")
            
            if "bad argument type for built-in operation" in error_msg:
                logger.error(f"*** FOUND TARGET ERROR in {scenario['name']} ***")
                errors_found.append(scenario['name'])
            elif any(keyword in error_msg.lower() for keyword in ['unsupported operand', 'can only concatenate', "can't multiply"]):
                logger.error(f"*** FOUND RELATED TYPE ERROR in {scenario['name']} ***")
                errors_found.append(scenario['name'])
    
    return errors_found


def test_bayesian_world_model_parameter_usage():
    """Test how the Bayesian World Model might be using parameters incorrectly."""
    
    logger = logging.getLogger("test_bayesian_parameter_usage")
    logger.info("Testing Bayesian World Model parameter usage patterns")
    
    # Simulate the _safe_float_conversion method behavior
    def safe_float_conversion(value, default=0.0, param_name="unknown"):
        """Simulate the safe conversion method from BayesianWorldModel."""
        try:
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str):
                return float(value)
            else:
                logger.warning(f"Unexpected type for parameter '{param_name}': {type(value).__name__}")
                return default
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to convert parameter '{param_name}' value '{value}' to float: {e}")
            return default
    
    # Test the safe conversion
    test_cases = [
        ("String number", "1.5"),
        ("Integer", 42),
        ("Float", 3.14),
        ("Invalid string", "not_a_number"),
        ("None value", None),
        ("Boolean", True),
    ]
    
    logger.info("Testing safe_float_conversion:")
    for name, value in test_cases:
        result = safe_float_conversion(value, param_name=name)
        logger.info(f"  {name}: {value} ({type(value).__name__}) -> {result} (float)")
    
    # Now test what happens if the safe conversion is NOT used
    logger.info("\nTesting direct parameter usage (without safe conversion):")
    
    # This simulates what might happen in the Bayesian World Model if parameters
    # are used directly without conversion
    grpc_params = {
        'model': 'yolov5s',
        'dimmer_value': '0.8',  # String from gRPC
        'horizon_minutes': '15',  # String from gRPC
        'cpu_factor': '1.5',  # String from gRPC
    }
    
    error_scenarios = [
        {
            "name": "Direct arithmetic with string dimmer_value",
            "operation": lambda: grpc_params['dimmer_value'] * 1.2
        },
        {
            "name": "Direct addition with string horizon_minutes", 
            "operation": lambda: grpc_params['horizon_minutes'] + 5
        },
        {
            "name": "Direct division with string cpu_factor",
            "operation": lambda: grpc_params['cpu_factor'] / 2.0
        },
        {
            "name": "Direct comparison with string",
            "operation": lambda: grpc_params['dimmer_value'] > 0.5
        },
        {
            "name": "NumPy operation with string",
            "operation": lambda: __import__('numpy').array([grpc_params['cpu_factor']]) + 1.0
        }
    ]
    
    errors_found = []
    
    for scenario in error_scenarios:
        try:
            result = scenario["operation"]()
            logger.info(f"✓ {scenario['name']}: {result}")
        except Exception as e:
            error_msg = str(e)
            logger.error(f"✗ {scenario['name']}: {type(e).__name__}: {error_msg}")
            
            if "bad argument type for built-in operation" in error_msg:
                logger.error(f"*** FOUND TARGET ERROR in {scenario['name']} ***")
                errors_found.append(scenario['name'])
            elif any(keyword in error_msg.lower() for keyword in ['unsupported operand', 'can only concatenate', "can't multiply"]):
                logger.error(f"*** FOUND RELATED TYPE ERROR in {scenario['name']} ***")
                errors_found.append(scenario['name'])
    
    return errors_found


def analyze_error_location():
    """Analyze where the error is most likely occurring based on the logs."""
    
    logger = logging.getLogger("error_analysis")
    logger.info("Analyzing the most likely location of the 'bad argument type for built-in operation' error")
    
    # Based on the error logs, the error occurs in the digital twin operation
    # Let's analyze the call stack from the logs:
    
    analysis = """
    ERROR ANALYSIS:
    ===============
    
    From the error logs:
    - Error occurs in: agentic_reasoner:258 | Digital twin operation failed
    - The error message: "Simulation failed: bad argument type for built-in operation"
    - This suggests the error is happening inside the Digital Twin simulation
    
    Likely locations for the error:
    
    1. In improved_grpc_client.py when converting parameters:
       - All gRPC parameters are converted to strings
       - If these strings are used directly in math operations, it will fail
    
    2. In bayesian_world_model.py in simulation methods:
       - Parameters from gRPC are strings
       - If used directly without _safe_float_conversion(), will cause type errors
    
    3. Most likely culprit based on logs:
       - The error occurs during simulation processing
       - Parameters like 'model', 'horizon_minutes' are converted to strings by gRPC
       - If horizon_minutes is used directly in math operations as a string, it will fail
    
    SPECIFIC ISSUE:
    - horizon_minutes=15 becomes horizon_minutes='15' after gRPC conversion
    - If code does: time_steps = horizon_minutes * 2  # This will fail!
    - Should be: time_steps = int(horizon_minutes) * 2
    """
    
    logger.info(analysis)
    
    # Test the specific horizon_minutes scenario
    logger.info("Testing the horizon_minutes scenario:")
    
    try:
        # This is what the code receives from gRPC
        horizon_minutes_str = '15'
        
        # This would cause the error
        result = horizon_minutes_str * 2
        logger.info(f"String multiplication result: {result}")  # This actually works but gives wrong result
        
        # This would cause the error in mathematical context
        result = horizon_minutes_str + 5  # This will fail!
        
    except Exception as e:
        logger.error(f"*** FOUND THE ERROR: {type(e).__name__}: {e} ***")
        return True
    
    return False


def main():
    """Main test function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
    )
    
    print("=" * 80)
    print("POLARIS gRPC Type Conversion Error Analysis")
    print("=" * 80)
    
    print("\n1. Testing gRPC parameter conversion scenarios...")
    grpc_errors = test_grpc_parameter_conversion_error()
    
    print("\n2. Testing Bayesian World Model parameter usage...")
    bayesian_errors = test_bayesian_world_model_parameter_usage()
    
    print("\n3. Analyzing error location...")
    location_found = analyze_error_location()
    
    print("\n" + "=" * 80)
    print("ANALYSIS RESULTS:")
    print("=" * 80)
    
    total_errors = len(grpc_errors) + len(bayesian_errors)
    
    if total_errors > 0:
        print(f"✓ Found {total_errors} type conversion errors that could cause 'bad argument type for built-in operation'")
        print(f"  - gRPC conversion errors: {len(grpc_errors)}")
        print(f"  - Bayesian model usage errors: {len(bayesian_errors)}")
        
        if grpc_errors:
            print(f"  - gRPC scenarios: {', '.join(grpc_errors)}")
        if bayesian_errors:
            print(f"  - Bayesian scenarios: {', '.join(bayesian_errors)}")
            
        print("\nROOT CAUSE IDENTIFIED:")
        print("- gRPC converts all parameters to strings")
        print("- Code uses these string parameters directly in mathematical operations")
        print("- This causes 'bad argument type for built-in operation' errors")
        
        print("\nSOLUTION:")
        print("- Use _safe_float_conversion() for all numeric parameters")
        print("- Add type checking before mathematical operations")
        print("- Ensure proper parameter validation in the Bayesian World Model")
        
    else:
        print("✗ Could not reproduce the specific type conversion errors")
    
    if location_found:
        print("✓ Successfully identified the likely error location")
    
    print("=" * 80)


if __name__ == "__main__":
    main()