#!/usr/bin/env python3
"""
Test script to verify the simulation KeyError fix.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timezone

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from polaris.models.world_model import SimulationRequest
from polaris.models.bayesian_world_model import BayesianWorldModel

async def test_simulation_fix():
    """Test the simulation fix for the 'state' KeyError."""
    
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    
    print("Testing simulation KeyError fix...")
    
    # Create a minimal config for the Bayesian world model
    config = {
        "prediction_horizon_minutes": 20,
        "max_history_points": 300,
        "update_interval_seconds": 30,
        "correlation_threshold": 0.65,
        "anomaly_threshold": 4.0,
        "prior_confidence": 0.4,
        "learning_rate": 0.15,
        "process_noise": 0.08,
        "measurement_noise": 0.06,
        "initial_uncertainty": 0.6,
        "max_concurrent_operations": 6,
        "computation_timeout_sec": 20
    }
    
    try:
        # Initialize the world model
        world_model = BayesianWorldModel(config, logger)
        await world_model.initialize()
        
        print("✓ World model initialized successfully")
        
        # Create a test simulation request with the problematic actions
        simulation_request = SimulationRequest(
            simulation_id="test-simulation",
            simulation_type="model_comparison",
            actions=[
                {
                    'action_type': 'SWITCH_MODEL_YOLOV5N',
                    'params': {'model': 'yolov5n'}
                },
                {
                    'action_type': 'SWITCH_MODEL_YOLOV5S', 
                    'params': {'model': 'yolov5s'}
                }
            ],
            horizon_minutes=5
        )
        
        print("✓ Simulation request created")
        
        # Run the simulation
        print("Running simulation...")
        result = await world_model.simulate(simulation_request)
        
        if result.success:
            print("✓ Simulation completed successfully!")
            print(f"  - Simulation ID: {result.simulation_id}")
            print(f"  - Confidence: {result.confidence:.3f}")
            print(f"  - Future states: {len(result.future_states)}")
            print(f"  - Explanation: {result.explanation}")
            
            # Check if future states have the correct structure
            if result.future_states:
                first_state = result.future_states[0]
                print(f"  - First state keys: {list(first_state.keys())}")
                if "metrics" in first_state:
                    print(f"  - Metrics in first state: {list(first_state['metrics'].keys())}")
                else:
                    print("  - Warning: No metrics in first state")
            
            return True
        else:
            print(f"✗ Simulation failed: {result.explanation}")
            if result.metadata:
                print(f"  - Error details: {result.metadata}")
            return False
            
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        try:
            await world_model.shutdown()
            print("✓ World model shutdown completed")
        except:
            pass

if __name__ == "__main__":
    success = asyncio.run(test_simulation_fix())
    if success:
        print("\n🎉 All tests passed! The KeyError fix is working.")
        sys.exit(0)
    else:
        print("\n❌ Tests failed. Please check the error messages above.")
        sys.exit(1)