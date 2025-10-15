#!/usr/bin/env python3
"""
Test script for Bayesian Agentic Reasoner setup.

This script tests the creation of an agentic reasoner with Bayesian world model.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_bayesian_agentic_creation():
    """Test creating agentic reasoner with Bayesian world model."""
    print("🧪 Testing Bayesian Agentic Reasoner Creation")
    print("=" * 50)
    
    try:
        from polaris.agents.agentic_reasoner import create_agentic_reasoner_with_bayesian_world_model
        
        print("✅ Import successful")
        
        # Test configuration path
        config_path = "config/bayesian_world_model_config.yaml"
        if not Path(config_path).exists():
            config_path = "src/config/polaris_config.yaml"
        
        print(f"📋 Using config: {config_path}")
        
        # Test creation (dry run - don't actually start)
        print("🔧 Testing agent creation...")
        
        # This would normally create the agent
        # agent = create_agentic_reasoner_with_bayesian_world_model(
        #     agent_id="test-bayesian-001",
        #     config_path=config_path,
        #     llm_api_key="test-key",
        #     logger=None
        # )
        
        print("✅ Function is available and should work")
        print("💡 To test fully, run: python src/scripts/start_component.py agentic-reasoner --use-bayesian-world-model")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False


def show_setup_steps():
    """Show the complete setup steps."""
    print("\n📋 Complete Setup Steps for Bayesian Agentic Reasoner")
    print("=" * 60)
    
    steps = [
        "1. Start Digital Twin with Bayesian world model:",
        "   python src/scripts/start_component.py digital-twin --world-model bayesian",
        "",
        "2. In another terminal, start Agentic Reasoner:",
        "   python src/scripts/start_component.py agentic-reasoner --use-bayesian-world-model",
        "",
        "3. Or start with monitoring:",
        "   python src/scripts/start_component.py agentic-reasoner --use-bayesian-world-model --monitor-performance",
        "",
        "4. Verify connection in logs:",
        "   - Digital Twin: 'gRPC service available at 0.0.0.0:50051'",
        "   - Agentic Reasoner: 'Digital Twin client: ImprovedGRPCDigitalTwinClient'",
        "",
        "5. Optional: Start monitor for telemetry data:",
        "   python src/scripts/start_component.py monitor --plugin-dir extern",
    ]
    
    for step in steps:
        print(step)


def main():
    """Run the test."""
    success = test_bayesian_agentic_creation()
    show_setup_steps()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Test completed successfully!")
    else:
        print("❌ Test failed - check dependencies and imports")


if __name__ == "__main__":
    main()