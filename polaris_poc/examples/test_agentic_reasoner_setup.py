#!/usr/bin/env python3
"""
Test script to verify agentic reasoner setup and reasoning implementations.

This script tests that the agentic reasoner has all required reasoning implementations.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_agentic_reasoner_implementations():
    """Test that agentic reasoner has all reasoning implementations."""
    print("🧪 Testing Agentic Reasoner Implementation Setup")
    print("=" * 60)
    
    try:
        from polaris.agents.agentic_reasoner import create_agentic_reasoner_with_bayesian_world_model
        from polaris.agents.reasoner_core import ReasoningType
        import logging
        
        print("✅ Imports successful")
        
        # Create a test logger
        logger = logging.getLogger("test")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        
        # Test configuration path
        config_path = "config/bayesian_world_model_config.yaml"
        if not Path(config_path).exists():
            config_path = "src/config/polaris_config.yaml"
        
        print(f"📋 Using config: {config_path}")
        
        # Create the agentic reasoner
        print("🔧 Creating agentic reasoner with Bayesian world model...")
        
        agent = create_agentic_reasoner_with_bayesian_world_model(
            agent_id="test-bayesian-reasoner",
            config_path=config_path,
            llm_api_key="test-key-for-validation",
            logger=logger
        )
        
        print("✅ Agentic reasoner created successfully")
        
        # Check reasoning implementations
        supported_types = agent.get_supported_reasoning_types()
        print(f"📊 Supported reasoning types: {len(supported_types)}")
        
        all_types = list(ReasoningType)
        print(f"📋 Expected reasoning types: {len(all_types)}")
        
        for reasoning_type in all_types:
            if reasoning_type in supported_types:
                print(f"   ✅ {reasoning_type.value}")
            else:
                print(f"   ❌ {reasoning_type.value} - MISSING!")
        
        # Check if all types are supported
        missing_types = [rt for rt in all_types if rt not in supported_types]
        
        if not missing_types:
            print("\n✅ All reasoning implementations are present!")
            return True
        else:
            print(f"\n❌ Missing reasoning implementations: {[rt.value for rt in missing_types]}")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_reasoning_types():
    """Show all available reasoning types."""
    print("\n📋 Available Reasoning Types")
    print("=" * 40)
    
    try:
        from polaris.agents.reasoner_core import ReasoningType
        
        for reasoning_type in ReasoningType:
            print(f"  - {reasoning_type.value}: {reasoning_type.name}")
        
        print(f"\nTotal: {len(list(ReasoningType))} reasoning types")
        
    except Exception as e:
        print(f"❌ Could not load reasoning types: {e}")


def main():
    """Run the test."""
    show_reasoning_types()
    success = test_agentic_reasoner_implementations()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Agentic reasoner setup test passed!")
        print("\n💡 The agentic reasoner should now handle all reasoning types")
        print("   including INFERENCE requests from the slow controller")
    else:
        print("❌ Agentic reasoner setup test failed!")
        print("\n🔧 This explains why you're getting the 'No implementation' error")
        print("   The reasoning implementations aren't being added properly")
    
    print("\n📚 Next steps:")
    print("1. Restart your agentic reasoner component")
    print("2. Check the logs for 'Added reasoning implementation for...' messages")
    print("3. Try the slow controller delegation again")


if __name__ == "__main__":
    main()