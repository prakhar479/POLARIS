#!/usr/bin/env python3
"""
Test script for regular agentic reasoner (without Bayesian world model).

This script tests the creation of a regular agentic reasoner to ensure
it works without the --use-bayesian-world-model flag.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_regular_agentic_reasoner():
    """Test creating regular agentic reasoner."""
    print("🧪 Testing Regular Agentic Reasoner Creation")
    print("=" * 50)
    
    try:
        from polaris.agents.agentic_reasoner import create_agentic_reasoner_agent
        from polaris.agents.reasoner_core import ReasoningType
        import logging
        
        print("✅ Imports successful")
        
        # Create a test logger
        logger = logging.getLogger("test")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        
        # Test configuration path
        config_path = "src/config/agentic_reasoner_config.yaml"
        if not Path(config_path).exists():
            config_path = "src/config/polaris_config.yaml"
        
        print(f"📋 Using config: {config_path}")
        
        # Create the regular agentic reasoner
        print("🔧 Creating regular agentic reasoner...")
        
        agent = create_agentic_reasoner_agent(
            agent_id="test-regular-reasoner",
            config_path=config_path,
            llm_api_key="test-key-for-validation",
            logger=logger,
            use_improved_grpc=True
        )
        
        print("✅ Regular agentic reasoner created successfully")
        
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


def show_startup_commands():
    """Show the correct startup commands."""
    print("\n📋 Correct Startup Commands")
    print("=" * 50)
    
    commands = [
        "# Regular agentic reasoner (works with any world model)",
        "python src/scripts/start_component.py agentic-reasoner",
        "",
        "# With performance monitoring",
        "python src/scripts/start_component.py agentic-reasoner --monitor-performance",
        "",
        "# With robust timeout configuration",
        "python src/scripts/start_component.py agentic-reasoner --timeout-config robust",
        "",
        "# With Bayesian world model (special setup)",
        "python src/scripts/start_component.py agentic-reasoner --use-bayesian-world-model",
        "",
        "# Complete hybrid system setup:",
        "# 1. Start Digital Twin",
        "python src/scripts/start_component.py digital-twin --world-model bayesian",
        "",
        "# 2. Start Agentic Reasoner",
        "python src/scripts/start_component.py agentic-reasoner --monitor-performance",
        "",
        "# 3. Start other components...",
        "python src/scripts/start_component.py kernel",
    ]
    
    for cmd in commands:
        print(cmd)


def main():
    """Run the test."""
    success = test_regular_agentic_reasoner()
    show_startup_commands()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Regular agentic reasoner test passed!")
        print("\n💡 You can now run:")
        print("   python src/scripts/start_component.py agentic-reasoner --monitor-performance")
    else:
        print("❌ Regular agentic reasoner test failed!")
        print("\n🔧 Check the error messages above for details")
    
    print("\n📚 For hybrid fast+slow controller:")
    print("1. Start digital-twin --world-model bayesian")
    print("2. Start agentic-reasoner --monitor-performance")
    print("3. Start kernel (for hybrid controller)")


if __name__ == "__main__":
    main()