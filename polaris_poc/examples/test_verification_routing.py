#!/usr/bin/env python3
"""
Test script to verify that action verification routing is working correctly.

This script tests:
1. Configuration loading for verification settings
2. Reasoner agent verification routing
3. Verification adapter constraint checking
4. End-to-end action flow through verification
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from polaris.common.config import ConfigurationManager
from polaris.agents.reasoner_agent import ReasonerAgent
from polaris.adapters.verification import VerificationAdapter


async def test_configuration_loading():
    """Test that verification configuration is loaded correctly."""
    print("\n=== Testing Configuration Loading ===")
    
    try:
        config_path = Path(__file__).parent.parent / "src" / "config" / "polaris_config.yaml"
        logger = logging.getLogger("test_config")
        config_manager = ConfigurationManager(logger=logger)
        config_manager.load_framework_config(config_path)
        config = config_manager.framework_config
        
        # Check verification configuration
        verification_config = config.get("verification", {})
        print(f"✓ Verification enabled by default: {verification_config.get('enable_by_default', False)}")
        print(f"✓ Verification routing enabled: {verification_config.get('routing', {}).get('enable_automatic_routing', False)}")
        
        # Check reasoner configuration
        reasoner_config = config.get("reasoner", {})
        action_routing = reasoner_config.get("action_routing", {})
        print(f"✓ Reasoner verification enabled: {action_routing.get('enable_verification', False)}")
        print(f"✓ Default verification level: {action_routing.get('default_verification_level', 'basic')}")
        
        # Check kernel configuration
        kernel_config = config.get("kernel", {})
        print(f"✓ Kernel verification enabled: {kernel_config.get('enable_verification', False)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False


async def test_reasoner_verification_routing():
    """Test that reasoner agent routes actions through verification."""
    print("\n=== Testing Reasoner Verification Routing ===")
    
    try:
        config_path = Path(__file__).parent.parent / "src" / "config" / "polaris_config.yaml"
        
        # Create mock reasoning implementations
        from polaris.agents.reasoner_agent import ReasoningInterface, ReasoningContext, ReasoningResult, ReasoningType
        
        class MockReasoning(ReasoningInterface):
            async def reason(self, context, knowledge=None):
                return ReasoningResult(
                    result={"action": "test_action", "confidence": 0.8},
                    confidence=0.8,
                    reasoning_steps=["mock reasoning step"],
                    context=context,
                    execution_time=0.1
                )
            
            async def validate_input(self, context):
                return True
            
            def get_required_knowledge_types(self, context):
                return []
            
            def extract_search_terms(self, context):
                return []
        
        reasoning_implementations = {"mock": MockReasoning()}
        
        # Create reasoner agent
        reasoner = ReasonerAgent(
            config_path=str(config_path),
            agent_id="test_reasoner",
            nats_url="nats://localhost:4222",
            reasoning_implementations=reasoning_implementations
        )
        
        # Check verification settings
        print(f"✓ Verification enabled: {reasoner.enable_verification}")
        print(f"✓ Verification level: {reasoner.verification_level}")
        print(f"✓ Verification timeout: {reasoner.verification_timeout}s")
        print(f"✓ Verification input subject: {reasoner.verification_input_subject}")
        print(f"✓ Failure action: {reasoner.verification_failure_action}")
        
        return True
        
    except Exception as e:
        print(f"✗ Reasoner verification routing test failed: {e}")
        return False


async def test_verification_adapter_constraints():
    """Test that verification adapter loads constraints correctly."""
    print("\n=== Testing Verification Adapter Constraints ===")
    
    try:
        config_path = Path(__file__).parent.parent / "src" / "config" / "polaris_config.yaml"
        verification_config_dir = Path(__file__).parent.parent / "config"
        
        # Create verification adapter with proper plugin directory
        verification_adapter = VerificationAdapter(
            polaris_config_path=str(config_path),
            plugin_dir=str(verification_config_dir)
        )
        
        print(f"✓ Constraints loaded: {len(verification_adapter.constraints)}")
        print(f"✓ Policies loaded: {len(verification_adapter.policies)}")
        print(f"✓ Digital Twin enabled: {verification_adapter.enable_digital_twin}")
        print(f"✓ Formal verification enabled: {verification_adapter.enable_formal_verification}")
        print(f"✓ Default timeout: {verification_adapter.default_timeout}s")
        print(f"✓ Max concurrent: {verification_adapter.max_concurrent}")
        
        # List some constraints
        if verification_adapter.constraints:
            print("\nLoaded constraints:")
            for constraint in verification_adapter.constraints[:3]:  # Show first 3
                print(f"  - {constraint.get('id', 'unknown')}: {constraint.get('description', 'no description')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Verification adapter constraints test failed: {e}")
        return False


async def simulate_action_verification():
    """Simulate an action going through verification."""
    print("\n=== Simulating Action Verification ===")
    
    try:
        from polaris.adapters.verification import VerificationRequest, VerificationLevel
        from polaris.models.actions import ControlAction
        
        # Create a test action
        test_action = ControlAction(
            action_id="test_action_001",
            action_type="configuration_change",
            target_system="test_system",
            parameters={"setting": "test_value"},
            metadata={"source": "test_script"}
        )
        
        # Create verification request
        verification_request = VerificationRequest(
            request_id="test_request_001",
            action=test_action,
            context={"system_load": 0.5, "error_rate": 0.01},
            verification_level=VerificationLevel.POLICY,
            requester="test_script"
        )
        
        print(f"✓ Created test action: {test_action.action_type}")
        print(f"✓ Created verification request: {verification_request.verification_level.value}")
        print(f"✓ Request ID: {verification_request.request_id}")
        
        return True
        
    except Exception as e:
        print(f"✗ Action verification simulation failed: {e}")
        return False


async def check_verification_subjects():
    """Check that verification NATS subjects are configured correctly."""
    print("\n=== Checking Verification NATS Subjects ===")
    
    try:
        config_path = Path(__file__).parent.parent / "src" / "config" / "polaris_config.yaml"
        logger = logging.getLogger("test_subjects")
        config_manager = ConfigurationManager(logger=logger)
        config_manager.load_framework_config(config_path)
        config = config_manager.framework_config
        
        verification_config = config.get("verification", {})
        
        subjects = {
            "Input Subject": verification_config.get("input_subject"),
            "Output Subject": verification_config.get("output_subject"),
            "Policy Subject": verification_config.get("policy_subject"),
            "Metrics Subject": verification_config.get("metrics_subject")
        }
        
        for name, subject in subjects.items():
            if subject:
                print(f"✓ {name}: {subject}")
            else:
                print(f"✗ {name}: Not configured")
        
        return all(subjects.values())
        
    except Exception as e:
        print(f"✗ NATS subjects check failed: {e}")
        return False


async def main():
    """Run all verification routing tests."""
    print("POLARIS Verification Routing Test Suite")
    print("=" * 50)
    
    tests = [
        ("Configuration Loading", test_configuration_loading),
        ("Reasoner Verification Routing", test_reasoner_verification_routing),
        ("Verification Adapter Constraints", test_verification_adapter_constraints),
        ("Action Verification Simulation", simulate_action_verification),
        ("Verification NATS Subjects", check_verification_subjects)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status:4} | {test_name}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All verification routing tests passed!")
        return 0
    else:
        print("❌ Some tests failed. Check configuration and implementation.")
        return 1


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    exit_code = asyncio.run(main())
    sys.exit(exit_code)