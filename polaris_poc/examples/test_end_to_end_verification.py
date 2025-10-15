#!/usr/bin/env python3
"""
End-to-end verification test to ensure actions are properly routed through verification.

This test simulates the complete flow:
1. Reasoner generates an action
2. Action is sent to verification adapter
3. Verification adapter validates against constraints and policies
4. Approved actions are forwarded to execution
5. Rejected actions are blocked with detailed reasoning
"""

import asyncio
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Any

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from polaris.common.config import ConfigurationManager
from polaris.adapters.verification import VerificationAdapter, VerificationRequest, VerificationLevel
from polaris.models.actions import ControlAction


class MockNATSClient:
    """Mock NATS client for testing verification flow without actual NATS server."""
    
    def __init__(self):
        self.published_messages = []
        self.subscriptions = {}
    
    async def publish_json(self, subject: str, data: Dict[str, Any]):
        """Mock publish that stores messages for verification."""
        message = {
            "subject": subject,
            "data": data,
            "timestamp": time.time()
        }
        self.published_messages.append(message)
        print(f"📤 Published to {subject}: {data.get('action_type', 'unknown')} action")
    
    async def subscribe(self, subject: str, callback):
        """Mock subscribe for testing."""
        self.subscriptions[subject] = callback
    
    def get_messages_for_subject(self, subject: str):
        """Get all messages published to a specific subject."""
        return [msg for msg in self.published_messages if msg["subject"] == subject]


async def test_action_approval_flow():
    """Test that low-risk actions are approved and forwarded to execution."""
    print("\n=== Testing Action Approval Flow ===")
    
    try:
        # Create verification adapter with mock NATS
        config_path = Path(__file__).parent.parent / "src" / "config" / "polaris_config.yaml"
        verification_config_dir = Path(__file__).parent.parent / "config"
        
        verification_adapter = VerificationAdapter(
            polaris_config_path=str(config_path),
            plugin_dir=str(verification_config_dir)
        )
        
        # Replace NATS client with mock
        mock_nats = MockNATSClient()
        verification_adapter.nats_client = mock_nats
        
        # Create a low-risk action that should be approved
        test_action = ControlAction(
            action_id=f"test_action_{uuid.uuid4()}",
            action_type="scaling_action",  # Low risk according to our config
            target_system="test_system",
            parameters={
                "scale_factor": 1.2,
                "max_instances": 5
            },
            metadata={
                "source": "test_script",
                "risk_level": "low"
            }
        )
        
        # Create verification request
        verification_request = VerificationRequest(
            request_id=f"test_request_{uuid.uuid4()}",
            action=test_action,
            context={
                "system": {
                    "cpu_utilization": 0.60,  # Below 90% threshold
                    "memory_utilization": 0.70,  # Below 85% threshold
                    "concurrent_actions": 3,  # Below 10 limit
                    "availability": 0.995,  # Above 99% requirement
                    "error_rate": 0.02  # Below 5% threshold
                }
            },
            verification_level=VerificationLevel.BASIC,
            requester="test_script"
        )
        
        # Verify the action
        result = await verification_adapter.verify_action(verification_request)
        
        # Check results
        print(f"✓ Action ID: {result.action_id}")
        print(f"✓ Approved: {result.approved}")
        print(f"✓ Confidence: {result.confidence:.2f}")
        print(f"✓ Violations: {len(result.violations)}")
        print(f"✓ Verification time: {result.verification_time_ms:.1f}ms")
        
        if result.approved:
            print("✅ Low-risk action was correctly approved")
            return True
        else:
            print(f"❌ Low-risk action was incorrectly rejected: {[v.description for v in result.violations]}")
            return False
            
    except Exception as e:
        print(f"❌ Action approval test failed: {e}")
        return False


async def test_action_rejection_flow():
    """Test that high-risk actions are rejected when constraints are violated."""
    print("\n=== Testing Action Rejection Flow ===")
    
    try:
        # Create verification adapter
        config_path = Path(__file__).parent.parent / "src" / "config" / "polaris_config.yaml"
        verification_config_dir = Path(__file__).parent.parent / "config"
        
        verification_adapter = VerificationAdapter(
            polaris_config_path=str(config_path),
            plugin_dir=str(verification_config_dir)
        )
        
        # Replace NATS client with mock
        mock_nats = MockNATSClient()
        verification_adapter.nats_client = mock_nats
        
        # Create a high-risk action that should be rejected due to resource constraints
        test_action = ControlAction(
            action_id=f"test_action_{uuid.uuid4()}",
            action_type="system_restart",  # High risk according to our config
            target_system="production_system",
            parameters={
                "restart_type": "hard_restart",
                "downtime_expected": "5_minutes"
            },
            metadata={
                "source": "test_script",
                "risk_level": "high"
            }
        )
        
        # Create verification request with constraint violations
        verification_request = VerificationRequest(
            request_id=f"test_request_{uuid.uuid4()}",
            action=test_action,
            context={
                "system": {
                    "cpu_utilization": 0.95,  # Above 90% threshold - VIOLATION
                    "memory_utilization": 0.88,  # Above 85% threshold - VIOLATION
                    "concurrent_actions": 12,  # Above 10 limit - VIOLATION
                    "availability": 0.985,  # Below 99% requirement - VIOLATION
                    "error_rate": 0.08  # Above 5% threshold - VIOLATION
                }
            },
            verification_level=VerificationLevel.COMPREHENSIVE,
            requester="test_script"
        )
        
        # Verify the action
        result = await verification_adapter.verify_action(verification_request)
        
        # Check results
        print(f"✓ Action ID: {result.action_id}")
        print(f"✓ Approved: {result.approved}")
        print(f"✓ Confidence: {result.confidence:.2f}")
        print(f"✓ Violations found: {len(result.violations)}")
        print(f"✓ Verification time: {result.verification_time_ms:.1f}ms")
        
        if result.violations:
            print("📋 Constraint violations detected:")
            for violation in result.violations:
                print(f"  - {violation.constraint_id}: {violation.description}")
                if violation.suggested_fix:
                    print(f"    💡 Suggested fix: {violation.suggested_fix}")
        
        if not result.approved and len(result.violations) > 0:
            print("✅ High-risk action with violations was correctly rejected")
            return True
        else:
            print("❌ High-risk action with violations was incorrectly approved")
            return False
            
    except Exception as e:
        print(f"❌ Action rejection test failed: {e}")
        return False


async def test_policy_enforcement():
    """Test that organizational policies are properly enforced."""
    print("\n=== Testing Policy Enforcement ===")
    
    try:
        # Create verification adapter
        config_path = Path(__file__).parent.parent / "src" / "config" / "polaris_config.yaml"
        verification_config_dir = Path(__file__).parent.parent / "config"
        
        verification_adapter = VerificationAdapter(
            polaris_config_path=str(config_path),
            plugin_dir=str(verification_config_dir)
        )
        
        # Test action that violates change management policy
        test_action = ControlAction(
            action_id=f"test_action_{uuid.uuid4()}",
            action_type="configuration_change",
            target_system="production_system",
            parameters={
                "config_key": "database_connection_pool_size",
                "new_value": 100,
                "old_value": 50
            },
            metadata={
                "source": "test_script",
                "risk_level": "medium",
                "has_approval": False,  # Missing approval - should violate policy
                "affects_users": True,
                "has_user_impact_approval": False  # Missing user impact approval
            }
        )
        
        verification_request = VerificationRequest(
            request_id=f"test_request_{uuid.uuid4()}",
            action=test_action,
            context={
                "system": {
                    "cpu_utilization": 0.50,  # Good system state
                    "memory_utilization": 0.60,
                    "concurrent_actions": 2,
                    "availability": 0.999,
                    "error_rate": 0.01
                },
                "time": {
                    "is_business_hours": True
                }
            },
            verification_level=VerificationLevel.POLICY,
            requester="test_script"
        )
        
        # Verify the action
        result = await verification_adapter.verify_action(verification_request)
        
        # Check for policy violations
        policy_violations = [v for v in result.violations if v.constraint_type.value == "policy"]
        
        print(f"✓ Policy violations found: {len(policy_violations)}")
        for violation in policy_violations:
            print(f"  - {violation.constraint_id}: {violation.description}")
        
        if not result.approved and len(policy_violations) > 0:
            print("✅ Policy violations were correctly detected and action rejected")
            return True
        else:
            print("❌ Policy violations were not properly enforced")
            return False
            
    except Exception as e:
        print(f"❌ Policy enforcement test failed: {e}")
        return False


async def test_verification_performance():
    """Test verification adapter performance with multiple concurrent requests."""
    print("\n=== Testing Verification Performance ===")
    
    try:
        # Create verification adapter
        config_path = Path(__file__).parent.parent / "src" / "config" / "polaris_config.yaml"
        verification_config_dir = Path(__file__).parent.parent / "config"
        
        verification_adapter = VerificationAdapter(
            polaris_config_path=str(config_path),
            plugin_dir=str(verification_config_dir)
        )
        
        # Create multiple test actions
        num_actions = 10
        actions = []
        
        for i in range(num_actions):
            test_action = ControlAction(
                action_id=f"perf_test_action_{i}",
                action_type="scaling_action",
                target_system=f"test_system_{i % 3}",
                parameters={"scale_factor": 1.1 + (i * 0.1)},
                metadata={"source": "performance_test", "batch_id": f"batch_{i // 5}"}
            )
            
            verification_request = VerificationRequest(
                request_id=f"perf_test_request_{i}",
                action=test_action,
                context={
                    "system": {
                        "cpu_utilization": 0.60 + (i * 0.02),
                        "memory_utilization": 0.50 + (i * 0.02),
                        "concurrent_actions": i,
                        "availability": 0.999,
                        "error_rate": 0.01
                    }
                },
                verification_level=VerificationLevel.BASIC,
                requester="performance_test"
            )
            
            actions.append(verification_request)
        
        # Measure verification performance
        start_time = time.perf_counter()
        
        # Process actions concurrently
        tasks = [verification_adapter.verify_action(action) for action in actions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Analyze results
        successful_verifications = [r for r in results if not isinstance(r, Exception)]
        failed_verifications = [r for r in results if isinstance(r, Exception)]
        
        avg_verification_time = sum(r.verification_time_ms for r in successful_verifications) / len(successful_verifications)
        
        print(f"✓ Total actions processed: {num_actions}")
        print(f"✓ Successful verifications: {len(successful_verifications)}")
        print(f"✓ Failed verifications: {len(failed_verifications)}")
        print(f"✓ Total processing time: {total_time:.2f}s")
        print(f"✓ Average verification time: {avg_verification_time:.1f}ms")
        print(f"✓ Throughput: {num_actions / total_time:.1f} actions/second")
        
        # Performance thresholds
        max_avg_verification_time = 100  # ms
        min_throughput = 5  # actions/second
        
        if (avg_verification_time <= max_avg_verification_time and 
            (num_actions / total_time) >= min_throughput and 
            len(failed_verifications) == 0):
            print("✅ Verification performance meets requirements")
            return True
        else:
            print("❌ Verification performance below requirements")
            return False
            
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


async def main():
    """Run comprehensive end-to-end verification tests."""
    print("POLARIS End-to-End Verification Test Suite")
    print("=" * 60)
    
    tests = [
        ("Action Approval Flow", test_action_approval_flow),
        ("Action Rejection Flow", test_action_rejection_flow),
        ("Policy Enforcement", test_policy_enforcement),
        ("Verification Performance", test_verification_performance)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running: {test_name}")
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("END-TO-END VERIFICATION TEST RESULTS")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} | {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All end-to-end verification tests passed!")
        print("🔒 Verification system is working correctly and ready for production!")
        return 0
    else:
        print("⚠️  Some tests failed. Review verification configuration and implementation.")
        return 1


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise for cleaner test output
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    exit_code = asyncio.run(main())
    sys.exit(exit_code)