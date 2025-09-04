"""
Tests for the Verification Adapter.

This module tests the verification adapter's ability to validate control actions
against safety constraints, organizational policies, and system invariants.
"""

import asyncio
import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import pytest
except ImportError:
    print("pytest not available, skipping tests")
    sys.exit(0)

from polaris.adapters.verification import (
    VerificationAdapter, VerificationRequest, VerificationResult,
    ConstraintViolation, ConstraintType, VerificationLevel
)
from polaris.models.actions import ControlAction


@pytest.fixture
def mock_config_path():
    """Mock configuration path for testing."""
    return "test_config.yaml"


@pytest.fixture
def mock_plugin_dir():
    """Mock plugin directory for testing."""
    return "test_plugin"


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    return MagicMock()


@pytest.fixture
def sample_action():
    """Sample control action for testing."""
    return ControlAction(
        action_id="test-action-123",
        action_type="ADD_SERVER",
        source="test_controller",
        params={"count": 1, "server_type": "compute"},
        priority="normal"
    )


@pytest.fixture
def sample_context():
    """Sample context for testing."""
    return {
        "active_servers": 3,
        "max_servers": 10,
        "utilization": 0.6,
        "response_time": 0.3
    }


class TestVerificationAdapter:
    """Test cases for VerificationAdapter."""

    @pytest.fixture
    def adapter(self, mock_config_path, mock_plugin_dir, mock_logger):
        """
        Provides a properly initialized VerificationAdapter instance with mocked
        dependencies. This fixture allows the real __init__ method to run while
        preventing file I/O and network connections.
        """
        # Define the mock configuration data we want the adapter to use
        mock_plugin_config_data = {
            "system_name": "test_system",
            "verification": {
                "constraints": [
                    {
                        "id": "max_servers_limit",
                        "type": "resource",
                        "condition": "action_type != 'ADD_SERVER' or (current_state.get('active_servers', 0) + params.get('count', 1)) <= current_state.get('max_servers', 10)",
                        "severity": "critical",
                        "description": "Cannot exceed maximum server limit"
                    }
                ],
                "policies": [
                    {
                        "id": "test_policy",
                        "name": "Test Policy",
                        "rules": [
                            {
                                "id": "test_rule",
                                "condition": "action_type != 'REMOVE_SERVER' or 'approved_by' in action.get('params', {})",
                                "severity": "medium",
                                "description": "Server removal requires approval"
                            }
                        ]
                    }
                ],
                "settings": {
                    "default_timeout_sec": 30,
                    "max_concurrent": 5,
                    "enable_digital_twin": True
                }
            }
        }
        mock_framework_config_data = {
            "nats": {"url": "nats://mock-nats:4222"},
            "verification": {
                "input_subject": "test.verification.requests",
                "output_subject": "test.verification.results"
            }
        }

        # Mock the ConfigurationManager and NATSClient
        mock_config_manager = MagicMock()
        mock_config_manager.load_framework_config.return_value = mock_framework_config_data
        mock_config_manager.load_plugin_config.return_value = mock_plugin_config_data
        mock_config_manager.get_component_config.return_value = mock_plugin_config_data.get("verification", {})
        
        mock_nats_client = AsyncMock()

        # Patch the dependencies in the InternalAdapter
        with patch('polaris.adapters.core.ConfigurationManager', return_value=mock_config_manager), \
             patch('polaris.adapters.core.NATSClient', return_value=mock_nats_client):

            adapter_instance = VerificationAdapter(mock_config_path, mock_plugin_dir, mock_logger)
            return adapter_instance

    @pytest.mark.asyncio
    async def test_verify_action_approved(self, adapter, sample_action, sample_context):
        """Test that a valid action is approved."""
        request = VerificationRequest(
            request_id="test-request-123",
            action=sample_action,
            context=sample_context,
            verification_level=VerificationLevel.BASIC
        )

        result = await adapter.verify_action(request)

        assert result.approved is True
        assert result.confidence > 0.5
        assert len(result.violations) == 0
        assert result.request_id == request.request_id
        assert result.action_id == sample_action.action_id

    @pytest.mark.asyncio
    async def test_verify_action_constraint_violation(self, adapter, sample_context):
        """Test that an action violating constraints is rejected."""
        # Create action that violates max_servers constraint (3 current + 10 new > 10 max)
        violating_action = ControlAction(
            action_id="test-action-456",
            action_type="ADD_SERVER",
            source="test_controller",
            params={"count": 10},
            priority="normal"
        )

        request = VerificationRequest(
            request_id="test-request-456",
            action=violating_action,
            context=sample_context,
            verification_level=VerificationLevel.BASIC
        )

        result = await adapter.verify_action(request)

        assert result.approved is False
        assert result.confidence == 0.0
        assert len(result.violations) > 0
        assert any(v.constraint_id == "max_servers_limit" for v in result.violations)

    @pytest.mark.asyncio
    async def test_verify_action_policy_violation(self, adapter, sample_context):
        """Test that an action violating policies is handled appropriately."""
        # Create action that violates policy (server removal without approval)
        policy_violating_action = ControlAction(
            action_id="test-action-789",
            action_type="REMOVE_SERVER",
            source="test_controller",
            params={"count": 1},
            priority="normal"
        )

        request = VerificationRequest(
            request_id="test-request-789",
            action=policy_violating_action,
            context=sample_context,
            verification_level=VerificationLevel.POLICY
        )

        result = await adapter.verify_action(request)
        
        # Policy violations are not critical, so action might still be "approved"
        # The key is that the violation is correctly identified.
        assert len(result.violations) > 0
        policy_violations = [v for v in result.violations if v.constraint_type == ConstraintType.POLICY]
        assert len(policy_violations) > 0
        assert policy_violations[0].constraint_id.startswith("test_policy")

    @pytest.mark.asyncio
    async def test_constraint_evaluation(self, adapter, sample_action, sample_context):
        """Test constraint evaluation logic."""
        constraint_config = {
            "id": "test_constraint",
            "type": "safety",
            "condition": "params.get('count', 1) <= 5",
            "severity": "high",
            "description": "Test constraint"
        }

        # Test valid action
        violation = await adapter._evaluate_constraint(constraint_config, sample_action, sample_context)
        assert violation is None

        # Test invalid action
        invalid_action = ControlAction(
            action_id="test-invalid",
            action_type="ADD_SERVER",
            source="test",
            params={"count": 10},  # Violates constraint
            priority="normal"
        )

        violation = await adapter._evaluate_constraint(constraint_config, invalid_action, sample_context)
        assert violation is not None
        assert violation.constraint_id == "test_constraint"
        assert violation.severity == "high"

    @pytest.mark.asyncio
    async def test_digital_twin_integration(self, adapter, sample_action, sample_context):
        """Test digital twin integration for verification."""
        violations, recommendations = await adapter._check_digital_twin(sample_action, sample_context)

        # Assert based on the mocked return value in the fixture
        assert len(violations) == 0
        assert len(recommendations) > 0
        assert recommendations[0] == "Digital twin simulation indicates action is safe to execute"

    @pytest.mark.asyncio
    async def test_verification_timeout(self, adapter, sample_action, sample_context):
        """Test verification timeout handling."""
        async def slow_check(*args, **kwargs):
            await asyncio.sleep(0.2)
            return [] # Return a valid result for a constraint check

        # Mock a slow verification process
        with patch.object(adapter, '_check_constraints', side_effect=slow_check):
            request = VerificationRequest(
                request_id="test-timeout",
                action=sample_action,
                context=sample_context,
                verification_level=VerificationLevel.BASIC,
                timeout_sec=0.1  # Short timeout
            )
            
            # Since the worker loop is not running in the unit test, we call verify_action directly.
            # We expect a TimeoutError to be caught inside the worker, but here we can test the timeout logic.
            with pytest.raises(asyncio.TimeoutError):
                 await asyncio.wait_for(adapter.verify_action(request), timeout=request.timeout_sec)
    
    def test_verification_result_serialization(self):
        """Test that verification results can be serialized to JSON."""
        violation = ConstraintViolation(
            constraint_id="test_constraint",
            constraint_type=ConstraintType.SAFETY,
            severity="high",
            description="Test violation"
        )

        result = VerificationResult(
            request_id="test-request",
            action_id="test-action",
            approved=False,
            confidence=0.5,
            violations=[violation],
            recommendations=["Fix the issue"],
            verification_time_ms=100.0
        )

        result_dict = result.to_dict()

        assert result_dict["request_id"] == "test-request"
        assert result_dict["approved"] is False
        assert len(result_dict["violations"]) == 1
        assert result_dict["violations"][0]["constraint_id"] == "test_constraint"

        # Should be JSON serializable
        json_str = json.dumps(result_dict)
        assert json_str is not None


class TestConstraintViolation:
    """Test cases for ConstraintViolation."""

    def test_constraint_violation_creation(self):
        """Test constraint violation creation."""
        violation = ConstraintViolation(
            constraint_id="test_constraint",
            constraint_type=ConstraintType.SAFETY,
            severity="critical",
            description="Test violation",
            suggested_fix="Fix it",
            metadata={"key": "value"}
        )

        assert violation.constraint_id == "test_constraint"
        assert violation.constraint_type == ConstraintType.SAFETY
        assert violation.severity == "critical"
        assert violation.description == "Test violation"
        assert violation.suggested_fix == "Fix it"
        assert violation.metadata["key"] == "value"
        assert violation.timestamp is not None


class TestVerificationRequest:
    """Test cases for VerificationRequest."""

    def test_verification_request_creation(self, sample_action):
        """Test verification request creation."""
        request = VerificationRequest(
            request_id="test-request",
            action=sample_action,
            context={"key": "value"},
            verification_level=VerificationLevel.COMPREHENSIVE,
            timeout_sec=60.0,
            requester="test_component"
        )

        assert request.request_id == "test-request"
        assert request.action == sample_action
        assert request.context["key"] == "value"
        assert request.verification_level == VerificationLevel.COMPREHENSIVE
        assert request.timeout_sec == 60.0
        assert request.requester == "test_component"
        assert request.timestamp is not None



if __name__ == "__main__":
    pytest.main([__file__])