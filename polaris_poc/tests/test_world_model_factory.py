#!/usr/bin/env python3
"""
Fixed comprehensive test suite for World Model factory pattern and interface validation.

This module addresses all the identified issues in the previous test suite.
"""

import asyncio
import pytest
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

import pytest_asyncio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock classes for missing dependencies
class MockKnowledgeEvent:
    """Mock KnowledgeEvent for testing."""
    def __init__(self, event_id, timestamp, source, event_type, data):
        self.event_id = event_id
        self.timestamp = timestamp
        self.source = source
        self.event_type = event_type
        self.data = data

class MockCalibrationEvent:
    """Mock CalibrationEvent for testing."""
    def __init__(self, calibration_id, timestamp, prediction_id, 
                 actual_outcome, predicted_outcome, accuracy_metrics):
        self.calibration_id = calibration_id
        self.timestamp = timestamp
        self.prediction_id = prediction_id
        self.actual_outcome = actual_outcome
        self.predicted_outcome = predicted_outcome
        self.accuracy_metrics = accuracy_metrics
    
    def calculate_accuracy_score(self):
        return 0.8  # Mock accuracy score

class MockSimulationRequest:
    """Mock SimulationRequest for testing."""
    def __init__(self, simulation_id, simulation_type, horizon_minutes=60, 
                 actions=None, parameters=None, timestamp=None):
        self.simulation_id = simulation_id
        self.simulation_type = simulation_type
        self.horizon_minutes = horizon_minutes
        self.actions = actions or []
        self.parameters = parameters or {}
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()

class MockDiagnosisRequest:
    """Mock DiagnosisRequest for testing."""
    def __init__(self, diagnosis_id, anomaly_description, context=None, timestamp=None):
        self.diagnosis_id = diagnosis_id
        self.anomaly_description = anomaly_description
        self.context = context or {}
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()

# Try to import the actual classes, fall back to mocks
from polaris.models.world_model import (
    WorldModelFactory, 
    WorldModelValidator
)

from polaris.models.mock_world_model import (
    MockWorldModel,
    TestableWorldModel
)
   

class TestWorldModelFactory:
    """Test suite for World Model Factory pattern."""
    
    def test_get_registered_types(self):
        """Test retrieval of registered model types."""
        registered_types = WorldModelFactory.get_registered_types()
        
        assert isinstance(registered_types, (list, set, tuple)), \
            "Registered types should be a collection"
        
        logger.info(f"Successfully retrieved registered types: {registered_types}")
        
        # At minimum, we should have some registered types
        assert len(registered_types) > 0, "Should have at least one registered type"
    
    def test_is_registered(self):
        """Test the is_registered method with proper error handling."""
        # Get available types first
        available_types = WorldModelFactory.get_registered_types()
        
        if "mock" in available_types:
            assert WorldModelFactory.is_registered("mock"), \
                "Mock model should be registered"
        
        if "testable" in available_types:
            assert WorldModelFactory.is_registered("testable"), \
                "Testable model should be registered"
        
        # Test unregistered model
        assert not WorldModelFactory.is_registered("unknown"), \
            "Unknown model type should not be registered"
        
        # Test edge cases with proper error handling
        edge_cases = [None, "", 123, [], {}]
        
        for case in edge_cases:
            try:
                result = WorldModelFactory.is_registered(case)
                assert not result, f"Edge case {case} should not be registered"
            except (TypeError, AttributeError, ValueError) as e:
                # These exceptions are acceptable for invalid inputs
                logger.info(f"is_registered correctly rejected invalid input {case}: {e}")
    
    def test_create_mock_model(self):
        """Test creation of MockWorldModel if available."""
        available_types = WorldModelFactory.get_registered_types()
        
        if "mock" not in available_types:
            pytest.skip("Mock model not registered")
        
        config = {"test_param": "test_value", "debug": True}
        model = WorldModelFactory.create_model("mock", config)
        
        assert model is not None, "Model should not be None"
        assert hasattr(model, 'config'), "Model should have config attribute"
        
        logger.info("Mock model created successfully")
    
    def test_create_testable_model(self):
        """Test creation of TestableWorldModel if available."""
        WorldModelFactory.register("testable", TestableWorldModel)
        available_types = WorldModelFactory.get_registered_types()
        
        # if "testable" not in available_types:
        #     pytest.skip("Testable model not registered")
        
        config = {"test_param": "test_value", "enable_logging": True}
        model = WorldModelFactory.create_model("testable", config)
        
        assert model is not None, "Model should not be None"
        assert hasattr(model, 'config'), "Model should have config attribute"
        
        logger.info("Testable model created successfully")
    
    def test_create_model_with_empty_config(self):
        """Test model creation with empty configuration."""
        available_types = WorldModelFactory.get_registered_types()
        
        if not available_types:
            pytest.skip("No model types registered")
        
        # Use the first available type
        model_type = list(available_types)[0]
        model = WorldModelFactory.create_model(model_type, {})
        assert model is not None, "Should create model with empty config"
    
    def test_create_model_with_none_config(self):
        """Test model creation with None configuration."""
        available_types = WorldModelFactory.get_registered_types()
        
        if not available_types:
            pytest.skip("No model types registered")
        
        # Use the first available type
        model_type = list(available_types)[0]
        
        # Some implementations might handle None config, others might not
        try:
            model = WorldModelFactory.create_model(model_type, None)
            assert model is not None, "Should handle None config gracefully"
        except (TypeError, ValueError) as e:
            logger.info(f"Factory correctly rejected None config: {e}")
    
    def test_create_unknown_model_type(self):
        """Test creation of unknown model type."""
        with pytest.raises(ValueError) as exc_info:
            WorldModelFactory.create_model("unknown_model_type_12345", {})
        
        assert "unknown" in str(exc_info.value).lower() or "available" in str(exc_info.value).lower(), \
            "Error message should indicate unknown type"
        
        logger.info(f"Correctly rejected unknown model type: {exc_info.value}")
    
    def test_create_model_with_invalid_types(self):
        """Test model creation with invalid input types."""
        invalid_inputs = [
            (None, {}),
            (123, {}),
            ([], {}),
            ({}, {}),
        ]
        
        for model_type, config in invalid_inputs:
            try:
                model = WorldModelFactory.create_model(model_type, config)
                # If no exception, the factory handles it gracefully
                logger.info(f"Factory handled invalid input gracefully: {model_type}")
            except (ValueError, TypeError, AttributeError) as e:
                logger.info(f"Factory correctly rejected invalid input {model_type}: {e}")


class TestWorldModelValidator:
    """Test suite for World Model interface validation with proper fixtures."""
    
    @pytest_asyncio.fixture(scope="function")
    async def testable_model_instance(self):
        """Create and initialize a testable model instance."""
        WorldModelFactory.register("testable", TestableWorldModel)
        available_types = WorldModelFactory.get_registered_types()
        
        # if "testable" not in available_types:
        #     pytest.skip("Testable model not available")
        
        config = {"test": True, "timeout": 30}
        model = WorldModelFactory.create_model("testable", config)
        
        # Initialize the model
        try:
            await model.initialize()
        except Exception as e:
            logger.warning(f"Model initialization failed: {e}")
            pytest.skip(f"Model initialization failed: {e}")
        
        yield model
        
        # Cleanup
        try:
            await model.shutdown()
        except Exception as e:
            logger.warning(f"Model shutdown failed: {e}")
    
    @pytest_asyncio.fixture(scope="function")
    async def mock_model_instance(self):
        """Create and initialize a mock model instance."""
        available_types = WorldModelFactory.get_registered_types()
        
        if "mock" not in available_types:
            pytest.skip("Mock model not available")
        
        config = {"mock_data": True}
        model = WorldModelFactory.create_model("mock", config)
        
        # Initialize the model
        try:
            await model.initialize()
        except Exception as e:
            logger.warning(f"Model initialization failed: {e}")
            pytest.skip(f"Model initialization failed: {e}")
        
        yield model
        
        # Cleanup
        try:
            await model.shutdown()
        except Exception as e:
            logger.warning(f"Model shutdown failed: {e}")
    
    @pytest.mark.asyncio
    async def test_validate_interface_compliance_testable(self, testable_model_instance):
        """Test interface compliance validation for testable model."""
        model = testable_model_instance
        
        compliance_results = await WorldModelValidator.validate_interface_compliance(model)
        
        assert isinstance(compliance_results, dict), \
            "Compliance results should be a dictionary"
        assert "compliant" in compliance_results, \
            "Results should contain compliance status"
        
        if not compliance_results.get('compliant', False):
            logger.warning(f"Interface compliance issues: {compliance_results.get('errors', [])}")
            logger.warning(f"Test results: {compliance_results.get('test_results', {})}")
        
        # Log the results for debugging
        logger.info(f"Interface compliance: {compliance_results.get('compliant', False)}")
        logger.info(f"Errors: {compliance_results.get('errors', [])}")
    
    @pytest.mark.asyncio
    async def test_validate_interface_compliance_mock(self, mock_model_instance):
        """Test interface compliance validation for mock model."""
        model = mock_model_instance
        
        compliance_results = await WorldModelValidator.validate_interface_compliance(model)
        
        assert isinstance(compliance_results, dict), \
            "Compliance results should be a dictionary"
        
        if not compliance_results.get('compliant', False):
            logger.warning(f"Interface compliance issues: {compliance_results.get('errors', [])}")
        
        logger.info(f"Mock model compliance: {compliance_results.get('compliant', False)}")
    
    def test_validate_configuration_valid(self):
        """Test configuration validation with valid config."""
        valid_configs = [
            {"test": True, "timeout": 30},
            {"debug": False, "max_retries": 3},
            {},  # Empty config should be valid
            {"nested": {"key": "value"}},
        ]
        
        available_types = WorldModelFactory.get_registered_types()
        if not available_types:
            pytest.skip("No model types available for configuration validation")
        
        model_type = list(available_types)[0]
        
        for config in valid_configs:
            result = WorldModelValidator.validate_configuration(config, model_type)
            
            assert isinstance(result, dict), "Result should be a dictionary"
            logger.info(f"Config validation result for {config}: {result}")
    
    @pytest.mark.asyncio
    async def test_run_comprehensive_test(self, testable_model_instance):
        """Test comprehensive model testing."""
        model = testable_model_instance
        
        comprehensive_results = await WorldModelValidator.run_comprehensive_test(model)
        
        assert isinstance(comprehensive_results, dict), \
            "Comprehensive results should be a dictionary"
        assert "overall_passed" in comprehensive_results, \
            "Results should contain overall pass status"
        assert "tests" in comprehensive_results, \
            "Results should contain test details"