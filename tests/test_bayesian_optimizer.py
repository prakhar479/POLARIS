"""Tests for Bayesian optimization components."""

import numpy as np
import pytest

from polaris.meta_learner.bayesian_optimizer import (
    AcquisitionFunction,
    GaussianProcessOptimizer,
    ParameterConfiguration,
    ParameterSpace,
    ParameterType,
)


class TestParameterSpace:
    """Test ParameterSpace validation and functionality."""

    def test_continuous_parameter_space_validation(self):
        """Continuous parameters need min and max values."""
        with pytest.raises(ValueError, match="needs min_value and max_value"):
            ParameterSpace(
                name="test_param", param_type=ParameterType.CONTINUOUS, current_value=1.0
            )

    def test_discrete_parameter_space_validation(self):
        """Discrete parameters need min and max values."""
        with pytest.raises(ValueError, match="needs min_value and max_value"):
            ParameterSpace(name="test_param", param_type=ParameterType.DISCRETE, current_value=1)

    def test_categorical_parameter_space_validation(self):
        """Categorical parameters need allowed values."""
        with pytest.raises(ValueError, match="needs allowed_values"):
            ParameterSpace(
                name="test_param", param_type=ParameterType.CATEGORICAL, current_value="option1"
            )

    def test_valid_parameter_spaces(self):
        """Valid parameter spaces should not raise exceptions."""
        # Continuous
        ps_cont = ParameterSpace(
            name="continuous_param",
            param_type=ParameterType.CONTINUOUS,
            min_value=0.0,
            max_value=10.0,
            current_value=5.0,
        )
        assert ps_cont.name == "continuous_param"
        assert ps_cont.param_type == ParameterType.CONTINUOUS

        # Discrete
        ps_disc = ParameterSpace(
            name="discrete_param",
            param_type=ParameterType.DISCRETE,
            min_value=1,
            max_value=10,
            current_value=5,
        )
        assert ps_disc.param_type == ParameterType.DISCRETE

        # Categorical
        ps_cat = ParameterSpace(
            name="categorical_param",
            param_type=ParameterType.CATEGORICAL,
            allowed_values=["a", "b", "c"],
            current_value="b",
        )
        assert ps_cat.param_type == ParameterType.CATEGORICAL


class TestGaussianProcessOptimizer:
    """Test Gaussian Process optimizer functionality."""

    @pytest.fixture
    def simple_parameter_spaces(self):
        """Create simple parameter spaces for testing."""
        return [
            ParameterSpace(
                name="threshold",
                param_type=ParameterType.CONTINUOUS,
                min_value=0.0,
                max_value=100.0,
                current_value=50.0,
            ),
            ParameterSpace(
                name="cooldown",
                param_type=ParameterType.DISCRETE,
                min_value=10,
                max_value=300,
                current_value=60,
            ),
        ]

    @pytest.fixture
    def optimizer(self, simple_parameter_spaces):
        """Create optimizer for testing."""
        return GaussianProcessOptimizer(
            parameter_spaces=simple_parameter_spaces,
            acquisition_function=AcquisitionFunction.EXPECTED_IMPROVEMENT,
            min_samples_for_optimization=5,
        )

    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.parameter_spaces is not None
        assert len(optimizer.parameter_spaces) == 2
        assert optimizer.acquisition_function == AcquisitionFunction.EXPECTED_IMPROVEMENT
        assert optimizer.min_samples_for_optimization == 5
        assert not optimizer.is_trained

    def test_parameter_normalization_continuous(self, optimizer):
        """Test normalization of continuous parameters."""
        params = {"threshold": 25.0, "cooldown": 100}
        normalized = optimizer._normalize_parameters(params)

        # threshold: (25 - 0) / (100 - 0) = 0.25
        # cooldown: (100 - 10) / (300 - 10) = 90 / 290 ≈ 0.31
        assert len(normalized) == 2
        assert abs(normalized[0] - 0.25) < 1e-6
        assert abs(normalized[1] - 0.31) < 1e-2

    def test_parameter_denormalization_continuous(self, optimizer):
        """Test denormalization of continuous parameters."""
        normalized = np.array([0.25, 0.31])
        params = optimizer._denormalize_parameters(normalized)

        # threshold: 0.25 * (100 - 0) + 0 = 25
        # cooldown: round(0.31 * (300 - 10) + 10) = round(99.9) = 100
        assert abs(params["threshold"] - 25.0) < 1e-6
        assert params["cooldown"] == 100

    def test_fit_with_insufficient_data(self, optimizer):
        """Test fitting with insufficient data."""
        configs = [
            ParameterConfiguration(parameters={"threshold": 50.0, "cooldown": 60}, performance=0.7)
        ]

        # Should return False with insufficient data
        result = optimizer.fit(configs)
        assert not result
        assert not optimizer.is_trained

    def test_fit_with_sufficient_data(self, optimizer):
        """Test fitting with sufficient data."""
        configs = [
            ParameterConfiguration(parameters={"threshold": 30.0, "cooldown": 50}, performance=0.6),
            ParameterConfiguration(
                parameters={"threshold": 70.0, "cooldown": 100}, performance=0.8
            ),
            ParameterConfiguration(parameters={"threshold": 50.0, "cooldown": 75}, performance=0.7),
            ParameterConfiguration(
                parameters={"threshold": 40.0, "cooldown": 60}, performance=0.65
            ),
            ParameterConfiguration(
                parameters={"threshold": 60.0, "cooldown": 90}, performance=0.75
            ),
        ]

        # Should return True with sufficient data
        result = optimizer.fit(configs)
        assert result
        assert optimizer.is_trained
        assert optimizer.X_train is not None
        assert optimizer.y_train is not None
        assert len(optimizer.X_train) == 5

    def test_predictions_when_untrained(self, optimizer):
        """Test predictions when model is not trained."""
        X = np.array([[0.5, 0.5]])
        mean, std = optimizer.predict(X)

        # Should return default predictions
        assert len(mean) == 1
        assert len(std) == 1
        assert mean[0] == 0.0
        assert std[0] == 1.0

    def test_predictions_when_trained(self, optimizer):
        """Test predictions when model is trained."""
        # Train the model first
        configs = [
            ParameterConfiguration(parameters={"threshold": 30.0, "cooldown": 50}, performance=0.6),
            ParameterConfiguration(
                parameters={"threshold": 70.0, "cooldown": 100}, performance=0.8
            ),
            ParameterConfiguration(parameters={"threshold": 50.0, "cooldown": 75}, performance=0.7),
            ParameterConfiguration(
                parameters={"threshold": 40.0, "cooldown": 60}, performance=0.65
            ),
            ParameterConfiguration(
                parameters={"threshold": 60.0, "cooldown": 90}, performance=0.75
            ),
        ]
        optimizer.fit(configs)

        # Test predictions
        X = np.array([[0.5, 0.5]])  # Middle of parameter space
        mean, std = optimizer.predict(X)

        # Should return reasonable predictions
        assert len(mean) == 1
        assert len(std) == 1
        assert 0.0 <= mean[0] <= 1.0  # Performance should be in valid range
        assert std[0] >= 0.0  # Standard deviation should be non-negative

    def test_acquisition_functions(self, optimizer):
        """Test different acquisition functions."""
        # Train the model first
        configs = [
            ParameterConfiguration(parameters={"threshold": 30.0, "cooldown": 50}, performance=0.6),
            ParameterConfiguration(
                parameters={"threshold": 70.0, "cooldown": 100}, performance=0.8
            ),
            ParameterConfiguration(parameters={"threshold": 50.0, "cooldown": 75}, performance=0.7),
            ParameterConfiguration(
                parameters={"threshold": 40.0, "cooldown": 60}, performance=0.65
            ),
            ParameterConfiguration(
                parameters={"threshold": 60.0, "cooldown": 90}, performance=0.75
            ),
        ]
        optimizer.fit(configs)

        X = np.array([[0.5, 0.5]])

        # Test Expected Improvement
        optimizer.acquisition_function = AcquisitionFunction.EXPECTED_IMPROVEMENT
        ei_values = optimizer._expected_improvement(X)
        assert len(ei_values) == 1
        assert ei_values[0] >= 0.0

        # Test Upper Confidence Bound
        optimizer.acquisition_function = AcquisitionFunction.UPPER_CONFIDENCE_BOUND
        ucb_values = optimizer._upper_confidence_bound(X)
        assert len(ucb_values) == 1
        assert isinstance(ucb_values[0], (int, float))

        # Test Probability of Improvement
        optimizer.acquisition_function = AcquisitionFunction.PROBABILITY_IMPROVEMENT
        pi_values = optimizer._probability_improvement(X)
        assert len(pi_values) == 1
        assert 0.0 <= pi_values[0] <= 1.0

    def test_suggest_parameters_when_untrained(self, optimizer):
        """Test parameter suggestions when model is not trained."""
        suggestions = optimizer.suggest_next_parameters(n_suggestions=2)

        # Should return random suggestions
        assert len(suggestions) == 2
        for suggestion in suggestions:
            assert "threshold" in suggestion
            assert "cooldown" in suggestion
            assert isinstance(suggestion["threshold"], (int, float))
            assert isinstance(suggestion["cooldown"], int)

    def test_suggest_parameters_when_trained(self, optimizer):
        """Test parameter suggestions when model is trained."""
        # Train the model first
        configs = [
            ParameterConfiguration(parameters={"threshold": 30.0, "cooldown": 50}, performance=0.6),
            ParameterConfiguration(
                parameters={"threshold": 70.0, "cooldown": 100}, performance=0.8
            ),
            ParameterConfiguration(parameters={"threshold": 50.0, "cooldown": 75}, performance=0.7),
            ParameterConfiguration(
                parameters={"threshold": 40.0, "cooldown": 60}, performance=0.65
            ),
            ParameterConfiguration(
                parameters={"threshold": 60.0, "cooldown": 90}, performance=0.75
            ),
        ]
        optimizer.fit(configs)

        suggestions = optimizer.suggest_next_parameters(n_suggestions=2)

        # Should return informed suggestions
        assert len(suggestions) == 2
        for suggestion in suggestions:
            assert "threshold" in suggestion
            assert "cooldown" in suggestion
            # Values should be within bounds
            assert 0.0 <= suggestion["threshold"] <= 100.0
            assert 10 <= suggestion["cooldown"] <= 300

    def test_optimization_confidence(self, optimizer):
        """Test optimization confidence calculation."""
        # When not trained, confidence should be 0
        confidence = optimizer.get_optimization_confidence()
        assert confidence == 0.0

        # When trained with few samples, confidence should be low
        configs = [
            ParameterConfiguration(parameters={"threshold": 50.0, "cooldown": 75}, performance=0.7)
        ]
        optimizer.min_samples_for_optimization = 1
        optimizer.fit(configs)

        confidence = optimizer.get_optimization_confidence()
        assert 0.0 < confidence < 1.0

        # When trained with many samples, confidence should be higher
        more_configs = configs * 20  # Duplicate to get more samples
        optimizer.fit(more_configs)

        confidence = optimizer.get_optimization_confidence()
        assert confidence > 0.0

    def test_categorical_parameters(self):
        """Test optimizer with categorical parameters."""
        cat_spaces = [
            ParameterSpace(
                name="strategy",
                param_type=ParameterType.CATEGORICAL,
                allowed_values=["conservative", "aggressive", "balanced"],
                current_value="balanced",
            )
        ]

        optimizer = GaussianProcessOptimizer(
            parameter_spaces=cat_spaces,
            acquisition_function=AcquisitionFunction.EXPECTED_IMPROVEMENT,
        )

        # Test normalization/denormalization
        params = {"strategy": "aggressive"}
        normalized = optimizer._normalize_parameters(params)
        assert len(normalized) == 3  # One-hot encoding
        assert normalized[1] == 1.0  # "aggressive" is second option
        assert normalized[0] == 0.0
        assert normalized[2] == 0.0

        denormalized = optimizer._denormalize_parameters(normalized)
        assert denormalized["strategy"] == "aggressive"


class TestParameterConfiguration:
    """Test ParameterConfiguration data class."""

    def test_parameter_configuration_creation(self):
        """Test creating parameter configurations."""
        config = ParameterConfiguration(
            parameters={"threshold": 50.0, "cooldown": 60}, performance=0.7, metadata={"test": True}
        )

        assert config.parameters == {"threshold": 50.0, "cooldown": 60}
        assert config.performance == 0.7
        assert config.metadata == {"test": True}

    def test_parameter_configuration_without_metadata(self):
        """Test creating parameter configurations without metadata."""
        config = ParameterConfiguration(parameters={"threshold": 50.0}, performance=0.8)

        assert config.parameters == {"threshold": 50.0}
        assert config.performance == 0.8
        assert config.metadata is None
