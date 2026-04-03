"""Bayesian optimization for intelligent parameter tuning.

Uses Gaussian Processes and acquisition functions to model parameter-performance
relationships and suggest optimal parameter configurations.
"""

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
from scipy.stats import norm

# Suppress scipy warnings for cleaner logs
warnings.filterwarnings("ignore", category=RuntimeWarning)


class AcquisitionFunction(str, Enum):
    """Acquisition function types for Bayesian optimization."""

    EXPECTED_IMPROVEMENT = "expected_improvement"
    UPPER_CONFIDENCE_BOUND = "upper_confidence_bound"
    PROBABILITY_IMPROVEMENT = "probability_improvement"


class ParameterType(str, Enum):
    """Parameter types for optimization."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"


@dataclass
class ParameterSpace:
    """Definition of a parameter's optimization space."""

    name: str
    param_type: ParameterType
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    current_value: Optional[Any] = None

    def __post_init__(self) -> None:
        """Validate parameter space definition."""
        if self.param_type == ParameterType.CONTINUOUS:
            if self.min_value is None or self.max_value is None:
                raise ValueError(f"Continuous parameter {self.name} needs min_value and max_value")
        elif self.param_type == ParameterType.DISCRETE:
            if self.min_value is None or self.max_value is None:
                raise ValueError(f"Discrete parameter {self.name} needs min_value and max_value")
        elif self.param_type == ParameterType.CATEGORICAL:
            if not self.allowed_values:
                raise ValueError(f"Categorical parameter {self.name} needs allowed_values")


@dataclass
class ParameterConfiguration:
    """A specific parameter configuration with performance metrics."""

    parameters: Dict[str, Any]
    performance: float  # Higher is better (e.g., success rate)
    metadata: Optional[Dict[str, Any]] = None


class GaussianProcessOptimizer:
    """Gaussian Process-based Bayesian optimizer.

    Uses RBF kernel with automatic relevance determination for modeling parameter-
    performance relationships.
    """

    def __init__(
        self,
        parameter_spaces: List[ParameterSpace],
        acquisition_function: AcquisitionFunction = AcquisitionFunction.EXPECTED_IMPROVEMENT,
        exploration_weight: float = 0.1,
        noise_level: float = 1e-6,
        length_scale: Optional[float] = None,
        min_samples_for_optimization: int = 10,
    ):
        """Initialize Gaussian Process optimizer.

        Args:
            parameter_spaces: List of parameter spaces to optimize
            acquisition_function: Acquisition function for exploration/exploitation
            exploration_weight: Weight for exploration in UCB acquisition
            noise_level: Noise level for GP observations
            length_scale: Initial length scale for RBF kernel (None for auto)
            min_samples_for_optimization: Minimum samples before optimization
        """
        self.parameter_spaces = parameter_spaces
        self.acquisition_function = acquisition_function
        self.exploration_weight = exploration_weight
        self.noise_level = noise_level
        self.length_scale = length_scale
        self.min_samples_for_optimization = min_samples_for_optimization

        # Training data
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.parameter_names: List[str] = [ps.name for ps in parameter_spaces]

        # GP hyperparameters
        self.kernel_length_scales: Optional[np.ndarray] = None
        self.signal_variance: float = 1.0
        self.is_trained: bool = False

    def _normalize_parameters(self, parameters: Dict[str, Any]) -> NDArray[np.float64]:
        """Normalize parameters to [0, 1] range for GP optimization."""
        normalized = []

        for param_space in self.parameter_spaces:
            value = parameters.get(param_space.name)
            if value is None:
                raise ValueError(f"Parameter {param_space.name} not found inside configuration")

            if param_space.param_type == ParameterType.CONTINUOUS:
                # Normalize to [0, 1]
                if param_space.min_value is not None and param_space.max_value is not None:
                    norm_value = (value - param_space.min_value) / (
                        param_space.max_value - param_space.min_value
                    )
                    normalized.append(np.clip(norm_value, 0, 1))

            elif param_space.param_type == ParameterType.DISCRETE:
                # Normalize to [0, 1]
                if param_space.min_value is not None and param_space.max_value is not None:
                    norm_value = (value - param_space.min_value) / (
                        param_space.max_value - param_space.min_value
                    )
                    normalized.append(np.clip(norm_value, 0, 1))

            elif param_space.param_type == ParameterType.CATEGORICAL:
                # One-hot encoding for categorical
                if param_space.allowed_values:
                    for allowed_val in param_space.allowed_values:
                        normalized.append(1.0 if value == allowed_val else 0.0)

        return np.array(normalized, dtype=float)

    def _denormalize_parameters(self, normalized_params: NDArray[np.float64]) -> Dict[str, Any]:
        """Convert normalized parameters back to original scale."""
        parameters = {}
        idx = 0

        # Handle both 1D and 2D arrays
        if normalized_params.ndim == 2:
            # Take the first (and only) row for single parameter set
            if normalized_params.shape[0] == 1:
                normalized_params = normalized_params[0]
            else:
                raise ValueError("Multiple parameter sets not supported in _denormalize_parameters")

        for param_space in self.parameter_spaces:
            if param_space.param_type == ParameterType.CONTINUOUS:
                if idx >= len(normalized_params):
                    break  # Safety check
                norm_value = normalized_params[idx]
                if param_space.min_value is not None and param_space.max_value is not None:
                    value = (
                        norm_value * (param_space.max_value - param_space.min_value)
                        + param_space.min_value
                    )
                    parameters[param_space.name] = value
                idx += 1

            elif param_space.param_type == ParameterType.DISCRETE:
                if idx >= len(normalized_params):
                    break  # Safety check
                norm_value = normalized_params[idx]
                if param_space.min_value is not None and param_space.max_value is not None:
                    value = (
                        norm_value * (param_space.max_value - param_space.min_value)
                        + param_space.min_value
                    )
                    # Round to nearest integer for discrete
                    parameters[param_space.name] = int(round(value))
                idx += 1

            elif param_space.param_type == ParameterType.CATEGORICAL:
                if param_space.allowed_values and idx + len(param_space.allowed_values) <= len(
                    normalized_params
                ):
                    # Find the category with highest probability
                    one_hot = normalized_params[idx : idx + len(param_space.allowed_values)]
                    max_idx = np.argmax(one_hot)
                    parameters[param_space.name] = param_space.allowed_values[max_idx]
                    idx += len(param_space.allowed_values)

        return parameters

    def _rbf_kernel(
        self, X1: NDArray[np.float64], X2: NDArray[np.float64], length_scales: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """RBF kernel with automatic relevance determination."""
        if self.kernel_length_scales is None:
            # Use default length scales if not trained
            n_features = X1.shape[1]
            length_scales = np.ones(n_features)
            if self.length_scale is not None:
                length_scales *= self.length_scale
        else:
            length_scales = self.kernel_length_scales

        # Scale features
        X1_scaled = X1 / length_scales
        X2_scaled = X2 / length_scales

        # Compute squared Euclidean distances
        distances = cdist(X1_scaled, X2_scaled, "sqeuclidean")

        # Apply RBF kernel
        K = self.signal_variance * np.exp(-0.5 * distances)

        return np.asarray(K, dtype=np.float64)

    def fit(self, configurations: List[ParameterConfiguration]) -> bool:
        """Fit Gaussian Process to historical data.

        Args:
            configurations: List of parameter configurations with performance

        Returns:
            True if fitting was successful, False otherwise
        """
        if len(configurations) < self.min_samples_for_optimization:
            return False

        # Extract and normalize training data
        X_list = []
        y_list = []

        for config in configurations:
            try:
                X_norm = self._normalize_parameters(config.parameters)
                X_list.append(X_norm)
                y_list.append(config.performance)
            except ValueError:
                # Skip configurations with missing parameters
                continue

        if len(X_list) < self.min_samples_for_optimization:
            return False

        self.X_train = np.array(X_list)
        self.y_train = np.array(y_list)

        # Optimize kernel hyperparameters using simple heuristic
        self._optimize_hyperparameters()

        self.is_trained = True
        return True

    def _optimize_hyperparameters(self) -> None:
        """Optimize GP hyperparameters using simple approach."""
        if self.X_train is None or len(self.X_train) == 0:
            return

        n_features = self.X_train.shape[1]

        # Simple heuristic for length scales based on data range
        if self.length_scale is None:
            data_ranges = np.max(self.X_train, axis=0) - np.min(self.X_train, axis=0)
            # Avoid zero ranges
            data_ranges = np.where(data_ranges < 1e-6, 1.0, data_ranges)
            self.kernel_length_scales = data_ranges
        else:
            self.kernel_length_scales = np.ones(n_features) * self.length_scale

        # Estimate signal variance from data
        if self.y_train is not None and len(self.y_train) > 1:
            self.signal_variance = np.var(self.y_train) + self.noise_level
        else:
            self.signal_variance = 1.0

    def predict(self, X: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Predict performance and uncertainty for given parameter configurations.

        Args:
            X: Normalized parameter configurations

        Returns:
            Tuple of (mean predictions, standard deviations)
        """
        if not self.is_trained or self.X_train is None or self.y_train is None:
            # Return default predictions if not trained
            n_samples = X.shape[0]
            return np.zeros(n_samples), np.ones(n_samples)

        # Compute kernel matrices
        length_scales = (
            self.kernel_length_scales
            if self.kernel_length_scales is not None
            else np.ones(self.X_train.shape[1])
        )
        K_train = self._rbf_kernel(self.X_train, self.X_train, length_scales)
        K_test = self._rbf_kernel(X, self.X_train, length_scales)
        K_test_test = self._rbf_kernel(X, X, length_scales)

        # Add noise to diagonal for numerical stability
        K_train += np.eye(len(K_train)) * self.noise_level

        try:
            # Compute inverse of training kernel
            K_train_inv = np.linalg.inv(K_train + np.eye(len(K_train)) * 1e-6)

            # Predict mean
            mean = K_test @ K_train_inv @ self.y_train

            # Predict variance
            var = np.diag(K_test_test) - np.sum(K_test @ K_train_inv @ K_test.T, axis=1)
            var = np.maximum(var, self.noise_level)  # Ensure positive variance

            return mean, np.sqrt(var)

        except np.linalg.LinAlgError:
            # Fallback to simple predictions if matrix inversion fails
            n_samples = X.shape[0]
            return np.zeros(n_samples), np.ones(n_samples)

    def _expected_improvement(
        self, X: NDArray[np.float64], xi: float = 0.01
    ) -> NDArray[np.float64]:
        """Calculate expected improvement acquisition function."""
        mean, std = self.predict(X)

        # Best observed performance
        f_best = np.max(self.y_train) if self.y_train is not None else 0.0

        # Compute improvement
        improvement = mean - f_best - xi

        # Expected improvement
        with np.errstate(divide="warn", invalid="warn"):
            Z = improvement / std
            ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
            ei[std == 0.0] = 0.0

        return np.asarray(ei, dtype=np.float64)

    def _upper_confidence_bound(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Upper Confidence Bound acquisition function."""
        mean, std = self.predict(X)
        return mean + self.exploration_weight * std

    def _probability_improvement(
        self, X: NDArray[np.float64], xi: float = 0.01
    ) -> NDArray[np.float64]:
        """Probability of Improvement acquisition function."""
        mean, std = self.predict(X)

        # Best observed performance
        f_best = np.max(self.y_train) if self.y_train is not None else 0.0

        # Probability of improvement
        with np.errstate(divide="warn", invalid="warn"):
            Z = (mean - f_best - xi) / std
            pi = norm.cdf(Z)
            pi[std == 0.0] = 0.0

        return np.asarray(pi, dtype=np.float64)

    def _acquisition_function(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute acquisition function values."""
        if self.acquisition_function == AcquisitionFunction.EXPECTED_IMPROVEMENT:
            return self._expected_improvement(X)
        elif self.acquisition_function == AcquisitionFunction.UPPER_CONFIDENCE_BOUND:
            return self._upper_confidence_bound(X)
        elif self.acquisition_function == AcquisitionFunction.PROBABILITY_IMPROVEMENT:
            return self._probability_improvement(X)
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition_function}")

    def suggest_next_parameters(self, n_suggestions: int = 1) -> List[Dict[str, Any]]:
        """Suggest optimal parameter configurations using Bayesian optimization.

        Args:
            n_suggestions: Number of parameter suggestions to return

        Returns:
            List of suggested parameter configurations
        """
        if not self.is_trained:
            # Return random suggestions if not trained
            return self._random_suggestions(n_suggestions)

        # Generate candidate points (grid search + random)
        candidates = self._generate_candidates(n_candidates=1000)

        # Evaluate acquisition function
        acquisition_values = self._acquisition_function(candidates)

        # Get top candidates
        top_indices = np.argsort(acquisition_values)[-n_suggestions:][::-1]

        suggestions = []
        for idx in top_indices:
            params = self._denormalize_parameters(candidates[idx : idx + 1])
            suggestions.append(params)

        return suggestions

    def _generate_candidates(self, n_candidates: int = 1000) -> NDArray[np.float64]:
        """Generate candidate parameter configurations."""
        n_dims = sum(
            1 if ps.param_type != ParameterType.CATEGORICAL else len(ps.allowed_values or [])
            for ps in self.parameter_spaces
        )

        candidates = np.random.uniform(0, 1, (n_candidates, n_dims))

        # Add some grid points for better coverage
        n_grid = min(5, int(np.sqrt(n_candidates)))
        if n_grid > 1:
            grid_points = np.meshgrid(*[np.linspace(0, 1, n_grid) for _ in range(n_dims)])
            grid_candidates = np.column_stack([g.ravel() for g in grid_points])

            # Combine random and grid candidates
            candidates = np.vstack([candidates, grid_candidates[: n_candidates // 2]])

        return candidates.astype(float)

    def _random_suggestions(self, n_suggestions: int) -> List[Dict[str, Any]]:
        """Generate random parameter suggestions when model is not trained."""
        suggestions = []

        for _ in range(n_suggestions):
            params = {}
            for param_space in self.parameter_spaces:
                if param_space.param_type == ParameterType.CONTINUOUS:
                    if param_space.min_value is not None and param_space.max_value is not None:
                        value = np.random.uniform(param_space.min_value, param_space.max_value)
                    else:
                        value = 0.0
                elif param_space.param_type == ParameterType.DISCRETE:
                    if param_space.min_value is not None and param_space.max_value is not None:
                        value = np.random.randint(
                            int(param_space.min_value), int(param_space.max_value) + 1
                        )
                    else:
                        value = 0
                elif param_space.param_type == ParameterType.CATEGORICAL:
                    if param_space.allowed_values:
                        value = np.random.choice(param_space.allowed_values)
                    else:
                        value = None

                params[param_space.name] = value

            suggestions.append(params)

        return suggestions

    def get_optimization_confidence(self) -> float:
        """Get confidence in optimization quality based on data quality and model fit.

        Returns:
            Confidence score between 0 and 1
        """
        if not self.is_trained or self.X_train is None:
            return 0.0

        n_samples = len(self.X_train)

        # Base confidence from sample count
        sample_confidence = min(1.0, n_samples / 50.0)  # Full confidence at 50 samples

        # Adjust for performance variance (lower variance = higher confidence)
        if self.y_train is not None and len(self.y_train) > 1:
            performance_variance = np.var(self.y_train)
            variance_confidence = 1.0 / (1.0 + performance_variance)
        else:
            variance_confidence = 0.5

        # Combined confidence
        confidence = 0.6 * sample_confidence + 0.4 * variance_confidence

        return float(np.clip(confidence, 0.0, 1.0))
