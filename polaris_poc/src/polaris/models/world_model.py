"""
World Model interface and factory for POLARIS Digital Twin.

This module defines the abstract World Model interface that enables
implementation-agnostic Digital Twin functionality, supporting different
AI/ML approaches while maintaining consistent contracts.
"""

import abc
import logging
from typing import Any, Dict, Optional, Type, Union
from datetime import datetime, timezone

from .digital_twin_events import KnowledgeEvent, CalibrationEvent


class WorldModelError(Exception):
    """Base exception for World Model operations."""

    pass


class WorldModelInitializationError(WorldModelError):
    """Raised when World Model initialization fails."""

    pass


class WorldModelOperationError(WorldModelError):
    """Raised when World Model operations fail."""

    pass


class QueryRequest:
    """Request object for World Model queries."""

    def __init__(
        self,
        query_id: str,
        query_type: str,
        query_content: str,
        parameters: Optional[Dict[str, str]] = None,
        timestamp: Optional[str] = None,
    ):
        """Initialize query request.

        Args:
            query_id: Unique identifier for this query
            query_type: Type of query ('current_state', 'historical', 'natural_language')
            query_content: The actual query content
            parameters: Additional query parameters
            timestamp: Query timestamp (defaults to current time)
        """
        self.query_id = query_id
        self.query_type = query_type
        self.query_content = query_content
        self.parameters = parameters or {}
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()


class QueryResponse:
    """Response object for World Model queries."""

    def __init__(
        self,
        query_id: str,
        success: bool,
        result: str,
        confidence: float,
        explanation: str,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize query response.

        Args:
            query_id: ID of the original query
            success: Whether the query was successful
            result: Query result content
            confidence: Confidence score (0.0 to 1.0)
            explanation: Human-readable explanation
            timestamp: Response timestamp (defaults to current time)
            metadata: Additional response metadata
        """
        self.query_id = query_id
        self.success = success
        self.result = result
        self.confidence = confidence
        self.explanation = explanation
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.metadata = metadata or {}


class SimulationRequest:
    """Request object for World Model simulations."""

    def __init__(
        self,
        simulation_id: str,
        simulation_type: str,
        actions: Optional[list] = None,
        horizon_minutes: int = 60,
        parameters: Optional[Dict[str, str]] = None,
        timestamp: Optional[str] = None,
    ):
        """Initialize simulation request.

        Args:
            simulation_id: Unique identifier for this simulation
            simulation_type: Type of simulation ('forecast', 'what_if', 'scenario')
            actions: List of actions to simulate
            horizon_minutes: Simulation time horizon in minutes
            parameters: Additional simulation parameters
            timestamp: Request timestamp (defaults to current time)
        """
        self.simulation_id = simulation_id
        self.simulation_type = simulation_type
        self.actions = actions or []
        self.horizon_minutes = horizon_minutes
        self.parameters = parameters or {}
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()


class SimulationResponse:
    """Response object for World Model simulations."""

    def __init__(
        self,
        simulation_id: str,
        success: bool,
        future_states: list,
        confidence: float,
        uncertainty_lower: float,
        uncertainty_upper: float,
        explanation: str,
        impact_estimates: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize simulation response.

        Args:
            simulation_id: ID of the original simulation
            success: Whether the simulation was successful
            future_states: List of predicted future states
            confidence: Overall confidence score (0.0 to 1.0)
            uncertainty_lower: Lower bound of uncertainty interval
            uncertainty_upper: Upper bound of uncertainty interval
            explanation: Human-readable explanation
            impact_estimates: Cost/performance/reliability estimates
            timestamp: Response timestamp (defaults to current time)
            metadata: Additional response metadata
        """
        self.simulation_id = simulation_id
        self.success = success
        self.future_states = future_states
        self.confidence = confidence
        self.uncertainty_lower = uncertainty_lower
        self.uncertainty_upper = uncertainty_upper
        self.explanation = explanation
        self.impact_estimates = impact_estimates or {}
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert the simulation response to a dictionary."""
        return {
            "simulation_id": self.simulation_id,
            "success": self.success,
            "future_states": self.future_states,
            "confidence": self.confidence,
            "uncertainty_lower": self.uncertainty_lower,
            "uncertainty_upper": self.uncertainty_upper,
            "explanation": self.explanation,
            "impact_estimates": self.impact_estimates,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class DiagnosisRequest:
    """Request object for World Model diagnostics."""

    def __init__(
        self,
        diagnosis_id: str,
        anomaly_description: str,
        context: Optional[Dict[str, str]] = None,
        timestamp: Optional[str] = None,
    ):
        """Initialize diagnosis request.

        Args:
            diagnosis_id: Unique identifier for this diagnosis
            anomaly_description: Description of the anomaly to diagnose
            context: Additional context information
            timestamp: Request timestamp (defaults to current time)
        """
        self.diagnosis_id = diagnosis_id
        self.anomaly_description = anomaly_description
        self.context = context or {}
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()


class DiagnosisResponse:
    """Response object for World Model diagnostics."""

    def __init__(
        self,
        diagnosis_id: str,
        success: bool,
        hypotheses: list,
        causal_chain: str,
        confidence: float,
        explanation: str,
        supporting_evidence: Optional[list] = None,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize diagnosis response.

        Args:
            diagnosis_id: ID of the original diagnosis request
            success: Whether the diagnosis was successful
            hypotheses: List of ranked causal hypotheses
            causal_chain: Identified causal chain
            confidence: Overall confidence score (0.0 to 1.0)
            explanation: Human-readable explanation
            supporting_evidence: List of supporting evidence
            timestamp: Response timestamp (defaults to current time)
            metadata: Additional response metadata
        """
        self.diagnosis_id = diagnosis_id
        self.success = success
        self.hypotheses = hypotheses
        self.causal_chain = causal_chain
        self.confidence = confidence
        self.explanation = explanation
        self.supporting_evidence = supporting_evidence or []
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.metadata = metadata or {}


class WorldModel(abc.ABC):
    """
    Abstract base class for World Model implementations.

    This interface defines the contract that all World Model implementations
    must follow to integrate with the POLARIS Digital Twin. It supports both
    synchronous and asynchronous operations for maximum flexibility.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize the World Model.

        Args:
            config: Configuration dictionary for the implementation
            logger: Logger instance (created if not provided)
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._initialized = False
        self._health_status = {"status": "initializing", "last_check": None}

    @abc.abstractmethod
    async def initialize(self) -> None:
        """Initialize the World Model implementation.

        This method should perform any necessary setup, including
        loading models, establishing connections, or preparing data structures.

        Raises:
            WorldModelInitializationError: If initialization fails
        """
        pass

    @abc.abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the World Model implementation.

        This method should cleanly release resources and close connections.
        """
        pass

    @abc.abstractmethod
    async def update_state(self, event: KnowledgeEvent) -> None:
        """Update the world model state with new knowledge.

        This method processes incoming knowledge events (telemetry, execution
        results, anomalies) and integrates them into the world model.

        Args:
            event: Knowledge event containing system state updates

        Raises:
            WorldModelOperationError: If state update fails
        """
        pass

    @abc.abstractmethod
    async def calibrate(self, event: CalibrationEvent) -> None:
        """Calibrate the world model based on prediction accuracy feedback.

        This method adjusts the model based on how accurate previous
        predictions were compared to actual outcomes.

        Args:
            event: Calibration event with accuracy feedback

        Raises:
            WorldModelOperationError: If calibration fails
        """
        pass

    @abc.abstractmethod
    async def query_state(self, request: QueryRequest) -> QueryResponse:
        """Query the current or historical system state.

        This method handles different types of queries:
        - current_state: Current system state
        - historical: Historical state at specific time
        - natural_language: Natural language queries

        Args:
            request: Query request with type and parameters

        Returns:
            Query response with results and confidence

        Raises:
            WorldModelOperationError: If query fails
        """
        pass

    @abc.abstractmethod
    async def simulate(self, request: SimulationRequest) -> SimulationResponse:
        """Perform predictive simulation ("what-if" analysis).

        This method simulates future system states based on proposed actions
        or scenarios, providing predictions with uncertainty quantification.

        Args:
            request: Simulation request with actions and parameters

        Returns:
            Simulation response with predictions and uncertainty

        Raises:
            WorldModelOperationError: If simulation fails
        """
        pass

    @abc.abstractmethod
    async def diagnose(self, request: DiagnosisRequest) -> DiagnosisResponse:
        """Perform root cause analysis and diagnosis.

        This method analyzes anomalies and problems to identify likely
        root causes and provide ranked hypotheses with explanations.

        Args:
            request: Diagnosis request with anomaly description

        Returns:
            Diagnosis response with hypotheses and explanations

        Raises:
            WorldModelOperationError: If diagnosis fails
        """
        pass

    @abc.abstractmethod
    async def get_health_status(self) -> Dict[str, Any]:
        """Get the current health status of the World Model.

        Returns:
            Dictionary containing health status information
        """
        pass

    @abc.abstractmethod
    async def reload_model(self) -> bool:
        """Reload the World Model implementation.

        This method should reload the model configuration and reset
        the model state if necessary.

        Returns:
            True if reload was successful, False otherwise
        """
        pass

    # Synchronous wrapper methods for compatibility
    def update_state_sync(self, event: KnowledgeEvent) -> None:
        """Synchronous wrapper for update_state."""
        import asyncio

        return asyncio.run(self.update_state(event))

    def calibrate_sync(self, event: CalibrationEvent) -> None:
        """Synchronous wrapper for calibrate."""
        import asyncio

        return asyncio.run(self.calibrate(event))

    def query_state_sync(self, request: QueryRequest) -> QueryResponse:
        """Synchronous wrapper for query_state."""
        import asyncio

        return asyncio.run(self.query_state(request))

    def simulate_sync(self, request: SimulationRequest) -> SimulationResponse:
        """Synchronous wrapper for simulate."""
        import asyncio

        return asyncio.run(self.simulate(request))

    def diagnose_sync(self, request: DiagnosisRequest) -> DiagnosisResponse:
        """Synchronous wrapper for diagnose."""
        import asyncio

        return asyncio.run(self.diagnose(request))

    def get_health_status_sync(self) -> Dict[str, Any]:
        """Synchronous wrapper for get_health_status."""
        import asyncio

        return asyncio.run(self.get_health_status())

    def reload_model_sync(self) -> bool:
        """Synchronous wrapper for reload_model."""
        import asyncio

        return asyncio.run(self.reload_model())

    @property
    def is_initialized(self) -> bool:
        """Check if the World Model is initialized."""
        return self._initialized

    def _set_initialized(self, status: bool) -> None:
        """Set the initialization status (for use by implementations)."""
        self._initialized = status
        if status:
            self._health_status["status"] = "healthy"
        else:
            self._health_status["status"] = "not_initialized"
        self._health_status["last_check"] = datetime.now(timezone.utc).isoformat()


class WorldModelFactory:
    """
    Factory for creating World Model implementations.

    This factory provides a registry-based approach for managing different
    World Model implementations, enabling easy experimentation and swapping
    of different AI/ML approaches.
    """

    _registry: Dict[str, Type[WorldModel]] = {}

    @classmethod
    def register(cls, model_type: str, model_class: Type[WorldModel]) -> None:
        """Register a World Model implementation.

        Args:
            model_type: String identifier for the model type
            model_class: World Model implementation class

        Raises:
            ValueError: If model_class is not a WorldModel subclass
        """
        if not issubclass(model_class, WorldModel):
            raise ValueError(f"{model_class} must be a subclass of WorldModel")

        cls._registry[model_type.lower()] = model_class

    @classmethod
    def create_model(
        cls, model_type: str, config: Dict[str, Any], logger: Optional[logging.Logger] = None
    ) -> WorldModel:
        """Create a World Model instance.

        Args:
            model_type: Type of model to create
            config: Configuration for the model
            logger: Logger instance

        Returns:
            Instantiated World Model

        Raises:
            ValueError: If model_type is not registered
            WorldModelInitializationError: If model creation fails
        """
        model_type_lower = model_type.lower()

        if model_type_lower not in cls._registry:
            available_types = list(cls._registry.keys())
            raise ValueError(
                f"Unknown model type '{model_type}'. " f"Available types: {available_types}"
            )

        model_class = cls._registry[model_type_lower]

        try:
            return model_class(config=config, logger=logger)
        except Exception as e:
            raise WorldModelInitializationError(
                f"Failed to create {model_type} model: {str(e)}"
            ) from e

    @classmethod
    def get_registered_types(cls) -> list:
        """Get list of registered model types.

        Returns:
            List of registered model type strings
        """
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, model_type: str) -> bool:
        """Check if a model type is registered.

        Args:
            model_type: Model type to check

        Returns:
            True if registered, False otherwise
        """
        return model_type.lower() in cls._registry

    @classmethod
    def clear_registry(cls) -> None:
        """Clear the model registry (primarily for testing)."""
        cls._registry.clear()


# Interface validation utilities
class WorldModelValidator:
    """
    Utilities for validating World Model interface compliance.

    This class provides methods to test that World Model implementations
    correctly implement the required interface and behave as expected.
    """

    @staticmethod
    async def validate_interface_compliance(model: WorldModel) -> Dict[str, Any]:
        """Validate that a World Model implementation complies with the interface.

        Args:
            model: World Model instance to validate

        Returns:
            Dictionary containing validation results
        """
        results = {"compliant": True, "errors": [], "warnings": [], "test_results": {}}

        # Test 1: Check required methods exist
        required_methods = [
            "initialize",
            "shutdown",
            "update_state",
            "calibrate",
            "query_state",
            "simulate",
            "diagnose",
            "get_health_status",
            "reload_model",
        ]

        missing_methods = []
        for method_name in required_methods:
            if not hasattr(model, method_name):
                missing_methods.append(method_name)

        if missing_methods:
            results["compliant"] = False
            results["errors"].append(f"Missing required methods: {missing_methods}")

        results["test_results"]["required_methods"] = {
            "passed": len(missing_methods) == 0,
            "missing": missing_methods,
        }

        # Test 2: Check if methods are callable
        non_callable_methods = []
        for method_name in required_methods:
            if hasattr(model, method_name):
                method = getattr(model, method_name)
                if not callable(method):
                    non_callable_methods.append(method_name)

        if non_callable_methods:
            results["compliant"] = False
            results["errors"].append(f"Non-callable methods: {non_callable_methods}")

        results["test_results"]["callable_methods"] = {
            "passed": len(non_callable_methods) == 0,
            "non_callable": non_callable_methods,
        }

        # Test 3: Check initialization status property
        if not hasattr(model, "is_initialized"):
            results["compliant"] = False
            results["errors"].append("Missing 'is_initialized' property")

        results["test_results"]["initialization_property"] = {
            "passed": hasattr(model, "is_initialized")
        }

        # Test 4: Basic functionality test (if model is initialized)
        if hasattr(model, "is_initialized") and model.is_initialized:
            try:
                # Test health status
                health = await model.get_health_status()
                if not isinstance(health, dict):
                    results["warnings"].append("get_health_status should return a dictionary")

                results["test_results"]["health_status"] = {
                    "passed": isinstance(health, dict),
                    "result_type": type(health).__name__,
                }

                # Test query with mock request
                mock_query = QueryRequest(
                    query_id="test-query", query_type="current_state", query_content="test query"
                )

                query_response = await model.query_state(mock_query)
                if not isinstance(query_response, QueryResponse):
                    results["warnings"].append("query_state should return QueryResponse instance")

                results["test_results"]["query_functionality"] = {
                    "passed": isinstance(query_response, QueryResponse),
                    "result_type": type(query_response).__name__,
                }

            except Exception as e:
                results["warnings"].append(f"Basic functionality test failed: {str(e)}")
                results["test_results"]["basic_functionality"] = {"passed": False, "error": str(e)}
        else:
            results["test_results"]["basic_functionality"] = {
                "passed": False,
                "reason": "Model not initialized",
            }

        return results

    @staticmethod
    def validate_configuration(config: Dict[str, Any], model_type: str) -> Dict[str, Any]:
        """Validate configuration for a specific model type.

        Args:
            config: Configuration dictionary to validate
            model_type: Type of model the configuration is for

        Returns:
            Dictionary containing validation results
        """
        results = {"valid": True, "errors": [], "warnings": [], "recommendations": []}

        # Basic configuration validation
        if not isinstance(config, dict):
            results["valid"] = False
            results["errors"].append("Configuration must be a dictionary")
            return results

        # Model-specific validation
        if model_type.lower() == "mock":
            # Mock model has minimal requirements
            results["recommendations"].append("Mock model requires no specific configuration")

        elif model_type.lower() == "gemini" or model_type.lower() == "llm":
            # LLM-based models need API configuration
            required_fields = ["api_key_env", "model"]
            missing_fields = [field for field in required_fields if field not in config]

            if missing_fields:
                results["valid"] = False
                results["errors"].append(f"Missing required fields for LLM model: {missing_fields}")

            # Check for recommended fields
            recommended_fields = ["temperature", "max_tokens", "concurrent_requests"]
            missing_recommended = [field for field in recommended_fields if field not in config]

            if missing_recommended:
                results["recommendations"].append(
                    f"Consider adding recommended fields: {missing_recommended}"
                )

            # Validate specific values
            if "temperature" in config:
                temp = config["temperature"]
                if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                    results["warnings"].append("Temperature should be a number between 0 and 2")

            if "max_tokens" in config:
                max_tokens = config["max_tokens"]
                if not isinstance(max_tokens, int) or max_tokens <= 0:
                    results["warnings"].append("max_tokens should be a positive integer")

        else:
            results["warnings"].append(
                f"Unknown model type '{model_type}' - cannot validate specific requirements"
            )

        return results

    @staticmethod
    async def run_comprehensive_test(model: WorldModel) -> Dict[str, Any]:
        """Run a comprehensive test suite on a World Model implementation.

        Args:
            model: World Model instance to test

        Returns:
            Dictionary containing comprehensive test results
        """
        test_results = {
            "overall_passed": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_type": model.__class__.__name__,
            "tests": {},
        }

        # Interface compliance test
        compliance_results = await WorldModelValidator.validate_interface_compliance(model)
        test_results["tests"]["interface_compliance"] = compliance_results

        if not compliance_results["compliant"]:
            test_results["overall_passed"] = False

        # If model is not initialized, try to initialize it for testing
        if not model.is_initialized:
            try:
                await model.initialize()
                test_results["tests"]["initialization"] = {"passed": True}
            except Exception as e:
                test_results["tests"]["initialization"] = {"passed": False, "error": str(e)}
                test_results["overall_passed"] = False
                return test_results

        # Test all major operations
        operations_to_test = [
            (
                "update_state",
                lambda: model.update_state(
                    KnowledgeEvent(
                        event_id="test-event",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="test",
                        event_type="telemetry",
                        data={"cpu": 50.0, "memory": 60.0},
                    )
                ),
            ),
            (
                "calibrate",
                lambda: model.calibrate(
                    CalibrationEvent(
                        calibration_id="test-calibration",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        prediction_id="test-prediction",
                        actual_outcome={"cpu": 55.0},
                        predicted_outcome={"cpu": 50.0},
                        accuracy_metrics={"mae": 5.0},
                    )
                ),
            ),
            (
                "query_state",
                lambda: model.query_state(
                    QueryRequest(
                        query_id="test-query",
                        query_type="current_state",
                        query_content="What is the current CPU usage?",
                    )
                ),
            ),
            (
                "simulate",
                lambda: model.simulate(
                    SimulationRequest(
                        simulation_id="test-simulation",
                        simulation_type="forecast",
                        horizon_minutes=30,
                    )
                ),
            ),
            (
                "diagnose",
                lambda: model.diagnose(
                    DiagnosisRequest(
                        diagnosis_id="test-diagnosis", anomaly_description="High CPU usage detected"
                    )
                ),
            ),
            ("get_health_status", lambda: model.get_health_status()),
            ("reload_model", lambda: model.reload_model()),
        ]

        for operation_name, operation_func in operations_to_test:
            try:
                result = await operation_func()
                test_results["tests"][operation_name] = {
                    "passed": True,
                    "result_type": type(result).__name__ if result is not None else "None",
                }
            except Exception as e:
                test_results["tests"][operation_name] = {"passed": False, "error": str(e)}
                test_results["overall_passed"] = False

        return test_results


class ConfigurationValidator:
    """
    Utilities for validating World Model configurations.

    This class provides methods to validate configuration files and
    settings for different World Model implementations.
    """

    @staticmethod
    def validate_world_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a complete World Model configuration.

        Args:
            config: World Model configuration dictionary

        Returns:
            Dictionary containing validation results
        """
        results = {"valid": True, "errors": [], "warnings": [], "recommendations": []}

        # Check required top-level fields
        required_fields = ["implementation"]
        missing_fields = [field for field in required_fields if field not in config]

        if missing_fields:
            results["valid"] = False
            results["errors"].append(f"Missing required configuration fields: {missing_fields}")
            return results

        # Validate implementation type
        implementation = config["implementation"]
        if not isinstance(implementation, str):
            results["valid"] = False
            results["errors"].append("'implementation' must be a string")
            return results

        # Check if implementation is registered
        if not WorldModelFactory.is_registered(implementation):
            available_types = WorldModelFactory.get_registered_types()
            results["valid"] = False
            results["errors"].append(
                f"Unknown implementation '{implementation}'. " f"Available types: {available_types}"
            )

        # Validate implementation-specific configuration
        if "config" in config:
            impl_config = config["config"]
            config_validation = WorldModelValidator.validate_configuration(
                impl_config, implementation
            )

            if not config_validation["valid"]:
                results["valid"] = False
                results["errors"].extend(config_validation["errors"])

            results["warnings"].extend(config_validation["warnings"])
            results["recommendations"].extend(config_validation["recommendations"])
        else:
            results["warnings"].append("No implementation-specific configuration provided")

        # Check optional fields
        optional_fields = ["reload_on_failure", "health_check_interval_sec"]
        for field in optional_fields:
            if field in config:
                value = config[field]
                if field == "reload_on_failure" and not isinstance(value, bool):
                    results["warnings"].append(f"'{field}' should be a boolean")
                elif field == "health_check_interval_sec" and not isinstance(value, (int, float)):
                    results["warnings"].append(f"'{field}' should be a number")

        return results
