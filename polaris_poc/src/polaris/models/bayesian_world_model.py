"""
Bayesian/Kalman Filter World Model Implementation for POLARIS Digital Twin.

This module implements a deterministic World Model using Bayesian statistics
and Kalman filtering for metric prediction and system state estimation.
Provides formal mathematical foundations for system behavior modeling.
"""

import asyncio
import json
import logging
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict, deque
from dataclasses import dataclass, field
import uuid
from scipy import stats
from scipy.optimize import minimize
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

from .world_model import (
    WorldModel,
    WorldModelError,
    WorldModelInitializationError,
    WorldModelOperationError,
    QueryRequest,
    QueryResponse,
    SimulationRequest,
    SimulationResponse,
    DiagnosisRequest,
    DiagnosisResponse,
)
from .digital_twin_events import KnowledgeEvent, CalibrationEvent


@dataclass
class MetricState:
    """State representation for a single metric using Kalman filtering."""

    # Kalman filter instance
    kf: KalmanFilter

    # Historical data
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Bayesian parameters
    prior_mean: float = 0.0
    prior_variance: float = 1.0
    likelihood_variance: float = 0.1

    # Statistics
    mean: float = 0.0
    variance: float = 1.0
    trend: float = 0.0
    seasonality: Dict[str, float] = field(default_factory=dict)

    # Confidence intervals
    confidence_95_lower: float = 0.0
    confidence_95_upper: float = 0.0

    # Last update
    last_update: Optional[datetime] = None
    update_count: int = 0

    # Anomaly detection
    anomaly_threshold: float = 2.0  # Standard deviations
    recent_anomalies: deque = field(default_factory=lambda: deque(maxlen=100))


@dataclass
class SystemState:
    """Complete system state representation."""

    metrics: Dict[str, MetricState] = field(default_factory=dict)
    correlations: Dict[Tuple[str, str], float] = field(default_factory=dict)
    system_health_score: float = 1.0
    last_update: Optional[datetime] = None

    # System-level predictions
    predicted_states: List[Dict[str, Any]] = field(default_factory=list)
    prediction_horizon_minutes: int = 60

    # Causal relationships (learned)
    causal_graph: Dict[str, List[str]] = field(default_factory=dict)


class BayesianWorldModel(WorldModel):
    """
    Bayesian/Kalman Filter World Model implementation.

    This implementation uses formal statistical methods for:
    - Metric prediction using Kalman filters
    - Bayesian inference for system state estimation
    - Correlation analysis for dependency modeling
    - Anomaly detection using statistical thresholds
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize the Bayesian World Model.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(config, logger)

        # Configuration
        self.prediction_horizon_minutes = config.get("prediction_horizon_minutes", 60)
        self.max_history_points = config.get("max_history_points", 1000)
        self.correlation_threshold = config.get("correlation_threshold", 0.7)
        self.anomaly_threshold = config.get("anomaly_threshold", 2.0)
        self.update_interval_seconds = config.get("update_interval_seconds", 30)

        # Kalman filter parameters
        self.process_noise = config.get("process_noise", 0.01)
        self.measurement_noise = config.get("measurement_noise", 0.1)
        self.initial_uncertainty = config.get("initial_uncertainty", 1.0)

        # Bayesian parameters
        self.prior_confidence = config.get("prior_confidence", 0.5)
        self.learning_rate = config.get("learning_rate", 0.1)

        # System state
        self.system_state = SystemState()
        self._lock = asyncio.Lock()

        # Background tasks
        self._background_tasks = []
        self._running = False

        # Performance metrics
        self._prediction_accuracy = defaultdict(list)
        self._computation_times = deque(maxlen=100)

        self.logger.info(
            f"Initialized BayesianWorldModel with {len(config)} configuration parameters"
        )

    async def initialize(self) -> None:
        """Initialize the Bayesian World Model."""
        try:
            self.logger.info("Initializing Bayesian World Model...")

            # Start background processing tasks
            self._running = True

            # Task for periodic correlation analysis
            correlation_task = asyncio.create_task(self._periodic_correlation_analysis())
            self._background_tasks.append(correlation_task)

            # Task for system health monitoring
            health_task = asyncio.create_task(self._periodic_health_assessment())
            self._background_tasks.append(health_task)

            # Task for prediction accuracy tracking
            accuracy_task = asyncio.create_task(self._periodic_accuracy_assessment())
            self._background_tasks.append(accuracy_task)

            # Set initialization status
            self._set_initialized(True)
            self._health_status.update(
                {
                    "status": "healthy",
                    "model_type": "bayesian_kalman",
                    "last_check": datetime.now(timezone.utc).isoformat(),
                    "metrics_tracked": 0,
                    "background_tasks": len(self._background_tasks),
                }
            )

            self.logger.info("Bayesian World Model initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Bayesian World Model: {str(e)}")
            raise WorldModelInitializationError(f"Initialization failed: {str(e)}") from e

    async def shutdown(self) -> None:
        """Shutdown the Bayesian World Model."""
        try:
            self.logger.info("Shutting down Bayesian World Model...")

            # Stop background tasks
            self._running = False

            for task in self._background_tasks:
                if not task.done():
                    task.cancel()

            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)

            self._background_tasks.clear()

            # Set shutdown status
            self._set_initialized(False)
            self._health_status.update(
                {"status": "shutdown", "last_check": datetime.now(timezone.utc).isoformat()}
            )

            self.logger.info("Bayesian World Model shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")

    def _create_kalman_filter(self, initial_value: float = 0.0) -> KalmanFilter:
        """Create a Kalman filter for metric tracking.

        Args:
            initial_value: Initial value for the metric

        Returns:
            Configured Kalman filter
        """
        # Create 2D Kalman filter (value and velocity)
        kf = KalmanFilter(dim_x=2, dim_z=1)

        # State transition matrix (constant velocity model)
        dt = self.update_interval_seconds
        kf.F = np.array([[1.0, dt], [0.0, 1.0]])

        # Measurement function (we only observe position)
        kf.H = np.array([[1.0, 0.0]])

        # Process noise
        kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=self.process_noise)

        # Measurement noise
        kf.R = np.array([[self.measurement_noise]])

        # Initial state
        kf.x = np.array([[initial_value], [0.0]])  # [position, velocity]

        # Initial uncertainty
        kf.P = np.eye(2) * self.initial_uncertainty

        return kf

    async def update_state(self, event: KnowledgeEvent) -> None:
        """Update the world model state with new knowledge using Bayesian inference.

        Args:
            event: Knowledge event containing system state updates
        """
        try:
            async with self._lock:
                self.logger.debug(
                    f"Processing knowledge event: {event.event_type} from {event.source}"
                )

                if event.event_type == "telemetry":
                    await self._process_telemetry_bayesian(event)
                elif event.event_type == "execution_status":
                    await self._process_execution_bayesian(event)
                elif event.event_type == "anomaly":
                    await self._process_anomaly_bayesian(event)

                # Update system timestamp
                self.system_state.last_update = datetime.now(timezone.utc)

                self.logger.debug(f"Successfully processed knowledge event: {event.event_id}")

        except Exception as e:
            self.logger.error(f"Failed to update state with event {event.event_id}: {str(e)}")
            raise WorldModelOperationError(f"State update failed: {str(e)}") from e

    async def _process_telemetry_bayesian(self, event: KnowledgeEvent) -> None:
        """Process telemetry events using Bayesian updating."""
        try:
            telemetry_data = event.data

            # Handle both dict and object formats
            if isinstance(telemetry_data, dict):
                metric_name = telemetry_data.get("name")
                metric_value = telemetry_data.get("value")
            elif hasattr(telemetry_data, "name") and hasattr(telemetry_data, "value"):
                metric_name = telemetry_data.name
                metric_value = telemetry_data.value
            else:
                self.logger.warning(f"Invalid telemetry data format: {type(telemetry_data)}")
                return

            if metric_name is None or metric_value is None:
                self.logger.warning("Missing metric name or value in telemetry data")
                return

            try:
                metric_value = float(metric_value)
            except (ValueError, TypeError):
                self.logger.warning(f"Invalid metric value: {metric_value}")
                return

            # Parse timestamp
            try:
                timestamp_str = event.timestamp
                if timestamp_str.endswith("Z"):
                    timestamp_str = timestamp_str[:-1] + "+00:00"
                timestamp = datetime.fromisoformat(timestamp_str)
            except Exception:
                timestamp = datetime.now(timezone.utc)

            # Get or create metric state
            if metric_name not in self.system_state.metrics:
                self.system_state.metrics[metric_name] = MetricState(
                    kf=self._create_kalman_filter(metric_value),
                    prior_mean=metric_value,
                    anomaly_threshold=self.anomaly_threshold,
                )

            metric_state = self.system_state.metrics[metric_name]

            # Update Kalman filter
            metric_state.kf.predict()
            metric_state.kf.update(metric_value)

            # Store historical data
            metric_state.values.append(metric_value)
            metric_state.timestamps.append(timestamp)
            metric_state.last_update = timestamp
            metric_state.update_count += 1

            # Update Bayesian statistics
            await self._update_bayesian_statistics(metric_state, metric_value)

            # Anomaly detection
            await self._detect_anomalies(metric_state, metric_value, timestamp)

            self.logger.debug(f"Updated metric: {metric_name} = {metric_value}")

        except Exception as e:
            self.logger.warning(f"Failed to process telemetry event: {str(e)}")

    async def _update_bayesian_statistics(
        self, metric_state: MetricState, new_value: float
    ) -> None:
        """Update Bayesian statistics for a metric."""
        # Bayesian updating of mean and variance
        if len(metric_state.values) == 1:
            # First observation
            metric_state.mean = new_value
            metric_state.variance = metric_state.prior_variance
        else:
            # Bayesian update
            prior_precision = 1.0 / metric_state.variance
            likelihood_precision = 1.0 / metric_state.likelihood_variance

            # Update precision and mean
            posterior_precision = prior_precision + likelihood_precision
            posterior_variance = 1.0 / posterior_precision

            posterior_mean = (
                prior_precision * metric_state.mean + likelihood_precision * new_value
            ) / posterior_precision

            # Apply learning rate for stability
            metric_state.mean = (
                1 - self.learning_rate
            ) * metric_state.mean + self.learning_rate * posterior_mean
            metric_state.variance = (
                1 - self.learning_rate
            ) * metric_state.variance + self.learning_rate * posterior_variance

        # Update confidence intervals (95%)
        std_dev = np.sqrt(metric_state.variance)
        metric_state.confidence_95_lower = metric_state.mean - 1.96 * std_dev
        metric_state.confidence_95_upper = metric_state.mean + 1.96 * std_dev

        # Estimate trend using linear regression on recent values
        if len(metric_state.values) >= 5:
            recent_values = list(metric_state.values)[-10:]  # Last 10 values
            x = np.arange(len(recent_values))
            slope, _, _, _, _ = stats.linregress(x, recent_values)
            metric_state.trend = slope

    async def _detect_anomalies(
        self, metric_state: MetricState, value: float, timestamp: datetime
    ) -> None:
        """Detect anomalies using statistical methods."""
        if len(metric_state.values) < 5:  # Need minimum data for anomaly detection
            return

        # Z-score based anomaly detection
        z_score = abs(value - metric_state.mean) / np.sqrt(metric_state.variance)

        if z_score > metric_state.anomaly_threshold:
            anomaly = {
                "timestamp": timestamp.isoformat(),
                "value": value,
                "expected_mean": metric_state.mean,
                "z_score": z_score,
                "severity": "high" if z_score > 3.0 else "medium",
            }
            metric_state.recent_anomalies.append(anomaly)

            self.logger.warning(
                f"Anomaly detected in metric",
                extra={"value": value, "expected": metric_state.mean, "z_score": z_score},
            )

    async def calibrate(self, event: CalibrationEvent) -> None:
        """Calibrate the world model based on prediction accuracy feedback."""
        try:
            self.logger.debug(f"Processing calibration event: {event.calibration_id}")

            # Calculate accuracy score
            accuracy_score = event.calculate_accuracy_score()

            # Store accuracy for the relevant metrics
            for metric_name in event.accuracy_metrics.keys():
                self._prediction_accuracy[metric_name].append(accuracy_score)

                # Keep only recent accuracy scores
                if len(self._prediction_accuracy[metric_name]) > 100:
                    self._prediction_accuracy[metric_name] = self._prediction_accuracy[metric_name][
                        -100:
                    ]

            # Adjust model parameters based on accuracy
            await self._adjust_model_parameters(accuracy_score, event.accuracy_metrics)

            self.logger.info(f"Calibration complete: accuracy={accuracy_score:.2f}")

        except Exception as e:
            self.logger.error(f"Failed to calibrate with event {event.calibration_id}: {str(e)}")
            raise WorldModelOperationError(f"Calibration failed: {str(e)}") from e

    async def _adjust_model_parameters(
        self, accuracy_score: float, metrics: Dict[str, float]
    ) -> None:
        """Adjust model parameters based on calibration feedback."""
        # Adjust noise parameters based on accuracy
        if accuracy_score < 0.7:  # Poor accuracy
            self.measurement_noise *= 1.1  # Increase measurement noise
            self.process_noise *= 0.9  # Decrease process noise
        elif accuracy_score > 0.9:  # Good accuracy
            self.measurement_noise *= 0.95  # Decrease measurement noise
            self.process_noise *= 1.05  # Increase process noise

        # Update Kalman filters with new parameters
        for metric_state in self.system_state.metrics.values():
            metric_state.kf.R = np.array([[self.measurement_noise]])
            metric_state.kf.Q = Q_discrete_white_noise(
                dim=2, dt=self.update_interval_seconds, var=self.process_noise
            )

    async def query_state(self, request: QueryRequest) -> QueryResponse:
        """Query the current or historical system state using Bayesian inference."""
        try:
            start_time = datetime.now(timezone.utc)

            self.logger.debug(f"Processing query: {request.query_type} - {request.query_content}")

            result = None
            confidence = 0.8
            explanation = ""

            if request.query_type == "current_state":
                result, confidence, explanation = await self._query_current_state(request)
            elif request.query_type == "historical":
                result, confidence, explanation = await self._query_historical_state(request)
            elif request.query_type == "prediction":
                result, confidence, explanation = await self._query_prediction(request)
            elif request.query_type == "correlation":
                result, confidence, explanation = await self._query_correlations(request)
            else:
                result = f"Unknown query type: {request.query_type}"
                confidence = 0.0
                explanation = "Query type not supported by Bayesian World Model"

            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._computation_times.append(processing_time)

            response = QueryResponse(
                query_id=request.query_id,
                success=True,
                result=result,
                confidence=confidence,
                explanation=explanation,
                metadata={
                    "query_type": request.query_type,
                    "model": "bayesian_kalman",
                    "processing_time_sec": processing_time,
                    "metrics_analyzed": len(self.system_state.metrics),
                },
            )

            self.logger.debug(f"Query completed: {request.query_id} (confidence: {confidence:.2f})")
            return response

        except Exception as e:
            self.logger.error(f"Failed to process query {request.query_id}: {str(e)}")
            return QueryResponse(
                query_id=request.query_id,
                success=False,
                result="",
                confidence=0.0,
                explanation=f"Query failed: {str(e)}",
            )

    async def _query_current_state(
        self, request: QueryRequest
    ) -> Tuple[Dict[str, Any], float, str]:
        """Query current system state with Bayesian estimates."""
        current_state = {}
        total_confidence = 0.0
        metric_count = 0

        for metric_name, metric_state in self.system_state.metrics.items():
            # Get current Kalman filter estimate
            current_estimate = metric_state.kf.x[0, 0]  # Position estimate
            current_velocity = metric_state.kf.x[1, 0]  # Velocity estimate
            uncertainty = np.sqrt(metric_state.kf.P[0, 0])  # Position uncertainty

            # Calculate confidence based on uncertainty
            metric_confidence = max(0.1, 1.0 - (uncertainty / (abs(current_estimate) + 1.0)))

            current_state[metric_name] = {
                "value": float(current_estimate),
                "velocity": float(current_velocity),
                "uncertainty": float(uncertainty),
                "confidence": float(metric_confidence),
                "mean": metric_state.mean,
                "variance": metric_state.variance,
                "trend": metric_state.trend,
                "confidence_interval": [
                    metric_state.confidence_95_lower,
                    metric_state.confidence_95_upper,
                ],
                "last_update": (
                    metric_state.last_update.isoformat() if metric_state.last_update else None
                ),
                "anomalies_count": len(metric_state.recent_anomalies),
            }

            total_confidence += metric_confidence
            metric_count += 1

        overall_confidence = total_confidence / metric_count if metric_count > 0 else 0.0

        explanation = (
            f"Current state estimated using Kalman filtering for {metric_count} metrics. "
            f"System health score: {self.system_state.system_health_score:.2f}"
        )

        return current_state, overall_confidence, explanation

    async def simulate(self, request: SimulationRequest) -> SimulationResponse:
        """Perform predictive simulation using Kalman filter forecasting."""
        try:
            start_time = datetime.now(timezone.utc)

            self.logger.debug(
                f"Processing simulation: {request.simulation_type} - {request.simulation_id}"
            )

            # Generate future states using Kalman filter predictions
            future_states = []
            confidence_scores = []

            # Time steps for prediction
            time_steps = max(1, request.horizon_minutes // 5)  # 5-minute intervals

            for step in range(1, time_steps + 1):
                future_time = datetime.now(timezone.utc) + timedelta(minutes=step * 5)
                future_state = {}
                step_confidences = []

                for metric_name, metric_state in self.system_state.metrics.items():
                    # Create a copy of the Kalman filter for prediction
                    kf_copy = self._copy_kalman_filter(metric_state.kf)

                    # Predict forward 'step' time steps
                    for _ in range(step):
                        kf_copy.predict()

                    predicted_value = kf_copy.x[0, 0]
                    predicted_velocity = kf_copy.x[1, 0]
                    prediction_uncertainty = np.sqrt(kf_copy.P[0, 0])

                    # Calculate confidence (decreases with prediction horizon)
                    base_confidence = max(
                        0.1, 1.0 - (prediction_uncertainty / (abs(predicted_value) + 1.0))
                    )
                    time_decay = np.exp(-0.1 * step)  # Exponential decay with time
                    step_confidence = base_confidence * time_decay

                    future_state[metric_name] = {
                        "predicted_value": float(predicted_value),
                        "predicted_velocity": float(predicted_velocity),
                        "uncertainty": float(prediction_uncertainty),
                        "confidence": float(step_confidence),
                    }

                    step_confidences.append(step_confidence)

                # Apply action effects if specified
                if request.actions:
                    future_state = await self._apply_action_effects(
                        future_state, request.actions, step
                    )

                future_states.append(
                    {
                        "timestamp": future_time.isoformat(),
                        "time_offset_minutes": step * 5,
                        "state": future_state,
                    }
                )

                confidence_scores.append(np.mean(step_confidences) if step_confidences else 0.5)

            # Calculate overall confidence and uncertainty bounds
            overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.5
            uncertainty_lower = max(0.0, overall_confidence - 0.2)
            uncertainty_upper = min(1.0, overall_confidence + 0.1)

            # Generate impact estimates
            impact_estimates = await self._estimate_action_impacts(request.actions)

            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._computation_times.append(processing_time)

            response = SimulationResponse(
                simulation_id=request.simulation_id,
                success=True,
                future_states=future_states,
                confidence=overall_confidence,
                uncertainty_lower=uncertainty_lower,
                uncertainty_upper=uncertainty_upper,
                explanation=f"Kalman filter prediction over {request.horizon_minutes} minutes with {len(request.actions)} actions",
                impact_estimates=impact_estimates,
                metadata={
                    "simulation_type": request.simulation_type,
                    "horizon_minutes": request.horizon_minutes,
                    "time_steps": time_steps,
                    "actions_count": len(request.actions),
                    "model": "bayesian_kalman",
                    "processing_time_sec": processing_time,
                },
            )

            self.logger.debug(
                f"Simulation completed: {request.simulation_id} (confidence: {overall_confidence:.2f})"
            )
            return response

        except Exception as e:
            self.logger.error(f"Failed to process simulation {request.simulation_id}: {str(e)}")
            return SimulationResponse(
                simulation_id=request.simulation_id,
                success=False,
                future_states=[],
                confidence=0.0,
                uncertainty_lower=0.0,
                uncertainty_upper=0.0,
                explanation=f"Simulation failed: {str(e)}",
            )

    def _copy_kalman_filter(self, kf: KalmanFilter) -> KalmanFilter:
        """Create a deep copy of a Kalman filter for prediction."""
        kf_copy = KalmanFilter(dim_x=kf.x.shape[0], dim_z=kf.z.shape[0] if kf.z is not None else 1)
        kf_copy.x = kf.x.copy()
        kf_copy.P = kf.P.copy()
        kf_copy.F = kf.F.copy()
        kf_copy.H = kf.H.copy()
        kf_copy.Q = kf.Q.copy()
        kf_copy.R = kf.R.copy()
        return kf_copy

    async def _apply_action_effects(
        self, future_state: Dict[str, Any], actions: List[Any], time_step: int
    ) -> Dict[str, Any]:
        """Apply estimated effects of actions on future state."""
        # This is a simplified model - in practice, you'd have learned action effects
        for action in actions:
            action_type = None
            if isinstance(action, dict):
                action_type = action.get("action_type")
                params = action.get("parameters") or action.get("params") or {}
            else:
                action_type = getattr(action, "action_type", None)
                params = (
                    getattr(action, "parameters", None) or getattr(action, "params", None) or {}
                )

            if action_type == "ADD_SERVER":
                # Reduce CPU utilization, increase memory usage
                if "cpu_utilization" in future_state:
                    try:
                        current_cpu = float(future_state["cpu_utilization"]["predicted_value"])
                        reduction_factor = 0.8  # 20% reduction
                        future_state["cpu_utilization"]["predicted_value"] = (
                            current_cpu * reduction_factor
                        )
                    except (TypeError, ValueError):
                        self.logger.warning(
                            "Invalid type for cpu_utilization; skipping server removal effect"
                        )

            elif action_type == "SET_DIMMER":
                # Affect response time and throughput
                # params = (
                #     getattr(action, "parameters", None) or getattr(action, "params", None) or {}
                # )
                # dimmer_value = params.get("value", 1.0)
                # params = (
                #     getattr(action, "parameters", None) or getattr(action, "params", None) or {}
                # )
                dimmer_value = float(params.get("value", 1.0))

                # dimmer_value = action.parameters.get("value", 1.0) if hasattr(action, 'parameters') else 1.0

                if "response_time" in future_state:
                    current_rt = future_state["response_time"]["predicted_value"]
                    # Lower dimmer = better response time
                    future_state["response_time"]["predicted_value"] = current_rt * (
                        0.5 + 0.5 * dimmer_value
                    )

            elif action_type == "REMOVE_SERVER":
                # increase utilization
                if "cpu_utilization" in future_state:
                    try:
                        current_cpu = float(future_state["cpu_utilization"]["predicted_value"])
                        increase_factor = 1.15  # 15% increase
                        future_state["cpu_utilization"]["predicted_value"] = (
                            current_cpu * increase_factor
                        )
                    except (TypeError, ValueError):
                        self.logger.warning(
                            "Invalid type for cpu_utilization; skipping server removal effect"
                        )
                # Example: increase response time slightly per removed server
                if "response_time" in future_state:
                    try:
                        current_rt = float(future_state["response_time"]["predicted_value"])
                        increase_factor = 1.1  # 10% increase
                        future_state["response_time"]["predicted_value"] = (
                            current_rt * increase_factor
                        )
                    except (TypeError, ValueError):
                        self.logger.warning(
                            "Invalid type for response_time; skipping server removal effect"
                        )

        return future_state

    async def _estimate_action_impacts(self, actions: List[Any]) -> Dict[str, float]:
        """Estimate the impact of actions on system metrics."""
        impact_estimates = {
            "cost_impact": 0.0,
            "performance_impact": 0.0,
            "reliability_impact": 0.0,
        }

        for action in actions:
            action_type = None
            if isinstance(action, dict):
                action_type = action.get("action_type")
                params = action.get("parameters") or action.get("params") or {}
            else:
                action_type = getattr(action, "action_type", None)
                params = (
                    getattr(action, "parameters", None) or getattr(action, "params", None) or {}
                )

            if action_type == "ADD_SERVER":
                impact_estimates["cost_impact"] += 0.2  # Increase cost
                impact_estimates["performance_impact"] += 0.3  # Improve performance
                impact_estimates["reliability_impact"] += 0.1  # Improve reliability

            elif action_type == "REMOVE_SERVER":
                impact_estimates["cost_impact"] -= 0.2  # Reduce cost
                impact_estimates["performance_impact"] -= 0.3  # Reduce performance
                impact_estimates["reliability_impact"] -= 0.1  # Reduce reliability

            elif action_type == "SET_DIMMER":
                # dimmer_value = (
                #     action.parameters.get("value", 1.0)
                #     if hasattr(action, "parameters")
                #     else 1.0
                # )
                # params = (
                #     getattr(action, "parameters", None) or getattr(action, "params", None) or {}
                # )
                dimmer_value = float(params.get("value", 1.0))

                # Lower dimmer improves performance but may affect functionality
                impact_estimates["performance_impact"] += (1.0 - dimmer_value) * 0.2

        return impact_estimates

    async def diagnose(self, request: DiagnosisRequest) -> DiagnosisResponse:
        """Perform root cause analysis using statistical methods."""
        try:
            self.logger.debug(f"Processing diagnosis: {request.diagnosis_id}")

            # Analyze recent anomalies and correlations
            hypotheses = []
            supporting_evidence = []

            # Check for recent anomalies in metrics
            anomaly_metrics = []
            for metric_name, metric_state in self.system_state.metrics.items():
                if metric_state.recent_anomalies:
                    recent_anomaly = metric_state.recent_anomalies[-1]
                    anomaly_metrics.append(
                        {
                            "metric": metric_name,
                            "anomaly": recent_anomaly,
                            "z_score": recent_anomaly["z_score"],
                        }
                    )

            # Generate hypotheses based on anomalies
            if anomaly_metrics:
                # Sort by severity (z-score)
                anomaly_metrics.sort(key=lambda x: x["z_score"], reverse=True)

                primary_anomaly = anomaly_metrics[0]
                hypotheses.append(
                    f"Primary anomaly detected in {primary_anomaly['metric']} with z-score {primary_anomaly['z_score']:.2f}"
                )
                supporting_evidence.append(
                    f"Statistical deviation: {primary_anomaly['z_score']:.2f} standard deviations from expected value"
                )

                # Check for correlated anomalies
                for other_anomaly in anomaly_metrics[1:]:
                    correlation_key = tuple(
                        sorted([primary_anomaly["metric"], other_anomaly["metric"]])
                    )
                    if correlation_key in self.system_state.correlations:
                        correlation = self.system_state.correlations[correlation_key]
                        if abs(correlation) > self.correlation_threshold:
                            hypotheses.append(
                                f"Correlated anomaly in {other_anomaly['metric']} (correlation: {correlation:.2f})"
                            )
                            supporting_evidence.append(
                                f"Strong correlation ({correlation:.2f}) suggests cascading effect"
                            )

            # Check system health trends
            if self.system_state.system_health_score < 0.7:
                hypotheses.append("Overall system health degradation detected")
                supporting_evidence.append(
                    f"System health score: {self.system_state.system_health_score:.2f}"
                )

            # Generate causal chain
            causal_chain = (
                " -> ".join(hypotheses[:3]) if hypotheses else "No clear causal chain identified"
            )

            # Calculate confidence based on evidence strength
            confidence = min(0.9, len(supporting_evidence) * 0.2 + 0.3)

            explanation = (
                f"Statistical analysis of {len(self.system_state.metrics)} metrics identified "
                f"{len(anomaly_metrics)} anomalous metrics with {len(hypotheses)} potential root causes."
            )

            response = DiagnosisResponse(
                diagnosis_id=request.diagnosis_id,
                success=True,
                hypotheses=hypotheses,
                causal_chain=causal_chain,
                confidence=confidence,
                explanation=explanation,
                supporting_evidence=supporting_evidence,
                metadata={
                    "anomaly_count": len(anomaly_metrics),
                    "correlations_analyzed": len(self.system_state.correlations),
                    "system_health_score": self.system_state.system_health_score,
                    "model": "bayesian_kalman",
                },
            )

            self.logger.debug(
                f"Diagnosis completed: {request.diagnosis_id} (confidence: {confidence:.2f})"
            )
            return response

        except Exception as e:
            self.logger.error(f"Failed to process diagnosis {request.diagnosis_id}: {str(e)}")
            return DiagnosisResponse(
                diagnosis_id=request.diagnosis_id,
                success=False,
                hypotheses=[],
                causal_chain="",
                confidence=0.0,
                explanation=f"Diagnosis failed: {str(e)}",
            )

    async def _periodic_correlation_analysis(self) -> None:
        """Periodically analyze correlations between metrics."""
        while self._running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes

                if len(self.system_state.metrics) < 2:
                    continue

                # Calculate correlations between all metric pairs
                metric_names = list(self.system_state.metrics.keys())

                for i, metric1 in enumerate(metric_names):
                    for metric2 in metric_names[i + 1 :]:
                        await self._calculate_correlation(metric1, metric2)

                self.logger.debug(
                    f"Correlation analysis completed for {len(self.system_state.correlations)} pairs"
                )

            except Exception as e:
                self.logger.error(f"Error in correlation analysis: {str(e)}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _calculate_correlation(self, metric1: str, metric2: str) -> None:
        """Calculate correlation between two metrics."""
        try:
            state1 = self.system_state.metrics[metric1]
            state2 = self.system_state.metrics[metric2]

            if len(state1.values) < 10 or len(state2.values) < 10:
                return  # Need sufficient data

            # Get overlapping time periods
            values1 = list(state1.values)[-50:]  # Last 50 values
            values2 = list(state2.values)[-50:]

            min_length = min(len(values1), len(values2))
            if min_length < 10:
                return

            # Calculate Pearson correlation
            correlation, p_value = stats.pearsonr(values1[-min_length:], values2[-min_length:])

            # Store significant correlations
            if abs(correlation) > 0.3 and p_value < 0.05:  # Significant correlation
                correlation_key = tuple(sorted([metric1, metric2]))
                self.system_state.correlations[correlation_key] = correlation

                self.logger.debug(
                    f"Correlation found: {metric1} <-> {metric2} = {correlation:.3f} (p={p_value:.3f})"
                )

        except Exception as e:
            self.logger.warning(
                f"Failed to calculate correlation between {metric1} and {metric2}: {str(e)}"
            )

    async def _periodic_health_assessment(self) -> None:
        """Periodically assess overall system health."""
        while self._running:
            try:
                await asyncio.sleep(120)  # Run every 2 minutes

                if not self.system_state.metrics:
                    continue

                # Calculate system health score based on multiple factors
                health_factors = []

                # Factor 1: Anomaly rate
                total_anomalies = sum(
                    len(ms.recent_anomalies) for ms in self.system_state.metrics.values()
                )
                total_updates = sum(ms.update_count for ms in self.system_state.metrics.values())
                anomaly_rate = total_anomalies / max(1, total_updates)
                anomaly_health = max(0.0, 1.0 - anomaly_rate * 10)  # Scale anomaly rate
                health_factors.append(anomaly_health)

                # Factor 2: Prediction accuracy
                if self._prediction_accuracy:
                    avg_accuracy = np.mean(
                        [np.mean(scores) for scores in self._prediction_accuracy.values() if scores]
                    )
                    health_factors.append(avg_accuracy)

                # Factor 3: Data freshness
                current_time = datetime.now(timezone.utc)
                freshness_scores = []
                for metric_state in self.system_state.metrics.values():
                    if metric_state.last_update:
                        age_minutes = (current_time - metric_state.last_update).total_seconds() / 60
                        freshness = max(0.0, 1.0 - age_minutes / 60)  # Decay over 1 hour
                        freshness_scores.append(freshness)

                if freshness_scores:
                    health_factors.append(np.mean(freshness_scores))

                # Calculate overall health score
                self.system_state.system_health_score = (
                    np.mean(health_factors) if health_factors else 0.5
                )

                self.logger.debug(
                    f"System health score updated: {self.system_state.system_health_score:.3f}"
                )

            except Exception as e:
                self.logger.error(f"Error in health assessment: {str(e)}")
                await asyncio.sleep(60)

    async def _periodic_accuracy_assessment(self) -> None:
        """Periodically assess prediction accuracy."""
        while self._running:
            try:
                await asyncio.sleep(600)  # Run every 10 minutes

                # Clean up old accuracy scores
                for metric_name in list(self._prediction_accuracy.keys()):
                    scores = self._prediction_accuracy[metric_name]
                    if len(scores) > 100:
                        self._prediction_accuracy[metric_name] = scores[-100:]

                # Log accuracy statistics
                if self._prediction_accuracy:
                    overall_accuracy = np.mean(
                        [np.mean(scores) for scores in self._prediction_accuracy.values() if scores]
                    )
                    self.logger.info(f"Overall prediction accuracy: {overall_accuracy:.3f}")

            except Exception as e:
                self.logger.error(f"Error in accuracy assessment: {str(e)}")
                await asyncio.sleep(60)

    async def get_health_status(self) -> Dict[str, Any]:
        """Get the current health status of the World Model."""
        try:
            return {
                "status": "healthy" if self.is_initialized else "unhealthy",
                "model_type": "bayesian_kalman",
                "last_check": datetime.now(timezone.utc).isoformat(),
                "metrics": {
                    "metrics_tracked": len(self.system_state.metrics),
                    "correlations_discovered": len(self.system_state.correlations),
                    "system_health_score": self.system_state.system_health_score,
                    "background_tasks_running": len(
                        [t for t in self._background_tasks if not t.done()]
                    ),
                    "avg_computation_time_sec": (
                        np.mean(self._computation_times) if self._computation_times else 0.0
                    ),
                    "prediction_accuracy": {
                        metric: np.mean(scores) if scores else 0.0
                        for metric, scores in self._prediction_accuracy.items()
                    },
                },
                "configuration": {
                    "prediction_horizon_minutes": self.prediction_horizon_minutes,
                    "max_history_points": self.max_history_points,
                    "correlation_threshold": self.correlation_threshold,
                    "anomaly_threshold": self.anomaly_threshold,
                    "process_noise": self.process_noise,
                    "measurement_noise": self.measurement_noise,
                },
                "runtime_state": {
                    "running": self._running,
                    "last_update": (
                        self.system_state.last_update.isoformat()
                        if self.system_state.last_update
                        else None
                    ),
                    "total_anomalies": sum(
                        len(ms.recent_anomalies) for ms in self.system_state.metrics.values()
                    ),
                    "total_updates": sum(
                        ms.update_count for ms in self.system_state.metrics.values()
                    ),
                },
            }

        except Exception as e:
            return {
                "status": "error",
                "model_type": "bayesian_kalman",
                "error": str(e),
                "last_check": datetime.now(timezone.utc).isoformat(),
            }

    async def _process_execution_bayesian(self, event: KnowledgeEvent) -> None:
        """Process execution events with Bayesian updating."""
        # Implementation for execution event processing
        pass

    async def _process_anomaly_bayesian(self, event: KnowledgeEvent) -> None:
        """Process anomaly events with Bayesian analysis."""
        # Implementation for anomaly event processing
        pass

    async def _query_historical_state(self, request: QueryRequest) -> Tuple[Any, float, str]:
        """Query historical system state."""
        # Implementation for historical queries
        return "Historical query not yet implemented", 0.5, "Feature under development"

    async def _query_prediction(self, request: QueryRequest) -> Tuple[Any, float, str]:
        """Query predictions."""
        # Implementation for prediction queries
        return "Prediction query not yet implemented", 0.5, "Feature under development"

    async def _query_correlations(self, request: QueryRequest) -> Tuple[Any, float, str]:
        """Query metric correlations."""
        correlations = {}
        for (metric1, metric2), correlation in self.system_state.correlations.items():
            correlations[f"{metric1}__{metric2}"] = correlation

        return correlations, 0.9, f"Found {len(correlations)} significant correlations"

    def _set_initialized(self, initialized: bool) -> None:
        """Set the initialization status."""
        self._initialized = initialized

    @property
    def is_initialized(self) -> bool:
        """Check if the model is initialized."""
        return getattr(self, "_initialized", False)

    async def reload_model(self) -> bool:
        """Reload the Bayesian World Model implementation.

        Returns:
            True if reload was successful, False otherwise
        """
        try:
            self.logger.info("Reloading Bayesian World Model...")

            # Shutdown current instance
            await self.shutdown()

            # Clear existing state but preserve configuration
            self.system_state = SystemState()
            self._prediction_accuracy = defaultdict(list)
            self._computation_times = deque(maxlen=100)

            # Reinitialize
            await self.initialize()

            self.logger.info("Bayesian World Model reloaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to reload model: {str(e)}")
            return False


# Register the Bayesian World Model with the factory
def register_bayesian_world_model():
    """Register the Bayesian World Model implementation with the factory."""
    from .world_model import WorldModelFactory

    WorldModelFactory.register("bayesian", BayesianWorldModel)
    WorldModelFactory.register("kalman", BayesianWorldModel)  # Alias


# Auto-register when module is imported
register_bayesian_world_model()
