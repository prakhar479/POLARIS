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

# Enhanced anomaly detection imports
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available, some anomaly detection methods will be disabled")

from .world_model import (
    WorldModel, WorldModelError, WorldModelInitializationError, WorldModelOperationError,
    QueryRequest, QueryResponse, SimulationRequest, SimulationResponse,
    DiagnosisRequest, DiagnosisResponse
)
from .digital_twin_events import KnowledgeEvent, CalibrationEvent


@dataclass
class AnomalyDetectionResult:
    """Result from anomaly detection analysis."""
    is_anomaly: bool
    confidence: float
    method: str
    score: float
    threshold: float
    explanation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricState:
    """Enhanced state representation for a single metric with multiple anomaly detection methods."""
    
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
    
    # Enhanced anomaly detection
    anomaly_threshold: float = 4.0  # Standard deviations (increased from 3.0 to reduce false positives)
    recent_anomalies: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Multiple anomaly detection methods
    anomaly_detectors: Dict[str, Any] = field(default_factory=dict)
    anomaly_history: deque = field(default_factory=lambda: deque(maxlen=500))
    
    # IQR-based anomaly detection
    q1: float = 0.0
    q3: float = 0.0
    iqr: float = 0.0
    
    # Utility-based anomaly tracking (for SWITCH)
    utility_history: deque = field(default_factory=lambda: deque(maxlen=100))
    consecutive_utility_drops: int = 0
    
    # Model switching context (for SWITCH)
    current_model: Optional[str] = None
    last_model_switch: Optional[datetime] = None
    model_switch_stabilizing: bool = False
    
    # Seasonal decomposition
    seasonal_component: deque = field(default_factory=lambda: deque(maxlen=500))
    trend_component: deque = field(default_factory=lambda: deque(maxlen=500))
    residual_component: deque = field(default_factory=lambda: deque(maxlen=500))


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
    
    def _safe_float_conversion(self, value: Any, default: float = 0.0, param_name: str = "unknown") -> float:
        """Safely convert a parameter value to float, handling string conversions from gRPC.
        
        Args:
            value: The value to convert (could be string, int, float, etc.)
            default: Default value if conversion fails
            param_name: Name of parameter for logging
            
        Returns:
            Float value or default if conversion fails
        """
        try:
            self.logger.debug(f"_safe_float_conversion: Converting '{param_name}' value '{value}' (type: {type(value).__name__})")
            
            if isinstance(value, (int, float)):
                result = float(value)
                self.logger.debug(f"_safe_float_conversion: Numeric conversion successful: {result}")
                return result
            elif isinstance(value, str):
                result = float(value)
                self.logger.debug(f"_safe_float_conversion: String conversion successful: {result}")
                return result
            elif value is None:
                self.logger.debug(f"_safe_float_conversion: None value for '{param_name}', using default {default}")
                return default
            else:
                self.logger.warning(f"_safe_float_conversion: Unexpected type for parameter '{param_name}': {type(value).__name__} (value: {value}), using default {default}")
                return default
        except (ValueError, TypeError) as e:
            self.logger.error(f"_safe_float_conversion: Failed to convert parameter '{param_name}' value '{value}' (type: {type(value).__name__}) to float: {e}, using default {default}")
            return default
        except Exception as e:
            self.logger.error(f"_safe_float_conversion: Unexpected error converting parameter '{param_name}' value '{value}' (type: {type(value).__name__}): {e}, using default {default}")
            return default
    
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
        self.anomaly_threshold = config.get("anomaly_threshold", 4.0)
        self.update_interval_seconds = config.get("update_interval_seconds", 30)
        
        # Enhanced anomaly detection configuration
        self.anomaly_config = config.get("anomaly_detection", {})
        self.enable_multiple_methods = self.anomaly_config.get("enable_multiple_methods", True)
        self.switch_context = config.get("switch_context", {})
        self.metrics_config = config.get("metrics", {})
        
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
        
        self.logger.info(f"Initialized BayesianWorldModel with {len(config)} configuration parameters")
    
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
            self._health_status.update({
                "status": "healthy",
                "model_type": "bayesian_kalman",
                "last_check": datetime.now(timezone.utc).isoformat(),
                "metrics_tracked": 0,
                "background_tasks": len(self._background_tasks)
            })
            
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
            self._health_status.update({
                "status": "shutdown",
                "last_check": datetime.now(timezone.utc).isoformat()
            })
            
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
        kf.F = np.array([[1., dt],
                         [0., 1.]])
        
        # Measurement function (we only observe position)
        kf.H = np.array([[1., 0.]])
        
        # Process noise
        kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=self.process_noise)
        
        # Measurement noise
        kf.R = np.array([[self.measurement_noise]])
        
        # Initial state
        kf.x = np.array([[initial_value], [0.]])  # [position, velocity]
        
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
                self.logger.debug(f"Processing knowledge event: {event.event_type} from {event.source}")
                
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
                metric_name = telemetry_data.get('name')
                metric_value = telemetry_data.get('value')
            elif hasattr(telemetry_data, 'name') and hasattr(telemetry_data, 'value'):
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
                if timestamp_str.endswith('Z'):
                    timestamp_str = timestamp_str[:-1] + '+00:00'
                timestamp = datetime.fromisoformat(timestamp_str)
            except Exception:
                timestamp = datetime.now(timezone.utc)
            
            # Get or create metric state
            if metric_name not in self.system_state.metrics:
                self.system_state.metrics[metric_name] = MetricState(
                    kf=self._create_kalman_filter(metric_value),
                    prior_mean=metric_value,
                    anomaly_threshold=self.anomaly_threshold
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
    
    async def _update_bayesian_statistics(self, metric_state: MetricState, new_value: float) -> None:
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
                (prior_precision * metric_state.mean + likelihood_precision * new_value) /
                posterior_precision
            )
            
            # Apply learning rate for stability
            metric_state.mean = (1 - self.learning_rate) * metric_state.mean + self.learning_rate * posterior_mean
            metric_state.variance = (1 - self.learning_rate) * metric_state.variance + self.learning_rate * posterior_variance
        
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
    
    async def _detect_anomalies(self, metric_state: MetricState, value: float, timestamp: datetime) -> None:
        """Detect anomalies using statistical methods."""
        if len(metric_state.values) < 5:  # Need minimum data for anomaly detection
            self.logger.debug(f"Insufficient data for anomaly detection: {len(metric_state.values)} values")
            return
        
        # Z-score based anomaly detection
        std_dev = np.sqrt(metric_state.variance)
        if std_dev == 0:
            self.logger.debug("Standard deviation is 0, skipping anomaly detection")
            return
            
        z_score = abs(value - metric_state.mean) / std_dev
        
        self.logger.debug(
            f"Anomaly detection: value={value:.3f}, mean={metric_state.mean:.3f}, "
            f"std_dev={std_dev:.3f}, variance={metric_state.variance:.3f}, "
            f"z_score={z_score:.3f}, threshold={metric_state.anomaly_threshold}, "
            f"data_points={len(metric_state.values)}"
        )
        
        if z_score > metric_state.anomaly_threshold:
            anomaly = {
                "timestamp": timestamp.isoformat(),
                "value": value,
                "expected_mean": metric_state.mean,
                "std_dev": std_dev,
                "z_score": z_score,
                "severity": "high" if z_score > 5.0 else "medium",
                "data_points_used": len(metric_state.values)
            }
            metric_state.recent_anomalies.append(anomaly)
            
            self.logger.warning(
                f"ANOMALY DETECTED: z_score={z_score:.3f} > threshold={metric_state.anomaly_threshold}",
                extra={
                    "metric_value": value,
                    "expected_mean": metric_state.mean,
                    "std_deviation": std_dev,
                    "z_score": z_score,
                    "threshold": metric_state.anomaly_threshold,
                    "severity": anomaly["severity"],
                    "data_points": len(metric_state.values)
                }
            )
        else:
            self.logger.debug(f"Normal value: z_score={z_score:.3f} <= threshold={metric_state.anomaly_threshold}")
    
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
                    self._prediction_accuracy[metric_name] = self._prediction_accuracy[metric_name][-100:]
            
            # Adjust model parameters based on accuracy
            await self._adjust_model_parameters(accuracy_score, event.accuracy_metrics)
            
            self.logger.info(f"Calibration complete: accuracy={accuracy_score:.2f}")
            
        except Exception as e:
            self.logger.error(f"Failed to calibrate with event {event.calibration_id}: {str(e)}")
            raise WorldModelOperationError(f"Calibration failed: {str(e)}") from e
    
    async def _adjust_model_parameters(self, accuracy_score: float, metrics: Dict[str, float]) -> None:
        """Adjust model parameters based on calibration feedback."""
        # Adjust noise parameters based on accuracy
        if accuracy_score < 0.7:  # Poor accuracy
            self.measurement_noise *= 1.1  # Increase measurement noise
            self.process_noise *= 0.9      # Decrease process noise
        elif accuracy_score > 0.9:  # Good accuracy
            self.measurement_noise *= 0.95  # Decrease measurement noise
            self.process_noise *= 1.05      # Increase process noise
        
        # Update Kalman filters with new parameters
        for metric_state in self.system_state.metrics.values():
            metric_state.kf.R = np.array([[self.measurement_noise]])
            metric_state.kf.Q = Q_discrete_white_noise(
                dim=2, dt=self.update_interval_seconds, var=self.process_noise
            )
    
    async def query_state(self, request: QueryRequest) -> QueryResponse:
        """Enhanced query system with SWITCH-specific query types and advanced analytics."""
        try:
            start_time = datetime.now(timezone.utc)
            
            self.logger.debug(f"Processing enhanced query: {request.query_type} - {request.query_content}")
            
            result = None
            confidence = 0.8
            explanation = ""
            
            # Enhanced query routing with SWITCH-specific queries
            if request.query_type == "current_state":
                result, confidence, explanation = await self._query_current_state(request)
            elif request.query_type == "historical":
                result, confidence, explanation = await self._query_historical_state(request)
            elif request.query_type == "prediction":
                result, confidence, explanation = await self._query_prediction(request)
            elif request.query_type == "correlation":
                result, confidence, explanation = await self._query_correlations(request)
            elif request.query_type == "utility_optimization":
                result, confidence, explanation = await self._query_utility_optimization(request)
            elif request.query_type == "model_performance":
                result, confidence, explanation = await self._query_model_performance(request)
            elif request.query_type == "temporal_patterns":
                result, confidence, explanation = await self._query_temporal_patterns(request)
            elif request.query_type == "anomaly_analysis":
                result, confidence, explanation = await self._query_anomaly_analysis(request)
            elif request.query_type == "system_health":
                result, confidence, explanation = await self._query_system_health(request)
            elif request.query_type == "performance_forecast":
                result, confidence, explanation = await self._query_performance_forecast(request)
            else:
                result = f"Unknown query type: {request.query_type}"
                confidence = 0.0
                explanation = "Query type not supported by Enhanced Bayesian World Model"
            
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
                    "model": "enhanced_bayesian_kalman",
                    "processing_time_sec": processing_time,
                    "metrics_analyzed": len(self.system_state.metrics),
                    "switch_aware": True,
                    "enhanced_features": True
                }
            )
            
            self.logger.debug(f"Enhanced query completed: {request.query_id} (confidence: {confidence:.2f})")
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to process enhanced query {request.query_id}: {str(e)}")
            return QueryResponse(
                query_id=request.query_id,
                success=False,
                result="",
                confidence=0.0,
                explanation=f"Enhanced query failed: {str(e)}"
            )

    async def _query_utility_optimization(self, request: QueryRequest) -> Tuple[Dict, float, str]:
        """Query for SWITCH utility optimization opportunities."""
        
        current_metrics = {}
        current_utility = 0.5
        
        # Get current metrics
        if "image_processing_time" in self.system_state.metrics and "confidence" in self.system_state.metrics:
            rt_state = self.system_state.metrics["image_processing_time"]
            conf_state = self.system_state.metrics["confidence"]
            
            if rt_state.values and conf_state.values:
                current_rt = rt_state.values[-1]
                current_conf = conf_state.values[-1]
                current_utility = self._calculate_utility_score(current_rt, current_conf)
                
                current_metrics = {
                    "response_time": current_rt,
                    "confidence": current_conf,
                    "utility": current_utility
                }
        
        # Analyze each YOLO model's potential utility
        model_utilities = {}
        model_profiles = self.switch_context.get("yolo_models", {})
        
        for model_name, profile in model_profiles.items():
            expected_rt = profile.get("expected_response_time", 0.2)
            expected_conf = profile.get("expected_confidence", 0.75)
            expected_utility = self._calculate_utility_score(expected_rt, expected_conf)
            
            model_utilities[model_name] = {
                "expected_utility": expected_utility,
                "expected_response_time": expected_rt,
                "expected_confidence": expected_conf,
                "cpu_factor": profile.get("expected_cpu_factor", 1.0),
                "improvement": expected_utility - current_utility
            }
        
        # Find optimal model
        optimal_model = max(model_utilities, key=lambda m: model_utilities[m]["expected_utility"])
        
        # Get current CPU usage for feasibility check
        current_cpu = 50.0  # Default
        if "cpu_usage" in self.system_state.metrics and self.system_state.metrics["cpu_usage"].values:
            current_cpu = self.system_state.metrics["cpu_usage"].values[-1]
        
        # Check feasibility of optimal model
        optimal_cpu_factor = model_utilities[optimal_model]["cpu_factor"]
        feasible = (current_cpu * optimal_cpu_factor) < 85.0  # CPU safety threshold
        
        result = {
            "current_metrics": current_metrics,
            "current_model": self._get_current_switch_model(),
            "model_analysis": model_utilities,
            "optimal_model": optimal_model,
            "optimal_feasible": feasible,
            "potential_improvement": model_utilities[optimal_model]["improvement"],
            "current_cpu_usage": current_cpu,
            "recommendations": self._generate_optimization_recommendations(
                model_utilities, optimal_model, feasible, current_cpu
            )
        }
        
        confidence = 0.85 if len(model_profiles) >= 3 else 0.7
        explanation = f"Utility optimization analysis: {optimal_model} could improve utility by {result['potential_improvement']:.3f}"
        
        return result, confidence, explanation

    async def _query_model_performance(self, request: QueryRequest) -> Tuple[Dict, float, str]:
        """Query current model performance and trends."""
        
        model_name = request.parameters.get("model", self._get_current_switch_model())
        lookback_minutes = int(request.parameters.get("lookback_minutes", 60))
        
        performance_metrics = {}
        
        # Analyze performance for each metric
        for metric_name, metric_state in self.system_state.metrics.items():
            if len(metric_state.values) < 5:
                continue
            
            # Get recent values
            recent_count = min(len(metric_state.values), lookback_minutes // 2)  # Assuming 30s intervals
            recent_values = list(metric_state.values)[-recent_count:]
            
            performance_metrics[metric_name] = {
                "current_value": recent_values[-1] if recent_values else 0,
                "mean": np.mean(recent_values),
                "std": np.std(recent_values),
                "trend": metric_state.trend,
                "stability": 1.0 / (1.0 + np.std(recent_values)),  # Higher = more stable
                "recent_anomalies": len([a for a in metric_state.recent_anomalies 
                                       if (datetime.now(timezone.utc) - 
                                           datetime.fromisoformat(a["timestamp"].replace("Z", "+00:00"))).total_seconds() < lookback_minutes * 60])
            }
        
        # Calculate overall performance score
        overall_score = self._calculate_overall_performance_score(performance_metrics)
        
        result = {
            "model": model_name,
            "lookback_minutes": lookback_minutes,
            "performance_metrics": performance_metrics,
            "overall_performance_score": overall_score,
            "performance_grade": self._grade_performance(overall_score),
            "stability_assessment": self._assess_stability(performance_metrics),
            "recommendations": self._generate_performance_recommendations(performance_metrics, overall_score)
        }
        
        confidence = 0.8 if len(performance_metrics) >= 3 else 0.6
        explanation = f"Performance analysis for {model_name}: {result['performance_grade']} (score: {overall_score:.2f})"
        
        return result, confidence, explanation

    async def _query_temporal_patterns(self, request: QueryRequest) -> Tuple[Dict, float, str]:
        """Analyze temporal patterns in SWITCH metrics."""
        
        metric_name = request.parameters.get("metric", "utility")
        lookback_hours = int(request.parameters.get("lookback_hours", 24))
        
        if metric_name not in self.system_state.metrics:
            return {"error": f"Metric {metric_name} not found"}, 0.0, f"Metric {metric_name} not available"
        
        metric_state = self.system_state.metrics[metric_name]
        
        if len(metric_state.values) < 20:
            return {"error": "Insufficient data for pattern analysis"}, 0.0, "Need at least 20 data points"
        
        # Get data for analysis
        lookback_points = min(len(metric_state.values), lookback_hours * 2)  # Assuming 30min intervals
        values = list(metric_state.values)[-lookback_points:]
        timestamps = list(metric_state.timestamps)[-lookback_points:]
        
        # Perform pattern analysis
        patterns = {
            "trend_analysis": self._analyze_trend_pattern(values),
            "seasonality": self._detect_seasonality(values, timestamps),
            "volatility": self._calculate_volatility_pattern(values),
            "anomaly_clusters": self._detect_anomaly_clusters(metric_state, lookback_hours),
            "forecast": self._generate_pattern_forecast(values, 12)  # 6-hour forecast
        }
        
        result = {
            "metric": metric_name,
            "analysis_period_hours": lookback_hours,
            "data_points_analyzed": len(values),
            "patterns": patterns,
            "insights": self._generate_pattern_insights(patterns),
            "recommendations": self._generate_pattern_recommendations(patterns)
        }
        
        confidence = min(0.9, 0.5 + (len(values) / 100))  # Higher confidence with more data
        explanation = f"Temporal pattern analysis for {metric_name}: {patterns['trend_analysis']['direction']} trend with {patterns['volatility']['level']} volatility"
        
        return result, confidence, explanation

    async def _query_anomaly_analysis(self, request: QueryRequest) -> Tuple[Dict, float, str]:
        """Analyze recent anomalies and their patterns."""
        
        lookback_hours = int(request.parameters.get("lookback_hours", 12))
        severity_filter = request.parameters.get("severity", "all")  # all, high, medium, low
        
        all_anomalies = []
        
        # Collect anomalies from all metrics
        for metric_name, metric_state in self.system_state.metrics.items():
            for anomaly in metric_state.recent_anomalies:
                anomaly_time = datetime.fromisoformat(anomaly["timestamp"].replace("Z", "+00:00"))
                hours_ago = (datetime.now(timezone.utc) - anomaly_time).total_seconds() / 3600
                
                if hours_ago <= lookback_hours:
                    anomaly_data = anomaly.copy()
                    anomaly_data["metric"] = metric_name
                    anomaly_data["hours_ago"] = hours_ago
                    all_anomalies.append(anomaly_data)
        
        # Filter by severity if specified
        if severity_filter != "all":
            all_anomalies = [a for a in all_anomalies if a.get("severity") == severity_filter]
        
        # Analyze anomaly patterns
        analysis = {
            "total_anomalies": len(all_anomalies),
            "anomalies_by_metric": self._group_anomalies_by_metric(all_anomalies),
            "anomalies_by_severity": self._group_anomalies_by_severity(all_anomalies),
            "temporal_distribution": self._analyze_anomaly_temporal_distribution(all_anomalies),
            "correlation_analysis": self._analyze_anomaly_correlations(all_anomalies),
            "root_cause_candidates": self._identify_anomaly_root_causes(all_anomalies)
        }
        
        result = {
            "analysis_period_hours": lookback_hours,
            "severity_filter": severity_filter,
            "anomaly_analysis": analysis,
            "recent_anomalies": sorted(all_anomalies, key=lambda x: x["hours_ago"])[:10],  # 10 most recent
            "insights": self._generate_anomaly_insights(analysis),
            "recommendations": self._generate_anomaly_recommendations(analysis)
        }
        
        confidence = 0.8 if len(all_anomalies) >= 3 else 0.6
        explanation = f"Anomaly analysis: {len(all_anomalies)} anomalies in {lookback_hours}h, primary causes: {', '.join(analysis['root_cause_candidates'][:3])}"
        
        return result, confidence, explanation

    async def _query_system_health(self, request: QueryRequest) -> Tuple[Dict, float, str]:
        """Query overall system health and stability."""
        
        health_metrics = {}
        
        # Calculate health score for each metric
        for metric_name, metric_state in self.system_state.metrics.items():
            if len(metric_state.values) < 5:
                continue
            
            recent_values = list(metric_state.values)[-10:]  # Last 10 values
            
            # Health indicators
            stability = 1.0 / (1.0 + np.std(recent_values))
            trend_health = 1.0 - min(1.0, abs(metric_state.trend) / np.mean(recent_values)) if np.mean(recent_values) > 0 else 0.5
            anomaly_health = max(0.0, 1.0 - len(metric_state.recent_anomalies) / 10.0)
            
            health_score = (stability + trend_health + anomaly_health) / 3.0
            
            health_metrics[metric_name] = {
                "health_score": health_score,
                "stability": stability,
                "trend_health": trend_health,
                "anomaly_health": anomaly_health,
                "status": "healthy" if health_score > 0.7 else "degraded" if health_score > 0.4 else "critical"
            }
        
        # Overall system health
        overall_health = np.mean([m["health_score"] for m in health_metrics.values()]) if health_metrics else 0.5
        
        # System-level indicators
        system_indicators = {
            "overall_health_score": overall_health,
            "overall_status": "healthy" if overall_health > 0.7 else "degraded" if overall_health > 0.4 else "critical",
            "metrics_healthy": len([m for m in health_metrics.values() if m["status"] == "healthy"]),
            "metrics_degraded": len([m for m in health_metrics.values() if m["status"] == "degraded"]),
            "metrics_critical": len([m for m in health_metrics.values() if m["status"] == "critical"]),
            "total_metrics": len(health_metrics)
        }
        
        result = {
            "system_health": system_indicators,
            "metric_health": health_metrics,
            "health_trends": self._analyze_health_trends(),
            "alerts": self._generate_health_alerts(health_metrics, system_indicators),
            "recommendations": self._generate_health_recommendations(health_metrics, system_indicators)
        }
        
        confidence = 0.85 if len(health_metrics) >= 3 else 0.7
        explanation = f"System health: {system_indicators['overall_status']} (score: {overall_health:.2f}), {system_indicators['metrics_healthy']}/{system_indicators['total_metrics']} metrics healthy"
        
        return result, confidence, explanation

    async def _query_performance_forecast(self, request: QueryRequest) -> Tuple[Dict, float, str]:
        """Generate performance forecasts for SWITCH system."""
        
        forecast_horizon = int(request.parameters.get("horizon_minutes", 60))
        confidence_level = float(request.parameters.get("confidence_level", 0.95))
        
        forecasts = {}
        
        # Generate forecasts for key metrics
        key_metrics = ["image_processing_time", "confidence", "utility", "cpu_usage"]
        
        for metric_name in key_metrics:
            if metric_name not in self.system_state.metrics:
                continue
            
            metric_state = self.system_state.metrics[metric_name]
            if len(metric_state.values) < 10:
                continue
            
            # Generate forecast using Kalman filter
            forecast = self._generate_kalman_forecast(metric_state, forecast_horizon)
            forecasts[metric_name] = forecast
        
        # Generate scenario-based forecasts
        scenarios = {
            "current_trajectory": forecasts,
            "optimistic": self._generate_optimistic_forecast(forecasts),
            "pessimistic": self._generate_pessimistic_forecast(forecasts),
            "model_switch_impact": await self._forecast_model_switch_impact(forecasts)
        }
        
        result = {
            "forecast_horizon_minutes": forecast_horizon,
            "confidence_level": confidence_level,
            "scenarios": scenarios,
            "key_insights": self._generate_forecast_insights(scenarios),
            "recommendations": self._generate_forecast_recommendations(scenarios),
            "uncertainty_analysis": self._analyze_forecast_uncertainty(scenarios)
        }
        
        confidence = 0.75 if len(forecasts) >= 3 else 0.6
        explanation = f"Performance forecast for {forecast_horizon}min: {len(scenarios)} scenarios analyzed with {confidence_level*100}% confidence"
        
        return result, confidence, explanation
    
    async def _query_current_state(self, request: QueryRequest) -> Tuple[Dict[str, Any], float, str]:
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
                    metric_state.confidence_95_upper
                ],
                "last_update": metric_state.last_update.isoformat() if metric_state.last_update else None,
                "anomalies_count": len(metric_state.recent_anomalies)
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
        """Enhanced predictive simulation with SWITCH-aware dynamics and uncertainty quantification."""
        try:
            start_time = datetime.now(timezone.utc)
            
            self.logger.debug(f"Processing enhanced simulation: {request.simulation_id}")
            self.logger.debug(f"Simulation request details: type={request.simulation_type}, horizon={request.horizon_minutes}, actions={len(request.actions)}")
            
            # Log action details for debugging
            for i, action in enumerate(request.actions):
                self.logger.debug(f"Simulation action {i}: {action}")
            
            # Adaptive time stepping based on system volatility
            system_volatility = self._calculate_system_volatility()
            time_steps_array = self._calculate_adaptive_timesteps(request.horizon_minutes, system_volatility)
            
            self.logger.debug(f"System volatility: {system_volatility}, time steps: {len(time_steps_array)}")
            
            # Determine simulation method based on request type
            if request.simulation_type == "monte_carlo":
                return await self._run_monte_carlo_simulation(request, time_steps_array)
            else:
                return await self._run_deterministic_simulation(request, time_steps_array, system_volatility)
                
        except Exception as e:
            self.logger.error(f"Failed to process simulation {request.simulation_id}: {str(e)}", exc_info=True)
            return SimulationResponse(
                simulation_id=request.simulation_id,
                success=False,
                future_states=[],
                confidence=0.0,
                uncertainty_lower=0.0,
                uncertainty_upper=0.0,
                explanation=f"Simulation failed: {str(e)}",
                metadata={"error": str(e), "error_type": type(e).__name__}
            )

    def _calculate_system_volatility(self) -> float:
        """Calculate system volatility based on recent metric variations."""
        if not self.system_state.metrics:
            return 0.5  # Default medium volatility
        
        volatilities = []
        for metric_name, metric_state in self.system_state.metrics.items():
            if len(metric_state.values) >= 10:
                recent_values = list(metric_state.values)[-10:]
                volatility = np.std(recent_values) / (np.mean(recent_values) + 1e-6)
                volatilities.append(volatility)
        
        return np.mean(volatilities) if volatilities else 0.5

    def _calculate_adaptive_timesteps(self, horizon_minutes: int, system_volatility: float) -> np.ndarray:
        """Calculate adaptive time steps based on system dynamics."""
        if system_volatility > 0.8:  # High volatility - more frequent steps
            n_steps = min(horizon_minutes * 4, 120)  # Max 120 steps
        elif system_volatility > 0.5:  # Medium volatility
            n_steps = min(horizon_minutes * 2, 60)   # Max 60 steps
        else:  # Low volatility - fewer steps
            n_steps = min(horizon_minutes, 30)       # Max 30 steps
        
        return np.linspace(0, horizon_minutes, n_steps)

    async def _run_deterministic_simulation(self, request: SimulationRequest, time_steps_array: np.ndarray, system_volatility: float) -> SimulationResponse:
        """Run deterministic simulation with enhanced SWITCH dynamics."""
        start_time = datetime.now(timezone.utc)
        future_states = []
        confidence_scores = []
        
        # Create copies of Kalman filters for prediction
        prediction_filters = {}
        for metric_name, metric_state in self.system_state.metrics.items():
            prediction_filters[metric_name] = self._copy_kalman_filter(metric_state.kf)
        
        # Track model switching state
        current_switch_model = self._get_current_switch_model()
        model_switch_stabilization_remaining = 0
        
        # Generate future states
        for i, time_offset in enumerate(time_steps_array[1:]):  # Skip t=0
            future_time = start_time + timedelta(minutes=time_offset)
            future_state = {}
            step_confidences = []
            
            # Predict each metric with enhanced dynamics
            for metric_name, kf in prediction_filters.items():
                # Predict next state
                kf.predict()
                
                # Extract prediction and uncertainty
                predicted_value = float(kf.x[0, 0])
                predicted_velocity = float(kf.x[1, 0])
                uncertainty = float(np.sqrt(kf.P[0, 0]))
                
                # Enhanced confidence calculation considering SWITCH context
                base_confidence = max(0.1, 1.0 / (1.0 + uncertainty))
                
                # Reduce confidence during model switching
                if model_switch_stabilization_remaining > 0:
                    switch_penalty = 0.3 * (model_switch_stabilization_remaining / 10)
                    base_confidence *= (1.0 - switch_penalty)
                
                step_confidences.append(base_confidence)
                
                future_state[metric_name] = {
                    "predicted_value": predicted_value,
                    "predicted_velocity": predicted_velocity,
                    "uncertainty": uncertainty,
                    "confidence": base_confidence,
                    "trend": self._calculate_trend_indicator(metric_name, predicted_velocity)
                }
            
            # Apply enhanced action effects
            try:
                future_state, model_switch_info = await self._apply_enhanced_action_effects(
                    future_state, request.actions, i, current_switch_model
                )
                
                # Update model switching state
                if model_switch_info.get("model_switched"):
                    current_switch_model = model_switch_info["new_model"]
                    model_switch_stabilization_remaining = 10  # 10 steps to stabilize
                    self.logger.debug(f"Model switch detected: {model_switch_info['new_model']}")
                
                if model_switch_stabilization_remaining > 0:
                    model_switch_stabilization_remaining -= 1
                    
            except Exception as e:
                self.logger.error(f"Error applying enhanced action effects for step {i}: {str(e)}", exc_info=True)
            
            # Calculate SWITCH-specific metrics
            if "image_processing_time" in future_state and "confidence" in future_state:
                utility = self._calculate_utility_score(
                    future_state["image_processing_time"]["predicted_value"],
                    future_state["confidence"]["predicted_value"]
                )
                future_state["utility"] = {
                    "predicted_value": utility,
                    "confidence": 0.8,
                    "uncertainty": 0.1
                }
            
            # Convert complex future_state to simple metrics for gRPC compatibility
            simple_metrics = {}
            for metric_name, metric_data in future_state.items():
                if isinstance(metric_data, dict) and "predicted_value" in metric_data:
                    try:
                        simple_metrics[metric_name] = float(metric_data["predicted_value"])
                    except (ValueError, TypeError):
                        simple_metrics[metric_name] = 0.0
                elif isinstance(metric_data, (int, float)):
                    simple_metrics[metric_name] = float(metric_data)
            
            future_states.append({
                "timestamp": future_time.isoformat(),
                "time_offset_minutes": float(time_offset),
                "metrics": simple_metrics,  # Simple key-value pairs for gRPC
                "confidence": np.mean(step_confidences) if step_confidences else 0.5,
                "description": f"Predicted state at t+{time_offset:.1f}min with model {current_switch_model}",
                # Store complex data in metadata for internal use
                "metadata": {
                    "detailed_state": future_state,
                    "model_context": {
                        "current_model": current_switch_model,
                        "stabilizing": model_switch_stabilization_remaining > 0,
                        "stabilization_remaining": model_switch_stabilization_remaining
                    }
                }
            })
            
            confidence_scores.append(np.mean(step_confidences) if step_confidences else 0.5)
        
        # Enhanced confidence and uncertainty calculation
        overall_confidence = self._calculate_enhanced_confidence(confidence_scores, request.actions)
        uncertainty_bounds = self._calculate_uncertainty_bounds(confidence_scores, system_volatility)
        
        # Generate enhanced impact estimates
        impact_estimates = await self._estimate_enhanced_action_impacts(request.actions, future_states)
        
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        self._computation_times.append(processing_time)
        
        response = SimulationResponse(
            simulation_id=request.simulation_id,
            success=True,
            future_states=future_states,
            confidence=overall_confidence,
            uncertainty_lower=uncertainty_bounds["lower"],
            uncertainty_upper=uncertainty_bounds["upper"],
            explanation=f"Enhanced Kalman simulation over {request.horizon_minutes}min with {len(request.actions)} actions, {len(time_steps_array)} adaptive steps",
            impact_estimates=impact_estimates,
            metadata={
                "simulation_type": request.simulation_type,
                "horizon_minutes": request.horizon_minutes,
                "time_steps": len(time_steps_array),
                "actions_count": len(request.actions),
                "model": "enhanced_bayesian_kalman",
                "processing_time_sec": processing_time,
                "system_volatility": system_volatility,
                "adaptive_timesteps": True,
                "switch_aware": True
            }
        )
        
        self.logger.debug(f"Enhanced simulation completed: {request.simulation_id} (confidence: {overall_confidence:.2f})")
        return response

    async def _run_monte_carlo_simulation(self, request: SimulationRequest, time_steps_array: np.ndarray, n_scenarios: int = 50) -> SimulationResponse:
        """Run Monte Carlo simulation for uncertainty quantification."""
        start_time = datetime.now(timezone.utc)
        
        self.logger.debug(f"Running Monte Carlo simulation with {n_scenarios} scenarios")
        
        scenarios = []
        for scenario in range(n_scenarios):
            # Add noise to initial conditions
            noisy_request = self._add_noise_to_simulation_request(request, scenario)
            
            # Run single scenario (calculate volatility for each scenario)
            scenario_volatility = self._calculate_system_volatility()
            scenario_result = await self._run_deterministic_simulation(noisy_request, time_steps_array, scenario_volatility)
            if scenario_result.success:
                scenarios.append(scenario_result.future_states)
        
        if not scenarios:
            raise Exception("All Monte Carlo scenarios failed")
        
        # Aggregate scenarios
        aggregated_states = self._aggregate_monte_carlo_scenarios(scenarios, time_steps_array)
        
        # Calculate Monte Carlo statistics
        mc_confidence = len(scenarios) / n_scenarios  # Success rate
        mc_uncertainty = self._calculate_monte_carlo_uncertainty(scenarios)
        
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        return SimulationResponse(
            simulation_id=request.simulation_id,
            success=True,
            future_states=aggregated_states,
            confidence=mc_confidence,
            uncertainty_lower=max(0.0, mc_confidence - mc_uncertainty),
            uncertainty_upper=min(1.0, mc_confidence + mc_uncertainty),
            explanation=f"Monte Carlo simulation with {len(scenarios)}/{n_scenarios} successful scenarios",
            impact_estimates=await self._estimate_enhanced_action_impacts(request.actions, aggregated_states),
            metadata={
                "simulation_type": "monte_carlo",
                "scenarios_run": n_scenarios,
                "scenarios_successful": len(scenarios),
                "processing_time_sec": processing_time,
                "model": "monte_carlo_bayesian_kalman"
            }
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

    def _get_current_switch_model(self) -> str:
        """Get the current SWITCH model from system state."""
        # Try to infer from recent telemetry or use default
        for metric_state in self.system_state.metrics.values():
            if hasattr(metric_state, 'current_model') and metric_state.current_model:
                return metric_state.current_model
        return "yolov5s"  # Default model

    def _calculate_trend_indicator(self, metric_name: str, velocity: float) -> str:
        """Calculate trend indicator for a metric."""
        if abs(velocity) < 0.01:
            return "stable"
        elif velocity > 0:
            return "increasing"
        else:
            return "decreasing"

    async def _apply_enhanced_action_effects(self, future_state: Dict, actions: List[Any], time_step: int, current_model: str) -> Tuple[Dict, Dict]:
        """Apply enhanced action effects with SWITCH-specific dynamics."""
        model_switch_info = {"model_switched": False, "new_model": current_model}
        
        self.logger.debug(f"_apply_enhanced_action_effects: Processing {len(actions)} actions at time_step {time_step}")
        self.logger.debug(f"_apply_enhanced_action_effects: Current model: {current_model}")
        self.logger.debug(f"_apply_enhanced_action_effects: Future state keys: {list(future_state.keys())}")
        
        for i, action in enumerate(actions):
            try:
                self.logger.debug(f"_apply_enhanced_action_effects: Processing action {i}: {action} (type: {type(action)})")
                
                # Handle both dictionary and object formats
                action_type = None
                action_params = {}
                
                if isinstance(action, dict):
                    action_type = action.get('action_type')
                    action_params = action.get('params', {})
                    self.logger.debug(f"_apply_enhanced_action_effects: Dict action - type: {action_type}, params: {action_params} (params type: {type(action_params)})")
                elif hasattr(action, 'action_type'):
                    action_type = action.action_type
                    action_params = getattr(action, 'params', {})
                    self.logger.debug(f"_apply_enhanced_action_effects: Object action - type: {action_type}, params: {action_params} (params type: {type(action_params)})")
                else:
                    self.logger.warning(f"_apply_enhanced_action_effects: Unknown action format: {type(action)}")
                    continue
                
                # Log detailed parameter information
                if action_params:
                    for param_key, param_value in action_params.items():
                        self.logger.debug(f"_apply_enhanced_action_effects: Action param '{param_key}': '{param_value}' (type: {type(param_value).__name__})")
                
            except Exception as e:
                self.logger.error(f"_apply_enhanced_action_effects: Error processing action {i}: {e}", exc_info=True)
                continue
                
            if action_type and ("SWITCH_MODEL" in action_type or action_type == "SWITCH_MODEL"):
                # Get the target model with detailed logging
                raw_target_model = action_params.get("model") if action_params else None
                self.logger.debug(f"_apply_enhanced_action_effects: SWITCH_MODEL action - raw target: '{raw_target_model}' (type: {type(raw_target_model).__name__}), current: {current_model}")
                
                # Ensure target_model is a string
                target_model = str(raw_target_model) if raw_target_model is not None else None
                self.logger.debug(f"_apply_enhanced_action_effects: Converted target model: '{target_model}'")
                
                if target_model and target_model != current_model:
                    self.logger.info(f"_apply_enhanced_action_effects: Applying model switch from {current_model} to {target_model}")
                    try:
                        future_state = await self._apply_switch_model_effects(
                            future_state, current_model, target_model, time_step
                        )
                        model_switch_info = {"model_switched": True, "new_model": target_model}
                        self.logger.debug(f"_apply_enhanced_action_effects: Model switch successful")
                    except Exception as e:
                        self.logger.error(f"_apply_enhanced_action_effects: Model switch failed: {e}", exc_info=True)
                        raise
                else:
                    self.logger.debug(f"_apply_enhanced_action_effects: No model switch needed (target={target_model}, current={current_model})")
            
            elif action_type == "SET_DIMMER":
                self.logger.debug(f"_apply_enhanced_action_effects: SET_DIMMER action - params: {action_params}")
                # Apply dimmer effects with enhanced dynamics
                if action_params:
                    try:
                        # Get the dimmer value with detailed logging
                        raw_dimmer_value = action_params.get("value", "1.0")
                        self.logger.debug(f"_apply_enhanced_action_effects: Raw dimmer value: '{raw_dimmer_value}' (type: {type(raw_dimmer_value).__name__})")
                        
                        dimmer_value = self._safe_float_conversion(raw_dimmer_value, 1.0, "dimmer_value")
                        self.logger.debug(f"_apply_enhanced_action_effects: Converted dimmer value: {dimmer_value}")
                        
                        future_state = self._apply_dimmer_effects(future_state, dimmer_value, time_step)
                        self.logger.debug(f"_apply_enhanced_action_effects: Dimmer effects applied successfully")
                    except Exception as e:
                        self.logger.error(f"_apply_enhanced_action_effects: Dimmer effects failed: {e}", exc_info=True)
                        raise
                else:
                    self.logger.warning(f"_apply_enhanced_action_effects: SET_DIMMER action has no parameters")
            else:
                self.logger.warning(f"_apply_enhanced_action_effects: Unknown action type: {action_type}")
        
        self.logger.debug(f"_apply_enhanced_action_effects: Completed processing. Model switch info: {model_switch_info}")
        return future_state, model_switch_info

    async def _apply_switch_model_effects(self, future_state: Dict, current_model: str, target_model: str, time_step: int) -> Dict:
        """Apply SWITCH model switching effects with realistic transition dynamics."""
        
        self.logger.debug(f"_apply_switch_model_effects: Switching from {current_model} to {target_model} at step {time_step}")
        self.logger.debug(f"_apply_switch_model_effects: Future state structure: {list(future_state.keys())}")
        
        # Get model profiles from configuration
        model_profiles = self.switch_context.get("yolo_models", {})
        current_profile = model_profiles.get(current_model, {})
        target_profile = model_profiles.get(target_model, {})
        
        self.logger.debug(f"_apply_switch_model_effects: Current profile: {current_profile}")
        self.logger.debug(f"_apply_switch_model_effects: Target profile: {target_profile}")
        
        if not target_profile:
            self.logger.warning(f"_apply_switch_model_effects: Unknown target model: {target_model}")
            return future_state
        
        # Calculate switching progress (gradual transition over multiple steps)
        switch_duration = 3  # 3 time steps for full transition
        switch_progress = min(1.0, (time_step + 1) / switch_duration)
        self.logger.debug(f"_apply_switch_model_effects: Switch progress: {switch_progress:.2f}")
        
        # Apply gradual transition to response time
        if "image_processing_time" in future_state:
            try:
                current_rt = current_profile.get("expected_response_time", 0.2)
                target_rt = target_profile.get("expected_response_time", 0.2)
                
                self.logger.debug(f"_apply_switch_model_effects: RT transition - current: {current_rt}, target: {target_rt}")
                
                # Smooth transition with slight overshoot during switching
                overshoot_factor = 1.2 if switch_progress < 1.0 else 1.0
                new_rt = (current_rt * (1 - switch_progress) + target_rt * switch_progress) * overshoot_factor
                
                self.logger.debug(f"_apply_switch_model_effects: Calculated new RT: {new_rt}")
                self.logger.debug(f"_apply_switch_model_effects: Future state RT structure: {future_state['image_processing_time']}")
                
                future_state["image_processing_time"]["predicted_value"] = new_rt
                if "uncertainty" in future_state["image_processing_time"]:
                    future_state["image_processing_time"]["uncertainty"] *= 1.5  # Higher uncertainty during switch
                    
                self.logger.debug(f"_apply_switch_model_effects: RT updated successfully")
            except Exception as e:
                self.logger.error(f"_apply_switch_model_effects: Error updating response time: {e}", exc_info=True)
                raise
        
        # Apply gradual transition to confidence
        if "confidence" in future_state:
            try:
                current_conf = current_profile.get("expected_confidence", 0.75)
                target_conf = target_profile.get("expected_confidence", 0.75)
                
                self.logger.debug(f"_apply_switch_model_effects: Conf transition - current: {current_conf}, target: {target_conf}")
                
                # Confidence may dip slightly during switching
                dip_factor = 0.95 if switch_progress < 1.0 else 1.0
                new_conf = (current_conf * (1 - switch_progress) + target_conf * switch_progress) * dip_factor
                
                self.logger.debug(f"_apply_switch_model_effects: Calculated new confidence: {new_conf}")
                
                future_state["confidence"]["predicted_value"] = new_conf
                if "uncertainty" in future_state["confidence"]:
                    future_state["confidence"]["uncertainty"] *= 1.3
                    
                self.logger.debug(f"_apply_switch_model_effects: Confidence updated successfully")
            except Exception as e:
                self.logger.error(f"_apply_switch_model_effects: Error updating confidence: {e}", exc_info=True)
                raise
        
        # Apply CPU usage changes
        if "cpu_usage" in future_state:
            try:
                current_cpu_factor = current_profile.get("expected_cpu_factor", 1.0)
                target_cpu_factor = target_profile.get("expected_cpu_factor", 1.0)
                
                self.logger.debug(f"_apply_switch_model_effects: CPU factors - current: {current_cpu_factor}, target: {target_cpu_factor}")
                
                # CPU usage changes more immediately
                current_cpu_value = future_state["cpu_usage"]["predicted_value"]
                base_cpu = current_cpu_value / current_cpu_factor
                new_cpu = base_cpu * (current_cpu_factor * (1 - switch_progress) + target_cpu_factor * switch_progress)
                
                self.logger.debug(f"_apply_switch_model_effects: CPU calculation - base: {base_cpu}, new: {new_cpu}")
                
                future_state["cpu_usage"]["predicted_value"] = min(95.0, new_cpu)  # Cap at 95%
                
                self.logger.debug(f"_apply_switch_model_effects: CPU updated successfully")
            except Exception as e:
                self.logger.error(f"_apply_switch_model_effects: Error updating CPU usage: {e}", exc_info=True)
                raise
        
        self.logger.debug(f"_apply_switch_model_effects: Model switch effects applied successfully: {current_model} -> {target_model} (progress: {switch_progress:.2f})")
        return future_state

    def _apply_dimmer_effects(self, future_state: Dict, dimmer_value: float, time_step: int) -> Dict:
        """Apply dimmer effects with enhanced dynamics."""
        
        # Dimmer affects response time and throughput
        if "image_processing_time" in future_state:
            current_rt = future_state["image_processing_time"]["predicted_value"]
            # Lower dimmer = better response time (less processing)
            dimmer_factor = 0.7 + 0.3 * dimmer_value  # Range: 0.7 to 1.0
            future_state["image_processing_time"]["predicted_value"] = current_rt * dimmer_factor
        
        # Dimmer may slightly affect confidence (less processing = potentially lower accuracy)
        if "confidence" in future_state and dimmer_value < 0.8:
            confidence_penalty = (0.8 - dimmer_value) * 0.1  # Max 2% penalty
            current_conf = future_state["confidence"]["predicted_value"]
            future_state["confidence"]["predicted_value"] = max(0.5, current_conf - confidence_penalty)
        
        return future_state

    def _calculate_utility_score(self, response_time: float, confidence: float) -> float:
        """Calculate SWITCH utility score using the defined utility function."""
        
        # SWITCH utility function parameters (from config)
        Rmin, Rmax = 0.1, 1.0  # Response time bounds
        Cmin, Cmax = 0.5, 1.0  # Confidence bounds
        wd, we = 0.5, 0.5      # Weights
        pdv, pev = 5.0, 5.0    # Penalty scales
        
        # Normalize response time (lower is better)
        if response_time <= Rmin:
            rt_score = 1.0
        elif response_time >= Rmax:
            rt_score = 0.0 - pdv * (response_time - Rmax)  # Penalty for exceeding max
        else:
            rt_score = 1.0 - (response_time - Rmin) / (Rmax - Rmin)
        
        # Normalize confidence (higher is better)
        if confidence >= Cmax:
            conf_score = 1.0
        elif confidence <= Cmin:
            conf_score = 0.0 - pev * (Cmin - confidence)  # Penalty for below min
        else:
            conf_score = (confidence - Cmin) / (Cmax - Cmin)
        
        # Calculate weighted utility
        utility = we * conf_score + wd * rt_score
        
        return utility

    def _calculate_enhanced_confidence(self, confidence_scores: List[float], actions: List[Any]) -> float:
        """Calculate enhanced confidence considering action complexity."""
        if not confidence_scores:
            return 0.5
        
        base_confidence = np.mean(confidence_scores)
        
        # Reduce confidence for complex actions
        action_complexity_penalty = 0.0
        for action in actions:
            # Handle both dictionary and object formats
            if isinstance(action, dict):
                action_type = action.get('action_type')
            elif hasattr(action, 'action_type'):
                action_type = action.action_type
            else:
                continue
                
            if action_type and ("SWITCH_MODEL" in action_type or action_type == "SWITCH_MODEL"):
                action_complexity_penalty += 0.1  # Model switching adds uncertainty
            elif action_type == "SET_DIMMER":
                action_complexity_penalty += 0.05  # Dimmer changes add less uncertainty
        
        return max(0.1, base_confidence - action_complexity_penalty)

    def _calculate_uncertainty_bounds(self, confidence_scores: List[float], system_volatility: float) -> Dict[str, float]:
        """Calculate uncertainty bounds based on confidence and system volatility."""
        if not confidence_scores:
            return {"lower": 0.3, "upper": 0.7}
        
        mean_confidence = np.mean(confidence_scores)
        confidence_std = np.std(confidence_scores) if len(confidence_scores) > 1 else 0.1
        
        # Adjust bounds based on system volatility
        volatility_factor = 1.0 + system_volatility
        uncertainty_range = confidence_std * volatility_factor
        
        return {
            "lower": max(0.0, mean_confidence - uncertainty_range),
            "upper": min(1.0, mean_confidence + uncertainty_range)
        }

    async def _estimate_enhanced_action_impacts(self, actions: List[Any], future_states: List[Dict]) -> Dict[str, float]:
        """Estimate enhanced action impacts using future state predictions."""
        
        impact_estimates = {
            "cost_impact": 0.0,
            "performance_impact": 0.0,
            "reliability_impact": 0.0,
            "utility_impact": 0.0,
            "cost_currency": "USD",
            "impact_description": ""
        }
        
        if not future_states:
            return impact_estimates
        
        # Calculate utility impact from future states
        # Fix: Access utility from metrics instead of non-existent 'state' key
        initial_utility = future_states[0].get("metrics", {}).get("utility", 0.5)
        final_utility = future_states[-1].get("metrics", {}).get("utility", 0.5)
        impact_estimates["utility_impact"] = final_utility - initial_utility
        
        # Estimate impacts for each action
        for action in actions:
            # Handle both dictionary and object formats
            if isinstance(action, dict):
                action_type = action.get('action_type')
                action_params = action.get('params', {})
            elif hasattr(action, 'action_type'):
                action_type = action.action_type
                action_params = getattr(action, 'params', {})
            else:
                continue
                
            if action_type and ("SWITCH_MODEL" in action_type or action_type == "SWITCH_MODEL"):
                target_model = action_params.get("model") if action_params else None
                if target_model:
                    model_impacts = self._estimate_model_switch_impacts(target_model)
                    for key in ["cost_impact", "performance_impact", "reliability_impact"]:
                        impact_estimates[key] += model_impacts.get(key, 0.0)
            
            elif action_type == "SET_DIMMER":
                dimmer_value = self._safe_float_conversion(
                    action_params.get("value", "1.0") if action_params else "1.0", 
                    1.0, "dimmer_value"
                )
                dimmer_impacts = self._estimate_dimmer_impacts(dimmer_value)
                for key in ["cost_impact", "performance_impact", "reliability_impact"]:
                    impact_estimates[key] += dimmer_impacts.get(key, 0.0)
        
        # Generate impact description
        impact_estimates["impact_description"] = self._generate_impact_description(impact_estimates)
        
        return impact_estimates

    def _estimate_model_switch_impacts(self, target_model: str) -> Dict[str, float]:
        """Estimate impacts of switching to a specific model."""
        
        model_profiles = self.switch_context.get("yolo_models", {})
        target_profile = model_profiles.get(target_model, {})
        
        if not target_profile:
            return {"cost_impact": 0.0, "performance_impact": 0.0, "reliability_impact": 0.0}
        
        # Estimate based on model characteristics
        cpu_factor = target_profile.get("expected_cpu_factor", 1.0)
        expected_rt = target_profile.get("expected_response_time", 0.2)
        expected_conf = target_profile.get("expected_confidence", 0.75)
        
        return {
            "cost_impact": (cpu_factor - 1.0) * 0.1,  # Higher CPU = higher cost
            "performance_impact": (0.5 - expected_rt) * 0.5,  # Lower RT = better performance
            "reliability_impact": (expected_conf - 0.75) * 0.3  # Higher confidence = better reliability
        }

    def _estimate_dimmer_impacts(self, dimmer_value: float) -> Dict[str, float]:
        """Estimate impacts of dimmer changes."""
        
        # Lower dimmer = less processing = lower cost but potentially lower accuracy
        dimmer_effect = 1.0 - dimmer_value
        
        return {
            "cost_impact": -dimmer_effect * 0.05,  # Lower processing = lower cost
            "performance_impact": dimmer_effect * 0.1,  # Lower processing = better response time
            "reliability_impact": -dimmer_effect * 0.05  # Lower processing = potentially lower accuracy
        }

    def _generate_impact_description(self, impacts: Dict[str, float]) -> str:
        """Generate human-readable impact description."""
        
        descriptions = []
        
        if abs(impacts["utility_impact"]) > 0.05:
            direction = "improve" if impacts["utility_impact"] > 0 else "decrease"
            descriptions.append(f"Expected to {direction} utility by {abs(impacts['utility_impact']):.3f}")
        
        if abs(impacts["performance_impact"]) > 0.05:
            direction = "improve" if impacts["performance_impact"] > 0 else "degrade"
            descriptions.append(f"{direction} performance")
        
        if abs(impacts["cost_impact"]) > 0.02:
            direction = "increase" if impacts["cost_impact"] > 0 else "reduce"
            descriptions.append(f"{direction} resource costs")
        
        return "; ".join(descriptions) if descriptions else "Minimal expected impact"

    def _add_noise_to_simulation_request(self, request: SimulationRequest, scenario: int) -> SimulationRequest:
        """Add noise to simulation request for Monte Carlo scenarios."""
        # For now, return the original request
        # In a full implementation, you would add noise to initial conditions
        return request

    def _aggregate_monte_carlo_scenarios(self, scenarios: List[List[Dict]], time_steps_array: np.ndarray) -> List[Dict]:
        """Aggregate Monte Carlo scenarios into statistical summaries."""
        
        if not scenarios:
            return []
        
        aggregated_states = []
        n_timesteps = len(scenarios[0])
        
        for t in range(n_timesteps):
            # Collect all scenario states for this timestep
            timestep_states = [scenario[t] for scenario in scenarios if t < len(scenario)]
            
            if not timestep_states:
                continue
            
            # Aggregate metrics across scenarios
            aggregated_state = {
                "timestamp": timestep_states[0]["timestamp"],
                "time_offset_minutes": timestep_states[0]["time_offset_minutes"],
                "metrics": {},
                "confidence": 0.8,  # Monte Carlo confidence
                "description": f"Monte Carlo aggregated state from {len(timestep_states)} scenarios",
                "metadata": {
                    "monte_carlo_stats": {
                        "scenarios_count": len(timestep_states),
                        "confidence_interval": 0.95
                    }
                }
            }
            
            # Aggregate each metric
            all_metrics = set()
            for state in timestep_states:
                # Fix: Access metrics from the correct structure
                if "metrics" in state:
                    all_metrics.update(state["metrics"].keys())
            
            for metric in all_metrics:
                metric_values = []
                for state in timestep_states:
                    # Fix: Access metrics from the correct structure
                    if "metrics" in state and metric in state["metrics"]:
                        metric_values.append(state["metrics"][metric])
                
                if metric_values:
                    # Fix: Use metrics structure instead of state
                    if "metrics" not in aggregated_state:
                        aggregated_state["metrics"] = {}
                    aggregated_state["metrics"][metric] = np.mean(metric_values)
                    
                    # Store detailed stats in metadata
                    if "metadata" not in aggregated_state:
                        aggregated_state["metadata"] = {}
                    if "detailed_stats" not in aggregated_state["metadata"]:
                        aggregated_state["metadata"]["detailed_stats"] = {}
                    
                    aggregated_state["metadata"]["detailed_stats"][metric] = {
                        "mean": np.mean(metric_values),
                        "std": np.std(metric_values),
                        "percentile_5": np.percentile(metric_values, 5),
                        "percentile_95": np.percentile(metric_values, 95),
                        "min_value": np.min(metric_values),
                        "max_value": np.max(metric_values)
                    }
            
            aggregated_states.append(aggregated_state)
        
        return aggregated_states

    def _calculate_monte_carlo_uncertainty(self, scenarios: List[List[Dict]]) -> float:
        """Calculate uncertainty from Monte Carlo scenarios."""
        if len(scenarios) < 2:
            return 0.3  # Default uncertainty
        
        # Calculate coefficient of variation across scenarios
        scenario_utilities = []
        for scenario in scenarios:
            if scenario and "metrics" in scenario[-1] and "utility" in scenario[-1]["metrics"]:
                scenario_utilities.append(scenario[-1]["metrics"]["utility"])
        
        if len(scenario_utilities) < 2:
            return 0.3
        
        cv = np.std(scenario_utilities) / (np.mean(scenario_utilities) + 1e-6)
        return min(0.5, cv)  # Cap uncertainty at 0.5
    
    def _safe_float_conversion(self, value: Any, default: float, context: str) -> float:
        """Safely convert a value to float with detailed logging."""
        try:
            if isinstance(value, (int, float)):
                result = float(value)
                self.logger.debug(f"_safe_float_conversion ({context}): {value} ({type(value).__name__}) -> {result}")
                return result
            elif isinstance(value, str):
                result = float(value)
                self.logger.debug(f"_safe_float_conversion ({context}): '{value}' (str) -> {result}")
                return result
            else:
                self.logger.warning(f"_safe_float_conversion ({context}): Cannot convert {value} ({type(value).__name__}) to float, using default {default}")
                return default
        except (ValueError, TypeError) as e:
            self.logger.warning(f"_safe_float_conversion ({context}): Error converting {value} to float: {e}, using default {default}")
            return default
    
    async def _apply_action_effects(self, future_state: Dict[str, Any], actions: List[Any], time_step: int) -> Dict[str, Any]:
        """Apply estimated effects of actions on future state."""
        # This is a simplified model - in practice, you'd have learned action effects
        self.logger.debug(f"Applying action effects for time step {time_step}, actions count: {len(actions)}")

        for i, action in enumerate(actions):
            self.logger.debug(f"Processing action {i+1}/{len(actions)}: {action}")

            # Handle both dictionary and object formats
            if isinstance(action, dict):
                action_type = action.get('action_type')
                action_params = action.get('params', {})
            elif hasattr(action, 'action_type'):
                action_type = action.action_type
                action_params = getattr(action, 'params', {})
            else:
                self.logger.warning(f"Action {i+1} does not have action_type: {type(action)} - {action}")
                continue
                
            self.logger.debug(f"Action type: {action_type}")

            if action_type == "ADD_SERVER":
                # Reduce CPU utilization, increase memory usage
                if "cpu_utilization" in future_state:
                    current_cpu = future_state["cpu_utilization"]["predicted_value"]
                    reduction_factor = 0.8  # 20% reduction
                    self.logger.debug(f"ADD_SERVER: Reducing CPU from {current_cpu} by factor {reduction_factor}")
                    future_state["cpu_utilization"]["predicted_value"] = current_cpu * reduction_factor

            elif action_type == "SET_DIMMER":
                # Affect response time and throughput
                self.logger.debug(f"SET_DIMMER action parameters: {action_params}")

                # Check if action has parameters and handle different formats
                if action_params:
                    raw_value = action_params.get("value", "1.0")
                    self.logger.debug(f"SET_DIMMER: Raw parameter value: {raw_value} (type: {type(raw_value).__name__})")
                    
                    # Use safe conversion helper
                    dimmer_value = self._safe_float_conversion(raw_value, 1.0, "dimmer_value")
                    self.logger.debug(f"SET_DIMMER: Converted dimmer_value: {dimmer_value}")
                else:
                    dimmer_value = 1.0
                    self.logger.debug(f"SET_DIMMER: Using default dimmer_value: {dimmer_value}")

                if "response_time" in future_state:
                    current_rt = future_state["response_time"]["predicted_value"]
                    new_rt = current_rt * (0.5 + 0.5 * dimmer_value)
                    self.logger.debug(f"SET_DIMMER: Changing response time from {current_rt} to {new_rt}")
                    # Lower dimmer = better response time
                    future_state["response_time"]["predicted_value"] = new_rt
                else:
                    self.logger.warning(f"SET_DIMMER: response_time not found in future_state")

            else:
                self.logger.warning(f"Unknown action type: {action_type}")

        return future_state
    
    async def _estimate_action_impacts(self, actions: List[Any]) -> Dict[str, float]:
        """Estimate the impact of actions on system metrics."""
        impact_estimates = {
            "cost_impact": 0.0,
            "performance_impact": 0.0,
            "reliability_impact": 0.0
        }
        
        for action in actions:
            # Handle both dictionary and object formats
            if isinstance(action, dict):
                action_type = action.get('action_type')
                action_params = action.get('params', {})
            elif hasattr(action, 'action_type'):
                action_type = action.action_type
                action_params = getattr(action, 'params', {})
            else:
                continue
                
            if action_type == "ADD_SERVER":
                impact_estimates["cost_impact"] += 0.2  # Increase cost
                impact_estimates["performance_impact"] += 0.3  # Improve performance
                impact_estimates["reliability_impact"] += 0.1  # Improve reliability
                
            elif action_type == "REMOVE_SERVER":
                impact_estimates["cost_impact"] -= 0.2  # Reduce cost
                impact_estimates["performance_impact"] -= 0.3  # Reduce performance
                impact_estimates["reliability_impact"] -= 0.1  # Reduce reliability
                
            elif action_type == "SET_DIMMER":
                if action_params:
                    raw_value = action_params.get("value", "1.0")
                    dimmer_value = self._safe_float_conversion(raw_value, 1.0, "dimmer_value")
                else:
                    dimmer_value = 1.0
                # Lower dimmer improves performance but may affect functionality
                impact_estimates["performance_impact"] += (1.0 - dimmer_value) * 0.2
        
        return impact_estimates
    
    async def diagnose(self, request: DiagnosisRequest) -> DiagnosisResponse:
        """Enhanced root cause analysis with SWITCH-specific diagnostics and multi-layered analysis."""
        try:
            start_time = datetime.now(timezone.utc)
            self.logger.debug(f"Processing enhanced diagnosis: {request.diagnosis_id}")

            # Multi-layered diagnostic analysis
            diagnostic_results = {
                "anomaly_analysis": self._perform_anomaly_diagnosis(request),
                "causal_analysis": self._perform_causal_diagnosis(request),
                "switch_specific_analysis": self._perform_switch_specific_diagnosis(request),
                "temporal_analysis": self._perform_temporal_diagnosis(request),
                "correlation_analysis": self._perform_correlation_diagnosis(request)
            }

            # Synthesize hypotheses from all analyses
            all_hypotheses = []
            all_evidence = []
            confidence_scores = []

            for analysis_type, results in diagnostic_results.items():
                if results.get("hypotheses"):
                    for hypothesis in results["hypotheses"]:
                        if isinstance(hypothesis, dict):
                            hypothesis["source_analysis"] = analysis_type
                            all_hypotheses.append(hypothesis)
                        else:
                            # Convert string hypothesis to dict format
                            all_hypotheses.append({
                                "hypothesis": str(hypothesis),
                                "probability": 0.7,
                                "reasoning": f"Identified through {analysis_type}",
                                "evidence": [],
                                "rank": len(all_hypotheses) + 1,
                                "source_analysis": analysis_type
                            })
                
                if results.get("evidence"):
                    all_evidence.extend(results["evidence"])
                
                if results.get("confidence"):
                    confidence_scores.append(results["confidence"])

            # Rank hypotheses by likelihood and evidence strength
            ranked_hypotheses = self._rank_diagnostic_hypotheses(all_hypotheses, all_evidence)

            # Build comprehensive causal chain
            causal_chain = self._build_enhanced_causal_chain(ranked_hypotheses, diagnostic_results)

            # Calculate overall diagnostic confidence
            overall_confidence = self._calculate_diagnostic_confidence(confidence_scores, len(all_evidence))

            # Generate actionable recommendations
            recommendations = self._generate_diagnostic_recommendations(ranked_hypotheses, diagnostic_results)

            # Create detailed explanation
            explanation = self._generate_diagnostic_explanation(
                ranked_hypotheses, diagnostic_results, len(self.system_state.metrics)
            )

            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            # Convert hypotheses to expected format for response
            formatted_hypotheses = []
            for h in ranked_hypotheses[:10]:  # Top 10 hypotheses
                if isinstance(h, dict):
                    formatted_hypotheses.append({
                        "hypothesis": h.get("hypothesis", "Unknown hypothesis"),
                        "probability": h.get("probability", 0.5),
                        "reasoning": h.get("reasoning", ""),
                        "evidence": h.get("evidence", []),
                        "rank": h.get("rank", len(formatted_hypotheses) + 1)
                    })
                else:
                    formatted_hypotheses.append({
                        "hypothesis": str(h),
                        "probability": 0.5,
                        "reasoning": "Basic analysis",
                        "evidence": [],
                        "rank": len(formatted_hypotheses) + 1
                    })

            response = DiagnosisResponse(
                diagnosis_id=request.diagnosis_id,
                success=True,
                hypotheses=formatted_hypotheses,
                causal_chain=causal_chain,
                confidence=overall_confidence,
                explanation=explanation,
                supporting_evidence=all_evidence[:20],  # Top 20 pieces of evidence
                metadata={
                    "diagnosis_method": "enhanced_multi_layer",
                    "analyses_performed": list(diagnostic_results.keys()),
                    "total_hypotheses": len(all_hypotheses),
                    "total_evidence": len(all_evidence),
                    "processing_time_sec": processing_time,
                    "switch_aware": True,
                    "causal_inference": True,
                    "recommendations": recommendations,
                    "diagnostic_summary": {
                        "primary_issue": ranked_hypotheses[0].get("hypothesis", "No clear issue identified") if ranked_hypotheses else "No clear issue identified",
                        "confidence_level": "high" if overall_confidence > 0.8 else "medium" if overall_confidence > 0.6 else "low",
                        "urgency": self._assess_diagnostic_urgency(ranked_hypotheses),
                        "affected_systems": self._identify_affected_systems(ranked_hypotheses)
                    }
                }
            )

            self.logger.debug(f"Enhanced diagnosis completed: {request.diagnosis_id} (confidence: {overall_confidence:.2f})")
            return response

        except Exception as e:
            self.logger.error(f"Failed to process enhanced diagnosis {request.diagnosis_id}: {str(e)}", exc_info=True)
            return DiagnosisResponse(
                diagnosis_id=request.diagnosis_id,
                success=False,
                hypotheses=[],
                causal_chain="",
                confidence=0.0,
                explanation=f"Enhanced diagnosis failed: {str(e)}",
                metadata={"error": str(e), "diagnosis_method": "enhanced_multi_layer"}
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
                    for metric2 in metric_names[i+1:]:
                        await self._calculate_correlation(metric1, metric2)
                
                self.logger.debug(f"Correlation analysis completed for {len(self.system_state.correlations)} pairs")
                
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
                
                self.logger.debug(f"Correlation found: {metric1} <-> {metric2} = {correlation:.3f} (p={p_value:.3f})")
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate correlation between {metric1} and {metric2}: {str(e)}")
    
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
                total_anomalies = sum(len(ms.recent_anomalies) for ms in self.system_state.metrics.values())
                total_updates = sum(ms.update_count for ms in self.system_state.metrics.values())
                anomaly_rate = total_anomalies / max(1, total_updates)
                anomaly_health = max(0.0, 1.0 - anomaly_rate * 10)  # Scale anomaly rate
                health_factors.append(anomaly_health)
                
                # Factor 2: Prediction accuracy
                if self._prediction_accuracy:
                    avg_accuracy = np.mean([
                        np.mean(scores) for scores in self._prediction_accuracy.values() if scores
                    ])
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
                self.system_state.system_health_score = np.mean(health_factors) if health_factors else 0.5
                
                self.logger.debug(f"System health score updated: {self.system_state.system_health_score:.3f}")
                
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
                    overall_accuracy = np.mean([
                        np.mean(scores) for scores in self._prediction_accuracy.values() if scores
                    ])
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
                    "background_tasks_running": len([t for t in self._background_tasks if not t.done()]),
                    "avg_computation_time_sec": np.mean(self._computation_times) if self._computation_times else 0.0,
                    "prediction_accuracy": {
                        metric: np.mean(scores) if scores else 0.0
                        for metric, scores in self._prediction_accuracy.items()
                    }
                },
                "configuration": {
                    "prediction_horizon_minutes": self.prediction_horizon_minutes,
                    "max_history_points": self.max_history_points,
                    "correlation_threshold": self.correlation_threshold,
                    "anomaly_threshold": self.anomaly_threshold,
                    "process_noise": self.process_noise,
                    "measurement_noise": self.measurement_noise
                },
                "runtime_state": {
                    "running": self._running,
                    "last_update": self.system_state.last_update.isoformat() if self.system_state.last_update else None,
                    "total_anomalies": sum(len(ms.recent_anomalies) for ms in self.system_state.metrics.values()),
                    "total_updates": sum(ms.update_count for ms in self.system_state.metrics.values())
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "model_type": "bayesian_kalman",
                "error": str(e),
                "last_check": datetime.now(timezone.utc).isoformat()
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

    def _generate_optimization_recommendations(self, model_utilities: Dict, optimal_model: str, feasible: bool, current_cpu: float) -> List[str]:
        """Generate utility optimization recommendations."""
        recommendations = []
        
        if not feasible:
            recommendations.append(f"Optimal model {optimal_model} not feasible due to CPU constraints ({current_cpu:.1f}%)")
            # Find best feasible model
            feasible_models = {k: v for k, v in model_utilities.items() 
                             if current_cpu * v["cpu_factor"] < 85.0}
            if feasible_models:
                best_feasible = max(feasible_models, key=lambda m: feasible_models[m]["expected_utility"])
                recommendations.append(f"Consider {best_feasible} as best feasible alternative")
        else:
            improvement = model_utilities[optimal_model]["improvement"]
            if improvement > 0.05:
                recommendations.append(f"Switch to {optimal_model} for {improvement:.3f} utility improvement")
            elif improvement > 0.01:
                recommendations.append(f"Consider {optimal_model} for modest {improvement:.3f} utility gain")
            else:
                recommendations.append("Current model appears optimal")
        
        return recommendations

    def _calculate_overall_performance_score(self, performance_metrics: Dict) -> float:
        """Calculate overall performance score from individual metrics."""
        if not performance_metrics:
            return 0.5
        
        scores = []
        for metric_name, metrics in performance_metrics.items():
            # Weight different aspects
            stability_score = metrics["stability"]
            anomaly_score = max(0.0, 1.0 - metrics["recent_anomalies"] / 5.0)  # Penalize anomalies
            
            # Metric-specific scoring
            if metric_name == "utility":
                value_score = max(0.0, min(1.0, metrics["current_value"]))  # Utility should be 0-1
            elif metric_name == "image_processing_time":
                value_score = max(0.0, min(1.0, (1.0 - metrics["current_value"])))  # Lower RT is better
            elif metric_name == "confidence":
                value_score = max(0.0, min(1.0, metrics["current_value"]))  # Higher confidence is better
            else:
                value_score = 0.5  # Neutral for unknown metrics
            
            metric_score = (stability_score + anomaly_score + value_score) / 3.0
            scores.append(metric_score)
        
        return np.mean(scores)

    def _grade_performance(self, score: float) -> str:
        """Convert performance score to letter grade."""
        if score >= 0.9:
            return "A (Excellent)"
        elif score >= 0.8:
            return "B (Good)"
        elif score >= 0.7:
            return "C (Fair)"
        elif score >= 0.6:
            return "D (Poor)"
        else:
            return "F (Critical)"

    def _assess_stability(self, performance_metrics: Dict) -> Dict:
        """Assess system stability."""
        if not performance_metrics:
            return {"level": "unknown", "score": 0.5}
        
        stability_scores = [m["stability"] for m in performance_metrics.values()]
        avg_stability = np.mean(stability_scores)
        
        if avg_stability >= 0.8:
            level = "high"
        elif avg_stability >= 0.6:
            level = "medium"
        else:
            level = "low"
        
        return {
            "level": level,
            "score": avg_stability,
            "most_stable_metric": max(performance_metrics.keys(), key=lambda k: performance_metrics[k]["stability"]),
            "least_stable_metric": min(performance_metrics.keys(), key=lambda k: performance_metrics[k]["stability"])
        }

    def _generate_performance_recommendations(self, performance_metrics: Dict, overall_score: float) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        if overall_score < 0.6:
            recommendations.append("System performance is below acceptable levels - immediate attention required")
        
        # Find problematic metrics
        for metric_name, metrics in performance_metrics.items():
            if metrics["recent_anomalies"] > 3:
                recommendations.append(f"High anomaly rate in {metric_name} - investigate root cause")
            
            if metrics["stability"] < 0.5:
                recommendations.append(f"{metric_name} shows high volatility - consider stabilization measures")
        
        if not recommendations:
            recommendations.append("System performance is within acceptable ranges")
        
        return recommendations

    def _analyze_trend_pattern(self, values: List[float]) -> Dict:
        """Analyze trend patterns in time series data."""
        if len(values) < 5:
            return {"direction": "unknown", "strength": 0.0, "confidence": 0.0}
        
        # Linear regression for trend
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        # Determine trend direction and strength
        if abs(slope) < std_err:
            direction = "stable"
            strength = 0.0
        elif slope > 0:
            direction = "increasing"
            strength = abs(slope) / np.mean(values) if np.mean(values) > 0 else 0.0
        else:
            direction = "decreasing"
            strength = abs(slope) / np.mean(values) if np.mean(values) > 0 else 0.0
        
        return {
            "direction": direction,
            "strength": min(1.0, strength),
            "confidence": abs(r_value),
            "slope": slope,
            "r_squared": r_value ** 2
        }

    def _detect_seasonality(self, values: List[float], timestamps: List[datetime]) -> Dict:
        """Detect seasonal patterns in the data."""
        if len(values) < 24:  # Need at least 24 points for hourly seasonality
            return {"detected": False, "period": None, "strength": 0.0}
        
        # Simple autocorrelation-based seasonality detection
        # Check for common periods (hourly, daily patterns)
        periods_to_check = [12, 24, 48]  # 6h, 12h, 24h assuming 30min intervals
        
        best_period = None
        best_correlation = 0.0
        
        for period in periods_to_check:
            if len(values) >= 2 * period:
                # Calculate autocorrelation at this lag
                correlation = np.corrcoef(values[:-period], values[period:])[0, 1]
                if not np.isnan(correlation) and abs(correlation) > best_correlation:
                    best_correlation = abs(correlation)
                    best_period = period
        
        detected = best_correlation > 0.3  # Threshold for seasonality detection
        
        return {
            "detected": detected,
            "period": best_period,
            "strength": best_correlation,
            "period_hours": best_period * 0.5 if best_period else None  # Assuming 30min intervals
        }

    def _calculate_volatility_pattern(self, values: List[float]) -> Dict:
        """Calculate volatility patterns."""
        if len(values) < 10:
            return {"level": "unknown", "score": 0.0}
        
        # Calculate rolling volatility
        window_size = min(10, len(values) // 2)
        volatilities = []
        
        for i in range(window_size, len(values)):
            window = values[i-window_size:i]
            vol = np.std(window) / (np.mean(window) + 1e-6)  # Coefficient of variation
            volatilities.append(vol)
        
        avg_volatility = np.mean(volatilities) if volatilities else 0.0
        
        if avg_volatility < 0.1:
            level = "low"
        elif avg_volatility < 0.3:
            level = "medium"
        else:
            level = "high"
        
        return {
            "level": level,
            "score": avg_volatility,
            "trend": "increasing" if len(volatilities) > 5 and volatilities[-1] > volatilities[0] else "stable"
        }

    def _detect_anomaly_clusters(self, metric_state: MetricState, lookback_hours: int) -> List[Dict]:
        """Detect clusters of anomalies in time."""
        if not metric_state.recent_anomalies:
            return []
        
        # Filter anomalies within lookback period
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        recent_anomalies = [
            a for a in metric_state.recent_anomalies
            if datetime.fromisoformat(a["timestamp"].replace("Z", "+00:00")) > cutoff_time
        ]
        
        if len(recent_anomalies) < 2:
            return []
        
        # Simple clustering based on time proximity
        clusters = []
        current_cluster = [recent_anomalies[0]]
        
        for i in range(1, len(recent_anomalies)):
            prev_time = datetime.fromisoformat(recent_anomalies[i-1]["timestamp"].replace("Z", "+00:00"))
            curr_time = datetime.fromisoformat(recent_anomalies[i]["timestamp"].replace("Z", "+00:00"))
            
            time_diff = (curr_time - prev_time).total_seconds() / 60  # Minutes
            
            if time_diff <= 30:  # Within 30 minutes = same cluster
                current_cluster.append(recent_anomalies[i])
            else:
                if len(current_cluster) >= 2:  # Only keep clusters with 2+ anomalies
                    clusters.append({
                        "start_time": current_cluster[0]["timestamp"],
                        "end_time": current_cluster[-1]["timestamp"],
                        "anomaly_count": len(current_cluster),
                        "severity": max(a.get("severity", "medium") for a in current_cluster)
                    })
                current_cluster = [recent_anomalies[i]]
        
        # Don't forget the last cluster
        if len(current_cluster) >= 2:
            clusters.append({
                "start_time": current_cluster[0]["timestamp"],
                "end_time": current_cluster[-1]["timestamp"],
                "anomaly_count": len(current_cluster),
                "severity": max(a.get("severity", "medium") for a in current_cluster)
            })
        
        return clusters

    def _generate_pattern_forecast(self, values: List[float], steps: int) -> Dict:
        """Generate simple pattern-based forecast."""
        if len(values) < 5:
            return {"forecast": [], "confidence": 0.0}
        
        # Simple linear extrapolation
        x = np.arange(len(values))
        slope, intercept, r_value, _, _ = stats.linregress(x, values)
        
        forecast_x = np.arange(len(values), len(values) + steps)
        forecast_values = slope * forecast_x + intercept
        
        return {
            "forecast": forecast_values.tolist(),
            "confidence": abs(r_value),
            "method": "linear_extrapolation",
            "steps": steps
        }

    def _generate_pattern_insights(self, patterns: Dict) -> List[str]:
        """Generate insights from pattern analysis."""
        insights = []
        
        trend = patterns.get("trend_analysis", {})
        if trend.get("confidence", 0) > 0.7:
            direction = trend.get("direction", "unknown")
            strength = trend.get("strength", 0)
            insights.append(f"Strong {direction} trend detected (strength: {strength:.2f})")
        
        seasonality = patterns.get("seasonality", {})
        if seasonality.get("detected", False):
            period_hours = seasonality.get("period_hours", 0)
            insights.append(f"Seasonal pattern detected with {period_hours:.1f}h period")
        
        volatility = patterns.get("volatility", {})
        vol_level = volatility.get("level", "unknown")
        if vol_level != "unknown":
            insights.append(f"Volatility level: {vol_level}")
        
        clusters = patterns.get("anomaly_clusters", [])
        if clusters:
            insights.append(f"{len(clusters)} anomaly clusters detected")
        
        return insights

    def _generate_pattern_recommendations(self, patterns: Dict) -> List[str]:
        """Generate recommendations based on pattern analysis."""
        recommendations = []
        
        trend = patterns.get("trend_analysis", {})
        if trend.get("direction") == "decreasing" and trend.get("confidence", 0) > 0.7:
            recommendations.append("Declining trend detected - investigate potential causes")
        
        volatility = patterns.get("volatility", {})
        if volatility.get("level") == "high":
            recommendations.append("High volatility detected - consider stabilization measures")
        
        clusters = patterns.get("anomaly_clusters", [])
        if len(clusters) > 2:
            recommendations.append("Multiple anomaly clusters - systematic issue may be present")
        
        if not recommendations:
            recommendations.append("Patterns appear normal - continue monitoring")
        
        return recommendations

    def _perform_anomaly_diagnosis(self, request: DiagnosisRequest) -> Dict:
        """Perform anomaly-based diagnosis."""
        anomaly_metrics = []
        evidence = []
        
        for metric_name, metric_state in self.system_state.metrics.items():
            if metric_state.recent_anomalies:
                recent_anomaly = metric_state.recent_anomalies[-1]
                anomaly_metrics.append({
                    "metric": metric_name,
                    "anomaly": recent_anomaly,
                    "z_score": recent_anomaly.get("z_score", 0)
                })
                evidence.append(f"Anomaly in {metric_name}: z-score {recent_anomaly.get('z_score', 0):.2f}")
        
        hypotheses = []
        if anomaly_metrics:
            anomaly_metrics.sort(key=lambda x: x["z_score"], reverse=True)
            primary_anomaly = anomaly_metrics[0]
            
            hypotheses.append({
                "hypothesis": f"Primary anomaly in {primary_anomaly['metric']}",
                "probability": min(0.9, primary_anomaly["z_score"] / 5.0),
                "reasoning": f"Statistical deviation of {primary_anomaly['z_score']:.2f} standard deviations",
                "evidence": [f"Z-score: {primary_anomaly['z_score']:.2f}"],
                "rank": 1
            })
        
        return {
            "hypotheses": hypotheses,
            "evidence": evidence,
            "confidence": 0.8 if anomaly_metrics else 0.3
        }

    def _perform_causal_diagnosis(self, request: DiagnosisRequest) -> Dict:
        """Perform causal analysis for SWITCH system."""
        # SWITCH causal relationships
        causal_relationships = {
            "cpu_usage": ["image_processing_time", "system_stability"],
            "image_processing_time": ["utility", "user_satisfaction"],
            "confidence": ["utility", "detection_accuracy"],
            "utility": ["model_switching_frequency", "system_performance"],
            "model_complexity": ["cpu_usage", "image_processing_time", "confidence"]
        }
        
        hypotheses = []
        evidence = []
        
        # Identify metrics with issues
        problematic_metrics = []
        for metric_name, metric_state in self.system_state.metrics.items():
            if metric_state.recent_anomalies or len(metric_state.values) > 5:
                recent_values = list(metric_state.values)[-5:]
                if len(recent_values) >= 2:
                    trend = (recent_values[-1] - recent_values[0]) / len(recent_values)
                    if abs(trend) > np.std(recent_values):
                        problematic_metrics.append({
                            "metric": metric_name,
                            "trend": trend,
                            "severity": abs(trend) / (np.mean(recent_values) + 1e-6)
                        })
        
        # Trace causal chains
        for problem in problematic_metrics:
            metric = problem["metric"]
            if metric in causal_relationships:
                affected_metrics = causal_relationships[metric]
                for affected in affected_metrics:
                    if affected in self.system_state.metrics:
                        hypotheses.append({
                            "hypothesis": f"{metric} degradation causing {affected} issues",
                            "probability": 0.7,
                            "reasoning": f"Causal relationship: {metric} → {affected}",
                            "evidence": [f"Trend in {metric}: {problem['trend']:.3f}"],
                            "rank": len(hypotheses) + 1
                        })
                        evidence.append(f"Causal chain: {metric} → {affected}")
        
        return {
            "hypotheses": hypotheses,
            "evidence": evidence,
            "confidence": 0.75 if hypotheses else 0.4
        }

    def _perform_switch_specific_diagnosis(self, request: DiagnosisRequest) -> Dict:
        """Perform SWITCH-specific failure mode diagnosis."""
        hypotheses = []
        evidence = []
        
        # Check for model switching thrashing
        if self._detect_model_thrashing():
            hypotheses.append({
                "hypothesis": "Model switching thrashing detected",
                "probability": 0.85,
                "reasoning": "Rapid model switches indicate unstable optimization",
                "evidence": ["Frequent model changes", "Utility oscillation"],
                "rank": 1
            })
            evidence.extend(["Model switching frequency above threshold", "Utility instability"])
        
        # Check for utility degradation spiral
        if self._detect_utility_spiral():
            hypotheses.append({
                "hypothesis": "Utility degradation spiral",
                "probability": 0.8,
                "reasoning": "Continuous utility decline suggests systematic issue",
                "evidence": ["Declining utility trend", "Failed optimization attempts"],
                "rank": 2
            })
            evidence.extend(["Utility trend negative", "Multiple failed adaptations"])
        
        # Check for resource exhaustion
        if self._detect_resource_exhaustion():
            hypotheses.append({
                "hypothesis": "System resource exhaustion",
                "probability": 0.9,
                "reasoning": "Resource limits preventing optimal operation",
                "evidence": ["High CPU usage", "Memory pressure"],
                "rank": 1
            })
            evidence.extend(["CPU usage > 85%", "Resource constraints active"])
        
        # Check for model performance degradation
        current_model = self._get_current_switch_model()
        if self._detect_model_performance_degradation(current_model):
            hypotheses.append({
                "hypothesis": f"Performance degradation in {current_model}",
                "probability": 0.75,
                "reasoning": "Current model not meeting expected performance",
                "evidence": [f"{current_model} underperforming", "Metrics below baseline"],
                "rank": 3
            })
            evidence.append(f"Model {current_model} performance below expectations")
        
        return {
            "hypotheses": hypotheses,
            "evidence": evidence,
            "confidence": 0.8 if hypotheses else 0.5
        }

    def _perform_temporal_diagnosis(self, request: DiagnosisRequest) -> Dict:
        """Perform temporal pattern-based diagnosis."""
        hypotheses = []
        evidence = []
        
        # Analyze temporal patterns in key metrics
        for metric_name in ["utility", "image_processing_time", "confidence"]:
            if metric_name not in self.system_state.metrics:
                continue
            
            metric_state = self.system_state.metrics[metric_name]
            if len(metric_state.values) < 10:
                continue
            
            recent_values = list(metric_state.values)[-20:]  # Last 20 values
            
            # Check for sudden changes
            if len(recent_values) >= 10:
                first_half = recent_values[:10]
                second_half = recent_values[10:]
                
                first_mean = np.mean(first_half)
                second_mean = np.mean(second_half)
                
                if abs(second_mean - first_mean) > 2 * np.std(recent_values):
                    change_type = "improvement" if second_mean > first_mean else "degradation"
                    hypotheses.append({
                        "hypothesis": f"Sudden {change_type} in {metric_name}",
                        "probability": 0.7,
                        "reasoning": f"Significant change detected in recent {metric_name} values",
                        "evidence": [f"Mean change: {first_mean:.3f} → {second_mean:.3f}"],
                        "rank": len(hypotheses) + 1
                    })
                    evidence.append(f"Temporal shift in {metric_name}: {change_type}")
            
            # Check for cyclical patterns
            if self._detect_cyclical_issues(recent_values):
                hypotheses.append({
                    "hypothesis": f"Cyclical issues in {metric_name}",
                    "probability": 0.6,
                    "reasoning": "Repeating pattern of issues detected",
                    "evidence": ["Cyclical degradation pattern"],
                    "rank": len(hypotheses) + 1
                })
                evidence.append(f"Cyclical pattern detected in {metric_name}")
        
        return {
            "hypotheses": hypotheses,
            "evidence": evidence,
            "confidence": 0.7 if hypotheses else 0.4
        }

    def _perform_correlation_diagnosis(self, request: DiagnosisRequest) -> Dict:
        """Perform correlation-based diagnosis."""
        hypotheses = []
        evidence = []
        
        # Find strongly correlated metrics with issues
        for (metric1, metric2), correlation in self.system_state.correlations.items():
            if abs(correlation) > self.correlation_threshold:
                # Check if either metric has recent anomalies
                metric1_issues = metric1 in self.system_state.metrics and self.system_state.metrics[metric1].recent_anomalies
                metric2_issues = metric2 in self.system_state.metrics and self.system_state.metrics[metric2].recent_anomalies
                
                if metric1_issues or metric2_issues:
                    primary = metric1 if metric1_issues else metric2
                    secondary = metric2 if metric1_issues else metric1
                    
                    hypotheses.append({
                        "hypothesis": f"Correlated failure: {primary} affecting {secondary}",
                        "probability": min(0.9, abs(correlation)),
                        "reasoning": f"Strong correlation ({correlation:.2f}) suggests cascading effect",
                        "evidence": [f"Correlation coefficient: {correlation:.2f}"],
                        "rank": len(hypotheses) + 1
                    })
                    evidence.append(f"Strong correlation between {primary} and {secondary}: {correlation:.2f}")
        
        return {
            "hypotheses": hypotheses,
            "evidence": evidence,
            "confidence": 0.8 if hypotheses else 0.3
        }

    def _rank_diagnostic_hypotheses(self, hypotheses: List[Dict], evidence: List[str]) -> List[Dict]:
        """Rank diagnostic hypotheses by likelihood and evidence strength."""
        if not hypotheses:
            return []
        
        # Score each hypothesis
        for i, hypothesis in enumerate(hypotheses):
            score = 0.0
            
            # Base probability score
            score += hypothesis.get("probability", 0.5) * 0.4
            
            # Evidence strength score
            evidence_count = len(hypothesis.get("evidence", []))
            score += min(0.3, evidence_count * 0.1)
            
            # Source analysis diversity bonus
            if hypothesis.get("source_analysis") in ["switch_specific_analysis", "causal_analysis"]:
                score += 0.2
            
            # Recency bonus (if hypothesis relates to recent issues)
            if "recent" in hypothesis.get("hypothesis", "").lower():
                score += 0.1
            
            hypothesis["diagnostic_score"] = score
        
        # Sort by diagnostic score
        return sorted(hypotheses, key=lambda h: h.get("diagnostic_score", 0), reverse=True)

    def _build_enhanced_causal_chain(self, hypotheses: List[Dict], diagnostic_results: Dict) -> str:
        """Build enhanced causal chain from ranked hypotheses."""
        if not hypotheses:
            return "No clear causal chain identified"
        
        # Take top 3 hypotheses and build logical chain
        top_hypotheses = hypotheses[:3]
        
        # Try to build logical sequence
        chain_parts = []
        for hypothesis in top_hypotheses:
            h_text = hypothesis.get("hypothesis", "Unknown")
            probability = hypothesis.get("probability", 0.5)
            chain_parts.append(f"{h_text} ({probability:.1%})")
        
        return " → ".join(chain_parts)

    def _calculate_diagnostic_confidence(self, confidence_scores: List[float], evidence_count: int) -> float:
        """Calculate overall diagnostic confidence."""
        if not confidence_scores:
            return 0.3
        
        # Base confidence from individual analyses
        base_confidence = np.mean(confidence_scores)
        
        # Boost confidence with more evidence
        evidence_boost = min(0.2, evidence_count * 0.02)
        
        # Boost confidence with multiple analysis methods
        method_boost = min(0.1, len(confidence_scores) * 0.02)
        
        return min(0.95, base_confidence + evidence_boost + method_boost)

    def _generate_diagnostic_recommendations(self, hypotheses: List[Dict], diagnostic_results: Dict) -> List[str]:
        """Generate actionable diagnostic recommendations."""
        recommendations = []
        
        if not hypotheses:
            recommendations.append("No specific issues identified - continue monitoring")
            return recommendations
        
        top_hypothesis = hypotheses[0]
        hypothesis_text = top_hypothesis.get("hypothesis", "").lower()
        
        # SWITCH-specific recommendations
        if "thrashing" in hypothesis_text:
            recommendations.extend([
                "Increase model switching interval to reduce thrashing",
                "Review utility function parameters",
                "Implement switching cooldown period"
            ])
        
        if "utility" in hypothesis_text and "degradation" in hypothesis_text:
            recommendations.extend([
                "Investigate utility function calculation",
                "Check model performance baselines",
                "Consider recalibrating utility weights"
            ])
        
        if "resource" in hypothesis_text:
            recommendations.extend([
                "Scale system resources or switch to lighter model",
                "Implement resource monitoring alerts",
                "Optimize resource usage patterns"
            ])
        
        if "anomaly" in hypothesis_text:
            recommendations.extend([
                "Investigate root cause of anomalous behavior",
                "Review recent system changes",
                "Implement anomaly alerting"
            ])
        
        # Generic recommendations if no specific ones apply
        if not recommendations:
            recommendations.extend([
                "Monitor system closely for pattern development",
                "Consider preventive maintenance",
                "Review system configuration"
            ])
        
        return recommendations[:5]  # Limit to top 5 recommendations

    def _generate_diagnostic_explanation(self, hypotheses: List[Dict], diagnostic_results: Dict, metrics_count: int) -> str:
        """Generate comprehensive diagnostic explanation."""
        if not hypotheses:
            return f"Analysis of {metrics_count} metrics found no clear issues. System appears to be operating normally."
        
        top_hypothesis = hypotheses[0]
        analyses_count = len([r for r in diagnostic_results.values() if r.get("hypotheses")])
        total_evidence = sum(len(r.get("evidence", [])) for r in diagnostic_results.values())
        
        explanation = (
            f"Multi-layered diagnostic analysis of {metrics_count} metrics using {analyses_count} analysis methods "
            f"identified {len(hypotheses)} potential issues with {total_evidence} pieces of supporting evidence. "
            f"Primary concern: {top_hypothesis.get('hypothesis', 'Unknown issue')} "
            f"(confidence: {top_hypothesis.get('probability', 0.5):.1%}). "
            f"Recommendation: {top_hypothesis.get('reasoning', 'Further investigation needed')}."
        )
        
        return explanation

    def _assess_diagnostic_urgency(self, hypotheses: List[Dict]) -> str:
        """Assess urgency level of diagnostic findings."""
        if not hypotheses:
            return "low"
        
        top_hypothesis = hypotheses[0]
        hypothesis_text = top_hypothesis.get("hypothesis", "").lower()
        probability = top_hypothesis.get("probability", 0.5)
        
        # High urgency conditions
        if any(keyword in hypothesis_text for keyword in ["critical", "failure", "exhaustion", "thrashing"]):
            return "high"
        
        # Medium urgency conditions
        if probability > 0.8 or any(keyword in hypothesis_text for keyword in ["degradation", "anomaly", "spiral"]):
            return "medium"
        
        return "low"

    def _identify_affected_systems(self, hypotheses: List[Dict]) -> List[str]:
        """Identify systems affected by diagnostic findings."""
        affected_systems = set()
        
        for hypothesis in hypotheses[:3]:  # Top 3 hypotheses
            hypothesis_text = hypothesis.get("hypothesis", "").lower()
            
            if any(keyword in hypothesis_text for keyword in ["model", "yolo", "switching"]):
                affected_systems.add("model_switching")
            
            if any(keyword in hypothesis_text for keyword in ["utility", "performance"]):
                affected_systems.add("performance_optimization")
            
            if any(keyword in hypothesis_text for keyword in ["cpu", "resource", "memory"]):
                affected_systems.add("resource_management")
            
            if any(keyword in hypothesis_text for keyword in ["processing", "response", "latency"]):
                affected_systems.add("processing_pipeline")
        
        return list(affected_systems) if affected_systems else ["general_system"]

    def _analyze_health_trends(self) -> Dict:
        """Analyze health trends over time."""
        if not self.system_state.metrics:
            return {"trend": "unknown", "direction": "stable"}
        
        # Simple health trend analysis
        health_scores = []
        for metric_name, metric_state in self.system_state.metrics.items():
            if len(metric_state.values) >= 5:
                recent_values = list(metric_state.values)[-5:]
                stability = 1.0 / (1.0 + np.std(recent_values))
                anomaly_health = max(0.0, 1.0 - len(metric_state.recent_anomalies) / 5.0)
                health_score = (stability + anomaly_health) / 2.0
                health_scores.append(health_score)
        
        if len(health_scores) < 2:
            return {"trend": "insufficient_data", "direction": "stable"}
        
        # Calculate trend
        recent_health = np.mean(health_scores[-3:]) if len(health_scores) >= 3 else health_scores[-1]
        older_health = np.mean(health_scores[:-3]) if len(health_scores) >= 6 else health_scores[0]
        
        if recent_health > older_health + 0.1:
            direction = "improving"
        elif recent_health < older_health - 0.1:
            direction = "degrading"
        else:
            direction = "stable"
        
        return {
            "trend": "analyzed",
            "direction": direction,
            "current_health": recent_health,
            "health_change": recent_health - older_health
        }

    def _generate_health_alerts(self, health_metrics: Dict, system_indicators: Dict) -> List[str]:
        """Generate health-based alerts."""
        alerts = []
        
        overall_status = system_indicators.get("overall_status", "unknown")
        if overall_status == "critical":
            alerts.append("CRITICAL: System health is in critical state")
        elif overall_status == "degraded":
            alerts.append("WARNING: System health is degraded")
        
        critical_metrics = system_indicators.get("metrics_critical", 0)
        if critical_metrics > 0:
            alerts.append(f"ALERT: {critical_metrics} metrics in critical state")
        
        return alerts

    def _generate_health_recommendations(self, health_metrics: Dict, system_indicators: Dict) -> List[str]:
        """Generate health-based recommendations."""
        recommendations = []
        
        overall_status = system_indicators.get("overall_status", "unknown")
        if overall_status == "critical":
            recommendations.extend([
                "Immediate attention required for critical metrics",
                "Consider system restart or failover",
                "Review recent changes and configurations"
            ])
        elif overall_status == "degraded":
            recommendations.extend([
                "Monitor degraded metrics closely",
                "Consider preventive maintenance",
                "Review system resource usage"
            ])
        else:
            recommendations.append("System health appears normal - continue monitoring")
        
        return recommendations

    def _generate_kalman_forecast(self, metric_state: MetricState, horizon_minutes: int) -> Dict:
        """Generate Kalman filter-based forecast."""
        if len(metric_state.values) < 5:
            return {"forecast": [], "confidence": 0.0, "method": "insufficient_data"}
        
        # Create a copy of the Kalman filter for forecasting
        forecast_kf = self._copy_kalman_filter(metric_state.kf)
        
        # Generate forecast steps
        steps = min(horizon_minutes // 5, 20)  # Max 20 steps, 5-minute intervals
        forecast_values = []
        confidence_values = []
        
        for step in range(steps):
            forecast_kf.predict()
            predicted_value = float(forecast_kf.x[0, 0])
            uncertainty = float(np.sqrt(forecast_kf.P[0, 0]))
            confidence = max(0.1, 1.0 / (1.0 + uncertainty))
            
            forecast_values.append(predicted_value)
            confidence_values.append(confidence)
        
        return {
            "forecast": forecast_values,
            "confidence": np.mean(confidence_values),
            "method": "kalman_filter",
            "steps": steps,
            "uncertainty": np.mean([1.0/c - 1.0 for c in confidence_values])
        }

    def _generate_optimistic_forecast(self, base_forecasts: Dict) -> Dict:
        """Generate optimistic scenario forecast."""
        optimistic = {}
        for metric_name, forecast in base_forecasts.items():
            if isinstance(forecast, dict) and "forecast" in forecast:
                # Improve forecast by 10-20%
                base_values = forecast["forecast"]
                if metric_name in ["utility", "confidence"]:
                    # Higher is better
                    optimistic_values = [v * 1.15 for v in base_values]
                elif metric_name in ["image_processing_time", "cpu_usage"]:
                    # Lower is better
                    optimistic_values = [v * 0.85 for v in base_values]
                else:
                    optimistic_values = base_values
                
                optimistic[metric_name] = {
                    **forecast,
                    "forecast": optimistic_values,
                    "scenario": "optimistic"
                }
        
        return optimistic

    def _generate_pessimistic_forecast(self, base_forecasts: Dict) -> Dict:
        """Generate pessimistic scenario forecast."""
        pessimistic = {}
        for metric_name, forecast in base_forecasts.items():
            if isinstance(forecast, dict) and "forecast" in forecast:
                # Worsen forecast by 10-20%
                base_values = forecast["forecast"]
                if metric_name in ["utility", "confidence"]:
                    # Higher is better, so reduce
                    pessimistic_values = [v * 0.85 for v in base_values]
                elif metric_name in ["image_processing_time", "cpu_usage"]:
                    # Lower is better, so increase
                    pessimistic_values = [v * 1.15 for v in base_values]
                else:
                    pessimistic_values = base_values
                
                pessimistic[metric_name] = {
                    **forecast,
                    "forecast": pessimistic_values,
                    "scenario": "pessimistic"
                }
        
        return pessimistic

    async def _forecast_model_switch_impact(self, base_forecasts: Dict) -> Dict:
        """Forecast impact of potential model switches."""
        model_switch_forecasts = {}
        
        # Get current model and alternatives
        current_model = self._get_current_switch_model()
        model_profiles = self.switch_context.get("yolo_models", {})
        
        for model_name, profile in model_profiles.items():
            if model_name == current_model:
                continue  # Skip current model
            
            model_forecast = {}
            for metric_name, forecast in base_forecasts.items():
                if isinstance(forecast, dict) and "forecast" in forecast:
                    base_values = forecast["forecast"]
                    
                    # Apply model-specific adjustments
                    if metric_name == "image_processing_time":
                        expected_rt = profile.get("expected_response_time", 0.2)
                        current_rt = base_values[0] if base_values else 0.2
                        adjustment_factor = expected_rt / (current_rt + 1e-6)
                        adjusted_values = [v * adjustment_factor for v in base_values]
                    elif metric_name == "confidence":
                        expected_conf = profile.get("expected_confidence", 0.75)
                        current_conf = base_values[0] if base_values else 0.75
                        adjustment_factor = expected_conf / (current_conf + 1e-6)
                        adjusted_values = [v * adjustment_factor for v in base_values]
                    else:
                        adjusted_values = base_values
                    
                    model_forecast[metric_name] = {
                        **forecast,
                        "forecast": adjusted_values,
                        "scenario": f"switch_to_{model_name}"
                    }
            
            model_switch_forecasts[model_name] = model_forecast
        
        return model_switch_forecasts

    def _generate_forecast_insights(self, scenarios: Dict) -> List[str]:
        """Generate insights from forecast scenarios."""
        insights = []
        
        # Compare scenarios
        if "optimistic" in scenarios and "pessimistic" in scenarios:
            insights.append("Multiple scenarios analyzed for uncertainty quantification")
        
        if "model_switch_impact" in scenarios:
            switch_scenarios = scenarios["model_switch_impact"]
            if switch_scenarios:
                insights.append(f"Model switching impact analyzed for {len(switch_scenarios)} alternatives")
        
        # Analyze trends in current trajectory
        current_scenario = scenarios.get("current_trajectory", {})
        if current_scenario:
            for metric_name, forecast in current_scenario.items():
                if isinstance(forecast, dict) and "forecast" in forecast:
                    values = forecast["forecast"]
                    if len(values) >= 2:
                        trend = "increasing" if values[-1] > values[0] else "decreasing"
                        insights.append(f"{metric_name} forecast shows {trend} trend")
        
        return insights

    def _generate_forecast_recommendations(self, scenarios: Dict) -> List[str]:
        """Generate recommendations based on forecast scenarios."""
        recommendations = []
        
        # Analyze current trajectory
        current_scenario = scenarios.get("current_trajectory", {})
        if "utility" in current_scenario:
            utility_forecast = current_scenario["utility"]
            if isinstance(utility_forecast, dict) and "forecast" in utility_forecast:
                values = utility_forecast["forecast"]
                if values and values[-1] < 0.5:
                    recommendations.append("Utility forecast shows decline - consider optimization actions")
        
        # Analyze model switching opportunities
        if "model_switch_impact" in scenarios:
            switch_scenarios = scenarios["model_switch_impact"]
            best_model = None
            best_utility = -1
            
            for model_name, model_forecast in switch_scenarios.items():
                if "utility" in model_forecast:
                    utility_forecast = model_forecast["utility"]
                    if isinstance(utility_forecast, dict) and "forecast" in utility_forecast:
                        values = utility_forecast["forecast"]
                        if values and values[-1] > best_utility:
                            best_utility = values[-1]
                            best_model = model_name
            
            if best_model:
                recommendations.append(f"Consider switching to {best_model} for improved utility")
        
        if not recommendations:
            recommendations.append("Continue monitoring - no immediate actions recommended")
        
        return recommendations

    def _analyze_forecast_uncertainty(self, scenarios: Dict) -> Dict:
        """Analyze uncertainty across forecast scenarios."""
        uncertainty_analysis = {
            "overall_uncertainty": 0.5,
            "scenario_spread": 0.0,
            "confidence_range": {"min": 0.5, "max": 0.5}
        }
        
        # Calculate spread between optimistic and pessimistic scenarios
        if "optimistic" in scenarios and "pessimistic" in scenarios:
            opt_scenario = scenarios["optimistic"]
            pess_scenario = scenarios["pessimistic"]
            
            spreads = []
            for metric_name in opt_scenario.keys():
                if metric_name in pess_scenario:
                    opt_forecast = opt_scenario[metric_name]
                    pess_forecast = pess_scenario[metric_name]
                    
                    if (isinstance(opt_forecast, dict) and "forecast" in opt_forecast and
                        isinstance(pess_forecast, dict) and "forecast" in pess_forecast):
                        
                        opt_values = opt_forecast["forecast"]
                        pess_values = pess_forecast["forecast"]
                        
                        if opt_values and pess_values:
                            spread = abs(opt_values[-1] - pess_values[-1])
                            spreads.append(spread)
            
            if spreads:
                uncertainty_analysis["scenario_spread"] = np.mean(spreads)
        
        # Calculate confidence range
        confidences = []
        for scenario_name, scenario in scenarios.items():
            for metric_name, forecast in scenario.items():
                if isinstance(forecast, dict) and "confidence" in forecast:
                    confidences.append(forecast["confidence"])
        
        if confidences:
            uncertainty_analysis["confidence_range"] = {
                "min": min(confidences),
                "max": max(confidences)
            }
            uncertainty_analysis["overall_uncertainty"] = 1.0 - np.mean(confidences)
        
        return uncertainty_analysis

    def _detect_model_thrashing(self) -> bool:
        """Detect if model switching is thrashing."""
        # Simple heuristic: check if utility has been oscillating
        if "utility" not in self.system_state.metrics:
            return False
        
        utility_state = self.system_state.metrics["utility"]
        if len(utility_state.values) < 10:
            return False
        
        recent_values = list(utility_state.values)[-10:]
        
        # Count direction changes (sign of derivative changes)
        direction_changes = 0
        for i in range(2, len(recent_values)):
            prev_diff = recent_values[i-1] - recent_values[i-2]
            curr_diff = recent_values[i] - recent_values[i-1]
            
            if prev_diff * curr_diff < 0:  # Sign change
                direction_changes += 1
        
        # Thrashing if more than 4 direction changes in 10 samples
        return direction_changes > 4

    def _detect_utility_spiral(self) -> bool:
        """Detect utility degradation spiral."""
        if "utility" not in self.system_state.metrics:
            return False
        
        utility_state = self.system_state.metrics["utility"]
        if len(utility_state.values) < 8:
            return False
        
        recent_values = list(utility_state.values)[-8:]
        
        # Check if utility has been consistently declining
        declining_count = 0
        for i in range(1, len(recent_values)):
            if recent_values[i] < recent_values[i-1]:
                declining_count += 1
        
        # Spiral if more than 5 out of 7 changes are declining
        return declining_count > 5

    def _detect_resource_exhaustion(self) -> bool:
        """Detect system resource exhaustion."""
        if "cpu_usage" not in self.system_state.metrics:
            return False
        
        cpu_state = self.system_state.metrics["cpu_usage"]
        if not cpu_state.values:
            return False
        
        current_cpu = cpu_state.values[-1]
        return current_cpu > 85.0  # Above 85% is considered exhaustion

    def _detect_model_performance_degradation(self, model_name: str) -> bool:
        """Detect if current model is underperforming."""
        model_profiles = self.switch_context.get("yolo_models", {})
        if model_name not in model_profiles:
            return False
        
        expected_profile = model_profiles[model_name]
        
        # Check response time
        if "image_processing_time" in self.system_state.metrics:
            rt_state = self.system_state.metrics["image_processing_time"]
            if rt_state.values:
                current_rt = rt_state.values[-1]
                expected_rt = expected_profile.get("expected_response_time", 0.2)
                if current_rt > expected_rt * 1.5:  # 50% worse than expected
                    return True
        
        # Check confidence
        if "confidence" in self.system_state.metrics:
            conf_state = self.system_state.metrics["confidence"]
            if conf_state.values:
                current_conf = conf_state.values[-1]
                expected_conf = expected_profile.get("expected_confidence", 0.75)
                if current_conf < expected_conf * 0.8:  # 20% worse than expected
                    return True
        
        return False

    def _detect_cyclical_issues(self, values: List[float]) -> bool:
        """Detect cyclical patterns that might indicate systematic issues."""
        if len(values) < 12:
            return False
        
        # Simple autocorrelation check for cyclical patterns
        # Check for patterns with period 3, 4, 6
        for period in [3, 4, 6]:
            if len(values) >= 2 * period:
                correlation = np.corrcoef(values[:-period], values[period:])[0, 1]
                if not np.isnan(correlation) and correlation > 0.6:
                    return True
        
        return False
    
    def _set_initialized(self, initialized: bool) -> None:
        """Set the initialization status."""
        self._initialized = initialized
    
    @property
    def is_initialized(self) -> bool:
        """Check if the model is initialized."""
        return getattr(self, '_initialized', False)
    
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
