"""
SWIM-Specific Adaptation Strategies

This module implements adaptation strategies specifically designed for SWIM
(Simulated Web Infrastructure Manager) system characteristics and constraints.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from polaris_refactored.src.domain.models import (
    SystemState, AdaptationAction, MetricValue, HealthStatus
)


class AdaptationTrigger(Enum):
    """Types of adaptation triggers."""
    THRESHOLD_VIOLATION = "threshold_violation"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    PREDICTED_ISSUE = "predicted_issue"
    RESOURCE_OPTIMIZATION = "resource_optimization"
    MANUAL = "manual"


@dataclass
class SwimMetrics:
    """SWIM-specific metrics structure."""
    server_count: int
    active_servers: int
    max_servers: int
    dimmer: float
    basic_response_time: Optional[float] = None
    optional_response_time: Optional[float] = None
    server_utilization: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        
        # Calculate server utilization if not provided
        if self.server_utilization is None and self.max_servers > 0:
            self.server_utilization = self.active_servers / self.max_servers


@dataclass
class AdaptationContext:
    """Context information for adaptation decisions."""
    current_metrics: SwimMetrics
    historical_metrics: List[SwimMetrics]
    system_state: SystemState
    trigger: AdaptationTrigger
    confidence: float
    constraints: Dict[str, Any]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SwimAdaptationStrategy(ABC):
    """Base class for SWIM adaptation strategies."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.enabled = config.get("enabled", True)
        self.weight = config.get("weight", 1.0)
    
    @abstractmethod
    async def should_adapt(self, context: AdaptationContext) -> Tuple[bool, float]:
        """
        Determine if adaptation is needed.
        
        Returns:
            Tuple of (should_adapt, confidence_score)
        """
        pass
    
    @abstractmethod
    async def plan_adaptation(self, context: AdaptationContext) -> List[AdaptationAction]:
        """
        Plan adaptation actions based on context.
        
        Returns:
            List of adaptation actions to execute
        """
        pass
    
    async def validate_actions(self, actions: List[AdaptationAction], 
                             context: AdaptationContext) -> List[AdaptationAction]:
        """Validate and filter actions based on constraints."""
        valid_actions = []
        
        for action in actions:
            if await self._validate_single_action(action, context):
                valid_actions.append(action)
            else:
                self.logger.warning(f"Action {action.action_id} failed validation")
        
        return valid_actions
    
    async def _validate_single_action(self, action: AdaptationAction, 
                                    context: AdaptationContext) -> bool:
        """Validate a single action against constraints."""
        constraints = context.constraints
        current_metrics = context.current_metrics
        
        action_type = action.action_type.upper()
        
        if action_type in ["ADD_SERVER", "SCALE_UP"]:
            return current_metrics.server_count < constraints.get("max_servers", 10)
        
        elif action_type in ["REMOVE_SERVER", "SCALE_DOWN"]:
            return current_metrics.server_count > constraints.get("min_servers", 1)
        
        elif action_type in ["SET_DIMMER", "ADJUST_QOS"]:
            dimmer_value = action.parameters.get("value", 1.0)
            min_dimmer = constraints.get("min_dimmer", 0.0)
            max_dimmer = constraints.get("max_dimmer", 1.0)
            return min_dimmer <= dimmer_value <= max_dimmer
        
        return True


class ReactiveAdaptationStrategy(SwimAdaptationStrategy):
    """Reactive adaptation strategy that responds to threshold violations."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("reactive", config)
        
        # Threshold configuration
        self.response_time_threshold = config.get("response_time_threshold", 1000.0)
        self.utilization_high_threshold = config.get("utilization_high_threshold", 0.8)
        self.utilization_low_threshold = config.get("utilization_low_threshold", 0.3)
        self.dimmer_low_threshold = config.get("dimmer_low_threshold", 0.7)
    
    async def should_adapt(self, context: AdaptationContext) -> Tuple[bool, float]:
        """Check if reactive adaptation is needed based on thresholds."""
        metrics = context.current_metrics
        
        # Check response time threshold
        if (metrics.basic_response_time and 
            metrics.basic_response_time > self.response_time_threshold):
            self.logger.info(f"Response time threshold violated: {metrics.basic_response_time}ms")
            return True, 0.9
        
        # Check high utilization threshold
        if (metrics.server_utilization and 
            metrics.server_utilization > self.utilization_high_threshold):
            self.logger.info(f"High utilization detected: {metrics.server_utilization:.2f}")
            return True, 0.8
        
        # Check low utilization threshold
        if (metrics.server_utilization and 
            metrics.server_utilization < self.utilization_low_threshold):
            self.logger.info(f"Low utilization detected: {metrics.server_utilization:.2f}")
            return True, 0.7
        
        # Check dimmer threshold
        if metrics.dimmer < self.dimmer_low_threshold:
            self.logger.info(f"Low dimmer value detected: {metrics.dimmer}")
            return True, 0.6
        
        return False, 0.0
    
    async def plan_adaptation(self, context: AdaptationContext) -> List[AdaptationAction]:
        """Plan reactive adaptation actions."""
        actions = []
        metrics = context.current_metrics
        
        # High response time or high utilization -> scale up or reduce dimmer
        if ((metrics.basic_response_time and 
             metrics.basic_response_time > self.response_time_threshold) or
            (metrics.server_utilization and 
             metrics.server_utilization > self.utilization_high_threshold)):
            
            # Try to add server first
            if metrics.server_count < context.constraints.get("max_servers", 10):
                actions.append(AdaptationAction(
                    action_type="ADD_SERVER",
                    parameters={},
                    expected_impact={"server_count": 1, "utilization_reduction": 0.1}
                ))
            
            # If can't add server, reduce dimmer
            elif metrics.dimmer > 0.1:
                new_dimmer = max(0.1, metrics.dimmer - 0.1)
                actions.append(AdaptationAction(
                    action_type="SET_DIMMER",
                    parameters={"value": new_dimmer},
                    expected_impact={"dimmer": new_dimmer, "response_time_reduction": 100}
                ))
        
        # Low utilization -> scale down
        elif (metrics.server_utilization and 
              metrics.server_utilization < self.utilization_low_threshold and
              metrics.server_count > 1):
            
            actions.append(AdaptationAction(
                action_type="REMOVE_SERVER",
                parameters={},
                expected_impact={"server_count": -1, "utilization_increase": 0.1}
            ))
        
        # Low dimmer -> increase dimmer if utilization allows
        elif (metrics.dimmer < self.dimmer_low_threshold and
              metrics.server_utilization and 
              metrics.server_utilization < 0.7):
            
            new_dimmer = min(1.0, metrics.dimmer + 0.1)
            actions.append(AdaptationAction(
                action_type="SET_DIMMER",
                parameters={"value": new_dimmer},
                expected_impact={"dimmer": new_dimmer, "performance_improvement": 0.1}
            ))
        
        return actions


class PredictiveAdaptationStrategy(SwimAdaptationStrategy):
    """Predictive adaptation strategy that anticipates future issues."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("predictive", config)
        
        self.prediction_window = config.get("prediction_window", 300.0)  # 5 minutes
        self.trend_analysis_window = config.get("trend_analysis_window", 600.0)  # 10 minutes
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        self.min_trend_points = config.get("min_trend_points", 5)
    
    async def should_adapt(self, context: AdaptationContext) -> Tuple[bool, float]:
        """Check if predictive adaptation is needed based on trends."""
        if len(context.historical_metrics) < self.min_trend_points:
            return False, 0.0
        
        # Analyze trends in key metrics
        response_time_trend = self._analyze_trend(
            context.historical_metrics, "basic_response_time"
        )
        utilization_trend = self._analyze_trend(
            context.historical_metrics, "server_utilization"
        )
        
        # Predict future values
        predicted_response_time = self._predict_future_value(
            context.historical_metrics, "basic_response_time", self.prediction_window
        )
        predicted_utilization = self._predict_future_value(
            context.historical_metrics, "server_utilization", self.prediction_window
        )
        
        confidence = 0.0
        should_adapt = False
        
        # Check if predicted response time will exceed threshold
        if predicted_response_time and predicted_response_time > 1000.0:
            confidence = max(confidence, 0.8)
            should_adapt = True
            self.logger.info(f"Predicted response time issue: {predicted_response_time:.1f}ms")
        
        # Check if predicted utilization will be problematic
        if predicted_utilization:
            if predicted_utilization > 0.85:
                confidence = max(confidence, 0.7)
                should_adapt = True
                self.logger.info(f"Predicted high utilization: {predicted_utilization:.2f}")
            elif predicted_utilization < 0.25:
                confidence = max(confidence, 0.6)
                should_adapt = True
                self.logger.info(f"Predicted low utilization: {predicted_utilization:.2f}")
        
        return should_adapt and confidence >= self.confidence_threshold, confidence
    
    async def plan_adaptation(self, context: AdaptationContext) -> List[AdaptationAction]:
        """Plan predictive adaptation actions."""
        actions = []
        
        if len(context.historical_metrics) < self.min_trend_points:
            return actions
        
        # Predict future metrics
        predicted_response_time = self._predict_future_value(
            context.historical_metrics, "basic_response_time", self.prediction_window
        )
        predicted_utilization = self._predict_future_value(
            context.historical_metrics, "server_utilization", self.prediction_window
        )
        
        current_metrics = context.current_metrics
        
        # Proactive scaling based on predicted utilization
        if predicted_utilization:
            if (predicted_utilization > 0.85 and 
                current_metrics.server_count < context.constraints.get("max_servers", 10)):
                
                actions.append(AdaptationAction(
                    action_type="ADD_SERVER",
                    parameters={"reason": "predicted_high_utilization"},
                    expected_impact={
                        "server_count": 1,
                        "predicted_utilization": predicted_utilization - 0.15
                    }
                ))
            
            elif (predicted_utilization < 0.25 and 
                  current_metrics.server_count > 1):
                
                actions.append(AdaptationAction(
                    action_type="REMOVE_SERVER",
                    parameters={"reason": "predicted_low_utilization"},
                    expected_impact={
                        "server_count": -1,
                        "predicted_utilization": predicted_utilization + 0.15
                    }
                ))
        
        # Proactive QoS adjustment based on predicted response time
        if (predicted_response_time and predicted_response_time > 1000.0 and
            current_metrics.dimmer > 0.2):
            
            new_dimmer = max(0.2, current_metrics.dimmer - 0.15)
            actions.append(AdaptationAction(
                action_type="SET_DIMMER",
                parameters={
                    "value": new_dimmer,
                    "reason": "predicted_response_time_issue"
                },
                expected_impact={
                    "dimmer": new_dimmer,
                    "predicted_response_time": predicted_response_time * 0.8
                }
            ))
        
        return actions
    
    def _analyze_trend(self, metrics: List[SwimMetrics], metric_name: str) -> Optional[float]:
        """Analyze trend in a specific metric."""
        values = []
        timestamps = []
        
        for metric in metrics[-self.min_trend_points:]:
            if hasattr(metric, metric_name):
                value = getattr(metric, metric_name)
                if value is not None:
                    values.append(value)
                    timestamps.append(metric.timestamp.timestamp())
        
        if len(values) < 3:
            return None
        
        # Simple linear regression for trend
        n = len(values)
        sum_x = sum(timestamps)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(timestamps, values))
        sum_x2 = sum(x * x for x in timestamps)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def _predict_future_value(self, metrics: List[SwimMetrics], 
                            metric_name: str, seconds_ahead: float) -> Optional[float]:
        """Predict future value of a metric."""
        if len(metrics) < 3:
            return None
        
        trend = self._analyze_trend(metrics, metric_name)
        if trend is None:
            return None
        
        # Get current value
        current_metric = metrics[-1]
        if not hasattr(current_metric, metric_name):
            return None
        
        current_value = getattr(current_metric, metric_name)
        if current_value is None:
            return None
        
        # Predict future value based on trend
        predicted_value = current_value + (trend * seconds_ahead)
        
        # Apply reasonable bounds
        if metric_name == "server_utilization":
            predicted_value = max(0.0, min(1.0, predicted_value))
        elif metric_name == "basic_response_time":
            predicted_value = max(0.0, predicted_value)
        
        return predicted_value


class SwimAdaptationStrategyFactory:
    """Factory for creating SWIM adaptation strategies."""
    
    @staticmethod
    def create_strategy(strategy_type: str, config: Dict[str, Any]) -> SwimAdaptationStrategy:
        """Create an adaptation strategy of the specified type."""
        strategy_type = strategy_type.lower()
        
        if strategy_type == "reactive":
            return ReactiveAdaptationStrategy(config)
        elif strategy_type == "predictive":
            return PredictiveAdaptationStrategy(config)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    @staticmethod
    def create_strategies_from_config(config: Dict[str, Any]) -> List[SwimAdaptationStrategy]:
        """Create all strategies from configuration."""
        strategies = []
        
        adaptation_config = config.get("adaptation", {})
        strategies_config = adaptation_config.get("strategies", {})
        
        for strategy_name, strategy_config in strategies_config.items():
            if strategy_config.get("enabled", True):
                try:
                    strategy = SwimAdaptationStrategyFactory.create_strategy(
                        strategy_name, strategy_config
                    )
                    strategies.append(strategy)
                except Exception as e:
                    logging.getLogger("SwimAdaptationStrategyFactory").error(
                        f"Failed to create strategy {strategy_name}: {e}"
                    )
        
        return strategies


class SwimAdaptationCoordinator:
    """Coordinates multiple adaptation strategies for SWIM."""
    
    def __init__(self, strategies: List[SwimAdaptationStrategy], config: Dict[str, Any]):
        self.strategies = strategies
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Coordination parameters
        self.max_concurrent_adaptations = config.get("max_concurrent_adaptations", 3)
        self.adaptation_cooldown = config.get("adaptation_cooldown", 60.0)
        self.last_adaptation_time = None
    
    async def should_adapt(self, context: AdaptationContext) -> Tuple[bool, float, List[str]]:
        """
        Check if any strategy recommends adaptation.
        
        Returns:
            Tuple of (should_adapt, max_confidence, recommending_strategies)
        """
        recommendations = []
        
        for strategy in self.strategies:
            if not strategy.enabled:
                continue
            
            try:
                should_adapt, confidence = await strategy.should_adapt(context)
                if should_adapt:
                    recommendations.append((strategy.name, confidence * strategy.weight))
            except Exception as e:
                self.logger.error(f"Strategy {strategy.name} failed: {e}")
        
        if not recommendations:
            return False, 0.0, []
        
        # Calculate weighted confidence
        total_weight = sum(conf for _, conf in recommendations)
        max_confidence = max(conf for _, conf in recommendations)
        recommending_strategies = [name for name, _ in recommendations]
        
        return True, max_confidence, recommending_strategies
    
    async def plan_adaptations(self, context: AdaptationContext) -> List[AdaptationAction]:
        """Plan adaptations using all enabled strategies."""
        all_actions = []
        
        for strategy in self.strategies:
            if not strategy.enabled:
                continue
            
            try:
                should_adapt, confidence = await strategy.should_adapt(context)
                if should_adapt:
                    actions = await strategy.plan_adaptation(context)
                    validated_actions = await strategy.validate_actions(actions, context)
                    
                    # Add strategy metadata to actions
                    for action in validated_actions:
                        action.metadata = action.metadata or {}
                        action.metadata.update({
                            "strategy": strategy.name,
                            "confidence": confidence,
                            "weight": strategy.weight
                        })
                    
                    all_actions.extend(validated_actions)
            
            except Exception as e:
                self.logger.error(f"Strategy {strategy.name} planning failed: {e}")
        
        # Deduplicate and prioritize actions
        return self._prioritize_actions(all_actions)
    
    def _prioritize_actions(self, actions: List[AdaptationAction]) -> List[AdaptationAction]:
        """Prioritize and deduplicate actions."""
        if not actions:
            return []
        
        # Group actions by type
        action_groups = {}
        for action in actions:
            action_type = action.action_type
            if action_type not in action_groups:
                action_groups[action_type] = []
            action_groups[action_type].append(action)
        
        # Select best action from each group
        prioritized_actions = []
        for action_type, group_actions in action_groups.items():
            # Sort by confidence * weight
            group_actions.sort(
                key=lambda a: a.metadata.get("confidence", 0) * a.metadata.get("weight", 1),
                reverse=True
            )
            prioritized_actions.append(group_actions[0])
        
        # Limit concurrent adaptations
        return prioritized_actions[:self.max_concurrent_adaptations]