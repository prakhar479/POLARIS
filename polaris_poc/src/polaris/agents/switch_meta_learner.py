"""
Switch-Specific Meta Learner Agent for POLARIS Framework.

This module implements a specialized meta-learner for the SWITCH YOLO system,
focusing on utility function optimization, model switching pattern analysis,
and adaptive threshold tuning for maximum performance.
"""

import asyncio
import logging
import statistics
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .example_meta_learner import ExampleMetaLearnerAgent
from .meta_learner_agent import (
    TriggerType,
    ParameterType,
    MetaLearningContext,
    ParameterUpdate,
    CalibrationRequest,
    CalibrationResult,
    MetaLearningInsights,
    MetaLearningError,
)


class SwitchMetaLearnerAgent(ExampleMetaLearnerAgent):
    """
    SWITCH-specific meta learner optimized for YOLO model switching.
    
    Focuses on:
    - Utility function weight optimization
    - Model switching threshold tuning
    - Response time vs confidence tradeoff learning
    - CPU resource management optimization
    - Spiral detection and prevention
    """
    
    def __init__(
        self,
        agent_id: str = "switch_meta_learner",
        config_path: str = "config/switch_optimized_config.yaml",
        nats_url: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the SWITCH meta learner agent."""
        super().__init__(agent_id, config_path, nats_url, logger, config)
        
        # SWITCH-specific configuration
        self.switch_config = self.config.get("meta_learner", {})
        
        # HIGH FREQUENCY SETTINGS
        self.analysis_window_hours = self.switch_config.get("learning", {}).get("analysis_window_hours", 2.0)
        self.calibration_frequency_minutes = self.switch_config.get("learning", {}).get("calibration_frequency_minutes", 15)
        self.periodic_interval_minutes = self.switch_config.get("triggers", {}).get("periodic", {}).get("interval_minutes", 5)
        
        # SWITCH-specific thresholds
        self.utility_drop_threshold = self.switch_config.get("triggers", {}).get("performance_driven", {}).get("utility_drop_threshold", 0.1)
        self.response_time_spike_threshold = self.switch_config.get("triggers", {}).get("performance_driven", {}).get("response_time_spike_threshold", 2.0)
        
        # Learning state for SWITCH
        self.utility_history: List[float] = []
        self.model_switch_history: List[Dict[str, Any]] = []
        self.performance_baselines: Dict[str, float] = {}
        self.learned_patterns: Dict[str, Any] = {}
        
        self.logger.info(
            "SWITCH Meta Learner initialized with high frequency settings",
            extra={
                "periodic_interval_minutes": self.periodic_interval_minutes,
                "calibration_frequency_minutes": self.calibration_frequency_minutes,
                "analysis_window_hours": self.analysis_window_hours
            }
        )
    
    async def analyze_adaptation_patterns(
        self, context: MetaLearningContext
    ) -> MetaLearningInsights:
        """Analyze SWITCH-specific adaptation patterns."""
        try:
            self.logger.info(f"Analyzing SWITCH adaptation patterns: {context.trigger_type}")
            
            # Query SWITCH-specific data
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=context.time_window_hours)
            
            # Analyze utility trends
            utility_patterns = await self._analyze_utility_patterns(start_time, end_time)
            
            # Analyze model switching effectiveness
            switching_patterns = await self._analyze_model_switching_patterns(start_time, end_time)
            
            # Analyze response time vs confidence tradeoffs
            tradeoff_patterns = await self._analyze_rt_confidence_tradeoffs(start_time, end_time)
            
            # Analyze CPU resource utilization
            resource_patterns = await self._analyze_resource_patterns(start_time, end_time)
            
            # Generate SWITCH-specific recommendations
            recommendations = self._generate_switch_recommendations(
                utility_patterns, switching_patterns, tradeoff_patterns, resource_patterns
            )
            
            insights = MetaLearningInsights(
                analysis_window={"start": start_time, "end": end_time},
                adaptation_patterns=[
                    utility_patterns,
                    switching_patterns, 
                    tradeoff_patterns,
                    resource_patterns
                ],
                performance_trends={
                    "utility_trend": utility_patterns.get("trend", 0.0),
                    "switching_frequency": switching_patterns.get("frequency", 0.0),
                    "rt_confidence_correlation": tradeoff_patterns.get("correlation", 0.0),
                    "cpu_efficiency": resource_patterns.get("efficiency", 0.0)
                },
                recommendations=recommendations,
                confidence_overall=self._calculate_switch_confidence(
                    utility_patterns, switching_patterns, tradeoff_patterns
                ),
            )
            
            self.logger.info(
                f"SWITCH pattern analysis completed",
                extra={
                    "confidence": insights.confidence_overall,
                    "recommendations_count": len(recommendations),
                    "utility_trend": utility_patterns.get("trend", 0.0)
                }
            )
            
            return insights
            
        except Exception as e:
            self.logger.error(f"SWITCH pattern analysis failed: {e}")
            raise MetaLearningError(f"Failed to analyze SWITCH patterns: {e}")
    
    async def propose_parameter_updates(
        self, insights: MetaLearningInsights, context: MetaLearningContext
    ) -> List[ParameterUpdate]:
        """Propose SWITCH-specific parameter updates based on insights."""
        updates = []
        
        try:
            # Extract performance trends
            utility_trend = insights.performance_trends.get("utility_trend", 0.0)
            switching_frequency = insights.performance_trends.get("switching_frequency", 0.0)
            rt_confidence_correlation = insights.performance_trends.get("rt_confidence_correlation", 0.0)
            
            # 1. Utility function weight optimization
            if abs(rt_confidence_correlation) > 0.3:  # Strong correlation detected
                weight_updates = self._propose_utility_weight_updates(rt_confidence_correlation, utility_trend)
                updates.extend(weight_updates)
            
            # 2. Switching threshold optimization
            if switching_frequency > 15 or switching_frequency < 2:  # Too frequent or too rare
                threshold_updates = self._propose_threshold_updates(switching_frequency, utility_trend)
                updates.extend(threshold_updates)
            
            # 3. Controller strategy optimization
            if utility_trend < -0.1:  # Declining utility
                strategy_updates = self._propose_strategy_updates(insights)
                updates.extend(strategy_updates)
            
            # 4. Emergency spiral prevention
            current_utility = self._get_current_utility_estimate()
            if current_utility < 0.2:  # Potential spiral
                spiral_updates = self._propose_spiral_prevention_updates()
                updates.extend(spiral_updates)
            
            self.logger.info(
                f"Proposed {len(updates)} SWITCH parameter updates",
                extra={
                    "utility_trend": utility_trend,
                    "switching_frequency": switching_frequency,
                    "updates_count": len(updates)
                }
            )
            
            return updates
            
        except Exception as e:
            self.logger.error(f"Failed to propose SWITCH parameter updates: {e}")
            return []
    
    async def _analyze_utility_patterns(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Analyze utility function patterns and trends."""
        # In real implementation, query knowledge base for utility data
        # For now, simulate analysis
        
        return {
            "pattern_type": "utility_analysis",
            "trend": 0.05,  # Slight positive trend
            "volatility": 0.15,  # Moderate volatility
            "average_utility": 0.65,
            "min_utility": 0.2,
            "max_utility": 0.9,
            "spiral_events": 2,  # Number of utility spirals detected
            "recovery_time_avg": 45.0  # Average recovery time in seconds
        }
    
    async def _analyze_model_switching_patterns(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Analyze model switching effectiveness and patterns."""
        return {
            "pattern_type": "model_switching",
            "frequency": 8.5,  # Switches per hour
            "success_rate": 0.85,  # Successful switches
            "most_common_switch": "yolov5m -> yolov5s",
            "least_effective_switch": "yolov5l -> yolov5x",
            "average_improvement": 0.12,  # Average utility improvement
            "thrashing_events": 1,  # Number of rapid back-and-forth switches
            "stabilization_time_avg": 3.2  # Average time to stabilize after switch
        }
    
    async def _analyze_rt_confidence_tradeoffs(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Analyze response time vs confidence tradeoff effectiveness."""
        return {
            "pattern_type": "rt_confidence_tradeoff",
            "correlation": -0.65,  # Strong negative correlation (expected)
            "optimal_balance_point": {"rt": 0.25, "confidence": 0.78},
            "current_balance_point": {"rt": 0.30, "confidence": 0.75},
            "efficiency_score": 0.72,  # How well we're balancing the tradeoff
            "weight_recommendation": {"rt_weight": 0.65, "conf_weight": 0.35}
        }
    
    async def _analyze_resource_patterns(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Analyze CPU and resource utilization patterns."""
        return {
            "pattern_type": "resource_utilization",
            "efficiency": 0.78,  # Resource efficiency score
            "cpu_avg": 72.5,
            "cpu_spikes": 3,  # Number of CPU spikes > 90%
            "underutilization_periods": 15.2,  # Minutes of CPU < 40%
            "optimal_cpu_range": {"min": 60.0, "max": 80.0},
            "model_cpu_correlation": 0.89  # Strong correlation between model complexity and CPU
        }
    
    def _generate_switch_recommendations(
        self, 
        utility_patterns: Dict[str, Any],
        switching_patterns: Dict[str, Any], 
        tradeoff_patterns: Dict[str, Any],
        resource_patterns: Dict[str, Any]
    ) -> List[str]:
        """Generate SWITCH-specific recommendations."""
        recommendations = []
        
        # Utility-based recommendations
        if utility_patterns.get("spiral_events", 0) > 1:
            recommendations.append("Implement more aggressive spiral prevention - detected multiple utility spirals")
        
        if utility_patterns.get("average_utility", 0) < 0.6:
            recommendations.append("Overall utility below target - consider adjusting utility function weights")
        
        # Switching pattern recommendations
        if switching_patterns.get("frequency", 0) > 12:
            recommendations.append("High switching frequency detected - increase minimum switch interval")
        
        if switching_patterns.get("thrashing_events", 0) > 0:
            recommendations.append("Model thrashing detected - implement hysteresis in switching logic")
        
        # Tradeoff optimization recommendations
        rt_conf_correlation = tradeoff_patterns.get("correlation", 0)
        if abs(rt_conf_correlation) < 0.5:
            recommendations.append("Weak RT-confidence correlation - investigate model performance issues")
        
        # Resource optimization recommendations
        if resource_patterns.get("cpu_spikes", 0) > 2:
            recommendations.append("Frequent CPU spikes - lower CPU thresholds for proactive model switching")
        
        if resource_patterns.get("underutilization_periods", 0) > 10:
            recommendations.append("Significant CPU underutilization - consider more aggressive model upgrades")
        
        return recommendations
    
    def _propose_utility_weight_updates(self, correlation: float, utility_trend: float) -> List[ParameterUpdate]:
        """Propose utility function weight updates."""
        updates = []
        
        # If strong negative correlation and declining utility, favor response time more
        if correlation < -0.6 and utility_trend < 0:
            updates.append(ParameterUpdate(
                parameter_type=ParameterType.UTILITY_WEIGHTS,
                parameter_path="utility_function.response_time_weight",
                old_value=0.6,
                new_value=0.7,
                confidence=0.8,
                reasoning="Strong RT-confidence correlation with declining utility suggests favoring response time",
                expected_impact="Improved utility through faster response times",
                risk_assessment="Low risk - within acceptable weight bounds"
            ))
            
            updates.append(ParameterUpdate(
                parameter_type=ParameterType.UTILITY_WEIGHTS,
                parameter_path="utility_function.confidence_weight", 
                old_value=0.4,
                new_value=0.3,
                confidence=0.8,
                reasoning="Complementary adjustment to response time weight increase",
                expected_impact="Balanced utility function with RT emphasis",
                risk_assessment="Low risk - maintains minimum confidence weight"
            ))
        
        return updates
    
    def _propose_threshold_updates(self, switching_frequency: float, utility_trend: float) -> List[ParameterUpdate]:
        """Propose switching threshold updates."""
        updates = []
        
        # If switching too frequently, increase thresholds
        if switching_frequency > 15:
            updates.append(ParameterUpdate(
                parameter_type=ParameterType.THRESHOLD_VALUES,
                parameter_path="switching_thresholds.utility_low_threshold",
                old_value=0.3,
                new_value=0.25,
                confidence=0.75,
                reasoning="High switching frequency suggests thresholds too sensitive",
                expected_impact="Reduced switching frequency, more stable operation",
                risk_assessment="Medium risk - may delay necessary adaptations"
            ))
        
        # If not switching enough and utility declining, lower thresholds
        elif switching_frequency < 3 and utility_trend < -0.05:
            updates.append(ParameterUpdate(
                parameter_type=ParameterType.THRESHOLD_VALUES,
                parameter_path="switching_thresholds.utility_low_threshold",
                old_value=0.3,
                new_value=0.35,
                confidence=0.7,
                reasoning="Low switching frequency with declining utility suggests more proactive switching needed",
                expected_impact="More responsive adaptation to utility changes",
                risk_assessment="Low risk - increases system responsiveness"
            ))
        
        return updates
    
    def _propose_strategy_updates(self, insights: MetaLearningInsights) -> List[ParameterUpdate]:
        """Propose controller strategy updates."""
        updates = []
        
        # If utility declining, make system more reactive
        utility_trend = insights.performance_trends.get("utility_trend", 0.0)
        if utility_trend < -0.1:
            updates.append(ParameterUpdate(
                parameter_type=ParameterType.COORDINATION_STRATEGIES,
                parameter_path="controller_strategy.optimization_interval",
                old_value=15,
                new_value=10,
                confidence=0.7,
                reasoning="Declining utility trend requires more frequent optimization cycles",
                expected_impact="More responsive adaptation to system changes",
                risk_assessment="Low risk - increases adaptation frequency"
            ))
        
        return updates
    
    def _propose_spiral_prevention_updates(self) -> List[ParameterUpdate]:
        """Propose emergency spiral prevention updates."""
        updates = []
        
        # Reduce penalty factors to prevent death spirals
        updates.append(ParameterUpdate(
            parameter_type=ParameterType.UTILITY_WEIGHTS,
            parameter_path="utility_function.penalty_factors.response_time_penalty",
            old_value=2.0,
            new_value=1.5,
            confidence=0.9,
            reasoning="Potential utility spiral detected - reduce penalty factors",
            expected_impact="Prevent exponential utility degradation",
            risk_assessment="Low risk - emergency spiral prevention"
        ))
        
        return updates
    
    def _calculate_switch_confidence(
        self, 
        utility_patterns: Dict[str, Any],
        switching_patterns: Dict[str, Any], 
        tradeoff_patterns: Dict[str, Any]
    ) -> float:
        """Calculate confidence in SWITCH-specific analysis."""
        confidence_factors = []
        
        # Utility pattern confidence
        if utility_patterns.get("spiral_events", 0) == 0:
            confidence_factors.append(0.9)  # High confidence if no spirals
        else:
            confidence_factors.append(0.6)  # Lower confidence with spirals
        
        # Switching pattern confidence
        success_rate = switching_patterns.get("success_rate", 0.5)
        confidence_factors.append(success_rate)
        
        # Tradeoff analysis confidence
        correlation_strength = abs(tradeoff_patterns.get("correlation", 0))
        confidence_factors.append(min(1.0, correlation_strength + 0.3))
        
        return statistics.mean(confidence_factors)
    
    def _get_current_utility_estimate(self) -> float:
        """Get current utility estimate from recent history."""
        if self.utility_history:
            return self.utility_history[-1]
        return 0.5  # Default estimate
    
    async def handle_trigger(
        self, trigger_type: TriggerType, trigger_data: Dict[str, Any]
    ) -> bool:
        """Handle SWITCH-specific meta-learning triggers."""
        try:
            self.logger.info(f"Handling SWITCH meta-learning trigger: {trigger_type}")
            
            # Create context with SWITCH-specific focus
            context = MetaLearningContext(
                trigger_type=trigger_type,
                trigger_source="switch_meta_learner",
                time_window_hours=self.analysis_window_hours,
                focus_areas=[
                    "utility_optimization",
                    "model_switching_patterns", 
                    "response_time_trends",
                    "cpu_resource_management"
                ],
                constraints=trigger_data.get("constraints", {}),
                metadata=trigger_data
            )
            
            # Execute SWITCH-specific meta-learning cycle
            if trigger_type == TriggerType.PERIODIC:
                return await self._handle_switch_periodic_trigger(context)
            elif trigger_type == TriggerType.PERFORMANCE_DRIVEN:
                return await self._handle_switch_performance_trigger(context)
            elif trigger_type == TriggerType.EVENT_DRIVEN:
                return await self._handle_switch_event_trigger(context)
            else:
                return await self._handle_switch_periodic_trigger(context)
                
        except Exception as e:
            self.logger.error(f"SWITCH meta-learning trigger failed: {e}")
            return False
    
    async def _handle_switch_periodic_trigger(self, context: MetaLearningContext) -> bool:
        """Handle periodic SWITCH meta-learning cycle."""
        try:
            # Full SWITCH analysis and optimization cycle
            insights = await self.analyze_adaptation_patterns(context)
            proposed_updates = await self.propose_parameter_updates(insights, context)
            
            if proposed_updates:
                validated_updates = await self.validate_updates(proposed_updates)
                if validated_updates:
                    await self.apply_updates(validated_updates)
                    self.logger.info(f"Applied {len(validated_updates)} SWITCH parameter updates")
            
            # Calibrate world model for SWITCH metrics
            if self.last_calibration_time is None or \
               (datetime.now(timezone.utc) - self.last_calibration_time).total_seconds() > (self.calibration_frequency_minutes * 60):
                
                calibration_request = CalibrationRequest(
                    target_metrics=["utility", "image_processing_time", "confidence", "cpu_usage"],
                    validation_window_hours=0.5  # Short validation for high frequency
                )
                await self.calibrate_world_model(calibration_request)
                self.last_calibration_time = datetime.now(timezone.utc)
            
            return True
            
        except Exception as e:
            self.logger.error(f"SWITCH periodic trigger failed: {e}")
            return False
    
    async def _handle_switch_performance_trigger(self, context: MetaLearningContext) -> bool:
        """Handle performance-driven SWITCH trigger."""
        # Focus on immediate performance issues
        context.focus_areas = ["utility_optimization", "response_time_trends"]
        return await self._handle_switch_periodic_trigger(context)
    
    async def _handle_switch_event_trigger(self, context: MetaLearningContext) -> bool:
        """Handle event-driven SWITCH trigger."""
        # Focus on model switching patterns and spiral prevention
        context.focus_areas = ["model_switching_patterns", "utility_spiral_prevention"]
        return await self._handle_switch_periodic_trigger(context)