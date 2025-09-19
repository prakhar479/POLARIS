"""
Adaptive Controller Implementation

Implements the Adaptive Controller with a MAPE-K (Monitor-Analyze-Plan-Execute with Knowledge) loop
and pluggable control strategies with comprehensive observability. This component is responsible for:
- Monitoring system telemetry
- Analyzing the need for adaptation
- Planning appropriate adaptation actions
- Executing the adaptation process
- Maintaining knowledge for future decisions
- Full observability integration (logging, metrics, tracing)

Key Components:
- MAPE-K loop implementation with tracing
- Pluggable control strategies (Reactive, Predictive, Learning)
- Telemetry processing pipeline with metrics
- Adaptation need assessment with logging
- Strategy selection and execution with observability
"""


from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from ..domain.models import AdaptationAction, SystemState, HealthStatus
from ..framework.events import TelemetryEvent, AdaptationEvent
from ..infrastructure.di import Injectable
from ..infrastructure.observability import (
    observe_polaris_component, trace_adaptation_flow, get_logger,
    get_metrics_collector, get_tracer
)
from ..digital_twin.world_model import PolarisWorldModel
from ..digital_twin.world_model import PredictionResult, SimulationResult
from ..digital_twin.knowledge_base import PolarisKnowledgeBase
from ..framework.events import PolarisEventBus


class AdaptationNeed:
    """Represents an identified need for adaptation."""
    
    def __init__(
        self, 
        system_id: str, 
        is_needed: bool, 
        reason: str, 
        urgency: float = 0.5
    ):
        self.system_id = system_id
        self.is_needed = is_needed
        self.reason = reason
        self.urgency = urgency  # 0.0 to 1.0


class ControlStrategy(ABC):
    """Abstract base class for control strategies in the POLARIS adaptive control system.
    
    Control strategies implement different approaches to generating adaptation actions
    based on the current system state and adaptation needs. Concrete strategies should
    implement the generate_actions() method to provide specific adaptation logic.
    
    The strategy pattern allows for dynamic selection and composition of different
    control approaches at runtime based on the system's current context.
    """
    
    @abstractmethod
    async def generate_actions(
        self, 
        system_id: str, 
        current_state: Dict[str, Any],
        adaptation_need: AdaptationNeed
    ) -> List[AdaptationAction]:
        """Generate adaptation actions for the given situation."""
        pass


class ReactiveControlStrategy(ControlStrategy):
    """Reactive control strategy that responds to current system conditions.
    
    This strategy implements simple rule-based adaptation by analyzing the current
    system metrics and applying predefined rules to generate adaptation actions.
    It does not consider historical data or predict future states.
    
    Key Features:
    - Immediate response to threshold violations
    - Simple, deterministic behavior
    - Low computational overhead
    - No dependency on historical data or complex models
    
    Example Rules:
    - Scale out when CPU usage exceeds 85%
    - Adjust QoS settings when latency is too high
    """
    
    async def generate_actions(
        self, 
        system_id: str, 
        current_state: Dict[str, Any],
        adaptation_need: AdaptationNeed
    ) -> List[AdaptationAction]:
        actions: List[AdaptationAction] = []
        metrics = current_state.get("metrics", {})
        # Simple heuristic rules
        cpu = _get_numeric_metric(metrics, ["cpu", "cpu_usage", "cpu_percent"]) or 0.0
        latency = _get_numeric_metric(metrics, ["latency", "p95_latency", "response_time"]) or 0.0

        # Scale out if CPU high
        if cpu >= 0.85:
            actions.append(
                AdaptationAction(
                    action_id="",
                    action_type="scale_out",
                    target_system=system_id,
                    parameters={"scale_factor": 2},
                    priority=3,
                )
            )

        # Increase resources or degrade non-critical features if latency high
        if latency >= 0.85:
            actions.append(
                AdaptationAction(
                    action_id="",
                    action_type="tune_qos",
                    target_system=system_id,
                    parameters={"qos_level": "high"},
                    priority=2,
                )
            )

        return actions


class PredictiveControlStrategy(ControlStrategy):
    """Predictive control strategy that anticipates future system needs.
    
    This strategy uses the world model to simulate potential future states and
    evaluates different adaptation actions before selecting the most promising ones.
    It aims to prevent issues before they occur by taking proactive measures.
    
    Key Features:
    - Proactive adaptation based on predictions
    - Simulation of different adaptation scenarios
    - Consideration of system dynamics and constraints
    - Optimization of adaptation outcomes
    
    Dependencies:
    - Requires a properly configured world model
    - Benefits from historical data for accurate predictions
    """
    
    def __init__(self, world_model: Optional[PolarisWorldModel] = None):
        self._world_model = world_model

    async def generate_actions(
        self, 
        system_id: str, 
        current_state: Dict[str, Any],
        adaptation_need: AdaptationNeed
    ) -> List[AdaptationAction]:
        if not self._world_model:
            return []
        # Ask the world model to simulate impact of common actions, pick best
        candidates = [
            {"action_type": "scale_out", "parameters": {"scale_factor": 2}},
            {"action_type": "tune_qos", "parameters": {"qos_level": "high"}},
        ]
        best: Optional[Dict[str, Any]] = None
        best_score = float("-inf")
        for cand in candidates:
            sim: SimulationResult = await self._world_model.simulate_adaptation_impact(system_id, cand)
            score = _score_simulation_outcomes(sim.outcomes)
            if score > best_score:
                best_score = score
                best = cand
        if best:
            return [
                AdaptationAction(
                    action_id="",
                    action_type=best.get("action_type", "unknown"),
                    target_system=system_id,
                    parameters=best.get("parameters", {}),
                    priority=3,
                )
            ]
        return []


class LearningControlStrategy(ControlStrategy):
    """Learning-based control strategy that leverages historical patterns and experiences.
    
    This strategy uses the knowledge base to identify similar past situations and
    retrieves the most effective adaptation actions that were taken in those cases.
    It improves over time as more adaptation experiences are accumulated.
    
    Key Features:
    - Leverages historical adaptation experiences
    - Improves with more data over time
    - Can handle complex, non-linear relationships
    - Adapts to changing system behavior
    
    Dependencies:
    - Requires a populated knowledge base
    - Benefits from a diverse set of historical patterns
    - May require initial training period
    """
    
    def __init__(self, knowledge_base: Optional[PolarisKnowledgeBase] = None):
        self._kb = knowledge_base

    async def generate_actions(
        self, 
        system_id: str, 
        current_state: Dict[str, Any],
        adaptation_need: AdaptationNeed
    ) -> List[AdaptationAction]:
        if not self._kb:
            return []
        # Use current conditions to search similar learned patterns
        conditions = _conditions_from_state(current_state)
        patterns = await self._kb.get_similar_patterns(conditions, similarity_threshold=0.6)
        actions: List[AdaptationAction] = []
        for p in patterns[:2]:  # take top 2
            action_type = p.outcomes.get("action_type", "")
            params = p.outcomes.get("parameters", {})
            if action_type:
                actions.append(
                    AdaptationAction(
                        action_id="",
                        action_type=action_type,
                        target_system=system_id,
                        parameters=params,
                        priority=2,
                    )
                )
        return actions


@observe_polaris_component("adaptive_controller", auto_trace=True, auto_metrics=True, log_method_calls=True)
class PolarisAdaptiveController:
    """POLARIS Adaptive Controller implementing the MAPE-K (Monitor-Analyze-Plan-Execute with Knowledge) loop.
    
    This controller orchestrates the adaptation process by:
    1. Processing incoming telemetry (Monitor)
    2. Assessing the need for adaptation (Analyze)
    3. Selecting and executing appropriate control strategies (Plan)
    4. Triggering adaptation actions (Execute)
    5. Maintaining and utilizing system knowledge (Knowledge)
    
    The controller supports multiple control strategies that can be composed or selected
    based on the current system state and adaptation needs:
    - ReactiveControlStrategy: Responds to current system state
    - PredictiveControlStrategy: Anticipates future needs using world model
    - LearningControlStrategy: Leverages historical patterns and experiences
    
    Features:
    - Pluggable strategy architecture
    - Asynchronous processing
    - Comprehensive telemetry handling
    - Extensible design for custom strategies
    - Full observability integration (logging, metrics, tracing)
    """
    
    def __init__(
        self,
        control_strategies: Optional[List[ControlStrategy]] = None,
        world_model: Optional[PolarisWorldModel] = None,
        knowledge_base: Optional[PolarisKnowledgeBase] = None,
        event_bus: Optional[PolarisEventBus] = None,
    ):
        # Observability integration
        self.logger = get_logger("polaris.adaptive_controller")
        self.metrics = get_metrics_collector()
        self.tracer = get_tracer()
        
        # Dependencies
        self._world_model: Optional[PolarisWorldModel] = world_model
        self._kb: Optional[PolarisKnowledgeBase] = knowledge_base
        self._event_bus: Optional[PolarisEventBus] = event_bus

        # Strategies (wire dependencies where applicable)
        default_strategies: List[ControlStrategy] = [
            ReactiveControlStrategy(),
            PredictiveControlStrategy(world_model=world_model),
            LearningControlStrategy(knowledge_base=knowledge_base),
        ]
        self._control_strategies = control_strategies or default_strategies
        
        self.logger.info("Adaptive controller initialized", extra={
            "strategies_count": len(self._control_strategies),
            "world_model_available": self._world_model is not None,
            "knowledge_base_available": self._kb is not None,
            "event_bus_available": self._event_bus is not None
        })
    
    @trace_adaptation_flow("telemetry_processing")
    async def process_telemetry(self, telemetry: TelemetryEvent) -> None:
        """Process incoming telemetry and trigger adaptations if needed."""
        system_id = telemetry.system_state.system_id
        
        self.logger.debug("Processing telemetry", extra={
            "system_id": system_id,
            "health_status": telemetry.system_state.health_status.value,
            "metrics_count": len(telemetry.system_state.metrics)
        })
        
        # Monitor: update world model and optionally store in KB
        if self._world_model:
            try:
                with self.tracer.trace_operation("world_model_update"):
                    await self._world_model.update_system_state(telemetry)
                self.logger.debug("World model updated", extra={"system_id": system_id})
            except Exception as e:
                self.logger.warning("Failed to update world model", extra={
                    "system_id": system_id,
                    "error": str(e)
                }, exc_info=e)
        
        if self._kb:
            try:
                with self.tracer.trace_operation("knowledge_base_store"):
                    await self._kb.store_telemetry(telemetry)
                self.logger.debug("Telemetry stored in knowledge base", extra={"system_id": system_id})
            except Exception as e:
                self.logger.warning("Failed to store telemetry in knowledge base", extra={
                    "system_id": system_id,
                    "error": str(e)
                }, exc_info=e)

        # Analyze: assess need
        with self.tracer.trace_operation("adaptation_need_assessment"):
            adaptation_need = await self.assess_adaptation_need(telemetry)
        
        if not adaptation_need.is_needed:
            self.logger.debug("No adaptation needed", extra={
                "system_id": system_id,
                "reason": adaptation_need.reason
            })
            return
        
        self.logger.info("Adaptation needed", extra={
            "system_id": system_id,
            "reason": adaptation_need.reason,
            "urgency": adaptation_need.urgency
        })
        
        # Update metrics
        self.metrics.increment_adaptations_triggered(
            system_id, 
            "telemetry_driven", 
            adaptation_need.reason
        )

        # Plan & Execute trigger
        await self.trigger_adaptation_process(adaptation_need, telemetry.system_state)
    
    async def assess_adaptation_need(self, telemetry: TelemetryEvent) -> AdaptationNeed:
        """Assess if adaptation is needed based on telemetry."""
        state: SystemState = telemetry.system_state
        # Simple health-based rule
        if state.health_status in (HealthStatus.WARNING, HealthStatus.UNHEALTHY, HealthStatus.CRITICAL):
            urgency = {HealthStatus.WARNING: 0.5, HealthStatus.UNHEALTHY: 0.8, HealthStatus.CRITICAL: 1.0}.get(
                state.health_status, 0.6
            )
            return AdaptationNeed(system_id=state.system_id, is_needed=True, reason=str(state.health_status.value), urgency=urgency)

        # Metric thresholds (normalized 0..1 preferred; if >1, treat >85 as high)
        metrics = state.metrics or {}
        cpu = _get_numeric_metric(metrics, ["cpu", "cpu_usage", "cpu_percent"]) or 0.0
        latency = _get_numeric_metric(metrics, ["latency", "p95_latency", "response_time"]) or 0.0
        cpu_high = cpu >= 0.9 or cpu >= 85.0
        latency_high = latency >= 0.9 or latency >= 85.0
        if cpu_high or latency_high:
            reason = "High CPU" if cpu_high else "High Latency"
            urgency = 0.8 if cpu_high and latency_high else 0.6
            return AdaptationNeed(system_id=state.system_id, is_needed=True, reason=reason, urgency=urgency)

        return AdaptationNeed(system_id=state.system_id, is_needed=False, reason="No issues detected")
    
    async def trigger_adaptation_process(self, adaptation_need: AdaptationNeed, current_state_obj: Optional[SystemState] = None) -> None:
        """Trigger the adaptation process for an identified need.

        current_state_obj: optionally provide the live SystemState that triggered analysis.
        If not provided, the controller will fall back to knowledge base snapshot.
        """
        if not self._event_bus:
            return
        system_id = adaptation_need.system_id
        # Build planning context
        if current_state_obj is not None:
            current_state = {
                "metrics": current_state_obj.metrics,
                "health_status": current_state_obj.health_status.value,
                "timestamp": current_state_obj.timestamp,
            }
        else:
            current_state = await self._get_current_state_snapshot(system_id)
        strategy = await self.select_control_strategy(system_id, {"adaptation_need": adaptation_need.__dict__, "current_state": current_state})
        actions: List[AdaptationAction] = []
        if strategy:
            try:
                actions = await strategy.generate_actions(system_id, current_state, adaptation_need)
            except Exception:
                actions = []
        # Persist planned actions to KB for adaptation history
        if actions and self._kb:
            try:
                await self._kb.store_adaptation_actions(actions)
            except Exception:
                pass
        severity = _severity_from_urgency(adaptation_need.urgency)
        event = AdaptationEvent(system_id=system_id, reason=adaptation_need.reason, suggested_actions=actions, severity=severity)
        await self._event_bus.publish_adaptation_needed(event)
    
    async def select_control_strategy(
        self, 
        system_id: str, 
        context: Dict[str, Any]
    ) -> ControlStrategy:
        """Select the appropriate control strategy for the situation."""
        # Simple selection logic:
        # - If world model exists and can provide a meaningful prediction -> Predictive
        # - Else if KB exists and has similar patterns -> Learning
        # - Else -> Reactive
        need: AdaptationNeed = context.get("adaptation_need") if isinstance(context.get("adaptation_need"), AdaptationNeed) else None  # type: ignore[assignment]
        if self._world_model:
            try:
                pred: PredictionResult = await self._world_model.predict_system_behavior(system_id, time_horizon=60)
                if pred and pred.probability >= 0.6:
                    for s in self._control_strategies:
                        if isinstance(s, PredictiveControlStrategy):
                            return s
            except Exception:
                pass
        if self._kb:
            try:
                current_state = context.get("current_state", {})
                conditions = _conditions_from_state(current_state)
                sims = await self._kb.get_similar_patterns(conditions, similarity_threshold=0.7)
                if sims:
                    for s in self._control_strategies:
                        if isinstance(s, LearningControlStrategy):
                            return s
            except Exception:
                pass
        # Default reactive
        for s in self._control_strategies:
            if isinstance(s, ReactiveControlStrategy):
                return s
        return self._control_strategies[0] if self._control_strategies else None

    async def _get_current_state_snapshot(self, system_id: str) -> Dict[str, Any]:
        """Fetch current state data for planning; prefer KB if available, else rely on latest world model memory-less context.

        Returns a dict with at least a "metrics" map if known.
        """
        if self._kb:
            try:
                state = await self._kb.get_current_state(system_id)
                if state:
                    return {"metrics": state.metrics, "health_status": state.health_status.value, "timestamp": state.timestamp}
            except Exception:
                pass
        # Fallback minimal context
        return {"metrics": {}}


# ---------- Helpers ----------

def _get_numeric_metric(metrics: Dict[str, Any], names: List[str]) -> Optional[float]:
    for n in names:
        mv = metrics.get(n)
        if mv is None:
            continue
        try:
            # MetricValue or raw
            if hasattr(mv, "value"):
                return float(getattr(mv, "value"))
            return float(mv)
        except (TypeError, ValueError):
            continue
    return None


def _conditions_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    metrics = state.get("metrics", {}) if isinstance(state, dict) else {}
    cond: Dict[str, Any] = {}
    for k, v in metrics.items():
        try:
            val = float(getattr(v, "value", v))
            cond[k] = "high" if val >= 0.85 or val >= 85.0 else ("low" if val <= 0.15 else "normal")
        except Exception:
            # categorize non-numeric as present
            cond[k] = "present"
    return cond


def _score_simulation_outcomes(outcomes: Dict[str, Any]) -> float:
    score = 0.0
    for k, v in outcomes.items():
        try:
            val = float(v)
        except (TypeError, ValueError):
            continue
        # Lower latency/cpu is better; invert common load metrics
        if "latency" in k.lower() or "cpu" in k.lower():
            score += (1.0 - min(1.0, max(0.0, val)))
        else:
            score += val
    return score


def _severity_from_urgency(urgency: float) -> str:
    if urgency >= 0.9:
        return "critical"
    if urgency >= 0.75:
        return "high"
    if urgency >= 0.5:
        return "normal"
    return "low"