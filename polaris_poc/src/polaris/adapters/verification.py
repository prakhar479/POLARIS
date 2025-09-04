"""
Verification Adapter for POLARIS Framework.

This adapter validates control actions before execution, ensuring they comply
with safety constraints, organizational policies, and system invariants.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from nats.aio.msg import Msg

from polaris.adapters.core import InternalAdapter
from polaris.models.actions import ControlAction
from polaris.models.world_model import SimulationRequest, QueryRequest
from polaris.common.utils import safe_eval


class VerificationLevel(Enum):
    """Verification thoroughness levels."""
    BASIC = "basic"           # Simple constraint checking
    POLICY = "policy"         # Policy compliance checking  
    FORMAL = "formal"         # Model checking and formal verification
    COMPREHENSIVE = "comprehensive"  # All verification methods


class ConstraintType(Enum):
    """Types of constraints that can be violated."""
    SAFETY = "safety"         # Safety-critical constraints
    RESOURCE = "resource"     # Resource availability/limits
    POLICY = "policy"         # Organizational policies
    TEMPORAL = "temporal"     # Time-based constraints
    DEPENDENCY = "dependency" # Inter-system dependencies


class ConstraintViolation:
    """Represents a constraint violation found during verification."""
    
    def __init__(
        self,
        constraint_id: str,
        constraint_type: ConstraintType,
        severity: str,
        description: str,
        suggested_fix: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.constraint_id = constraint_id
        self.constraint_type = constraint_type
        self.severity = severity  # "critical", "high", "medium", "low"
        self.description = description
        self.suggested_fix = suggested_fix
        self.metadata = metadata or {}
        self.timestamp = datetime.now(timezone.utc).isoformat()


class VerificationRequest:
    """Request for action verification."""
    
    def __init__(
        self,
        request_id: str,
        action: ControlAction,
        context: Optional[Dict[str, Any]] = None,
        verification_level: VerificationLevel = VerificationLevel.BASIC,
        timeout_sec: float = 30.0,
        requester: str = "unknown"
    ):
        self.request_id = request_id
        self.action = action
        self.context = context or {}
        self.verification_level = verification_level
        self.timeout_sec = timeout_sec
        self.requester = requester
        self.timestamp = datetime.now(timezone.utc).isoformat()


class VerificationResult:
    """Result of action verification."""
    
    def __init__(
        self,
        request_id: str,
        action_id: str,
        approved: bool,
        confidence: float,
        violations: Optional[List[ConstraintViolation]] = None,
        recommendations: Optional[List[str]] = None,
        verification_time_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
        transformed_action: Optional[Dict[str, Any]] = None
    ):
        self.request_id = request_id
        self.action_id = action_id
        self.approved = approved
        self.confidence = confidence
        self.violations = violations or []
        self.recommendations = recommendations or []
        self.verification_time_ms = verification_time_ms
        self.metadata = metadata or {}
        self.transformed_action = transformed_action
        self.timestamp = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "request_id": self.request_id,
            "action_id": self.action_id,
            "approved": self.approved,
            "confidence": self.confidence,
            "violations": [
                {
                    "constraint_id": v.constraint_id,
                    "constraint_type": v.constraint_type.value,
                    "severity": v.severity,
                    "description": v.description,
                    "suggested_fix": v.suggested_fix,
                    "metadata": v.metadata,
                    "timestamp": v.timestamp
                }
                for v in self.violations
            ],
            "recommendations": self.recommendations,
            "verification_time_ms": self.verification_time_ms,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
        
        # Include transformed action if available
        if self.transformed_action is not None:
            result["transformed_action"] = self.transformed_action
            
        return result


class VerificationAdapter(InternalAdapter):
    """
    Verification adapter that validates control actions before execution.
    
    This adapter provides comprehensive action verification including:
    - Safety constraint checking
    - Policy compliance validation
    - Resource availability verification
    - Digital twin-based predictive analysis
    - Formal verification (when configured)
    
    The adapter uses plugin configuration to determine:
    - Which constraints to enforce
    - Verification policies and rules
    - Integration with external verification tools
    - Performance and timeout settings
    """
    
    def __init__(
        self,
        polaris_config_path: str,
        plugin_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the verification adapter.
        
        Args:
            polaris_config_path: Path to POLARIS framework configuration
            plugin_dir: Optional directory containing plugin configuration
            logger: Logger instance (created if not provided)
        """
        super().__init__(
            polaris_config_path, 
            plugin_dir, 
            logger, 
            component_name="verification-adapter"
        )
        
        # Get verification configuration (try plugin first, then framework)
        self.verification_config = self.get_component_config("verification")
        
        # If no verification config found, use built-in defaults
        if not self.verification_config:
            self.logger.info("No verification config found, using built-in defaults")
            self.verification_config = self._get_default_verification_config()
        
        # Extract configuration values with defaults
        self.constraints = self.verification_config.get("constraints", [])
        self.policies = self.verification_config.get("policies", [])
        self.verification_settings = self.verification_config.get("settings", {})
        
        # Verification settings with framework defaults
        framework_verification = self.framework_config.get("verification", {})
        self.default_timeout = self.verification_settings.get(
            "default_timeout_sec", 
            framework_verification.get("default_timeout_sec", 30.0)
        )
        self.max_concurrent = self.verification_settings.get(
            "max_concurrent", 
            framework_verification.get("max_concurrent_verifications", 5)
        )
        self.enable_digital_twin = self.verification_settings.get("enable_digital_twin", True)
        self.enable_formal_verification = self.verification_settings.get("enable_formal_verification", False)
        
        # NATS subjects from framework configuration
        self.request_subject = framework_verification.get(
            "input_subject", "polaris.verification.requests"
        )
        self.result_subject = framework_verification.get(
            "output_subject", "polaris.verification.results"
        )
        self.policy_subject = framework_verification.get(
            "policy_subject", "polaris.verification.policies"
        )
        self.metrics_subject = framework_verification.get(
            "metrics_subject", "polaris.verification.metrics"
        )
        
        # Execution subject for direct publishing of approved actions
        execution_config = self.framework_config.get("execution", {})
        self.execution_subject = execution_config.get(
            "action_subject", "polaris.execution.actions"
        )
        
        # Runtime state
        queue_size = self.verification_settings.get("queue_size", 1000)
        self.verification_queue: asyncio.Queue[VerificationRequest] = asyncio.Queue(
            maxsize=queue_size if queue_size > 0 else 0
        )
        self.worker_task: Optional[asyncio.Task] = None
        self._verification_semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Performance metrics
        self._metrics = {
            "requests_processed": 0,
            "requests_approved": 0,
            "requests_rejected": 0,
            "average_verification_time_ms": 0.0,
            "constraint_violations": 0,
            "policy_violations": 0
        }
        
        # Build constraint and policy mappings
        self.constraint_configs = {c["id"]: c for c in self.constraints}
        self.policy_configs = {p["id"]: p for p in self.policies}
        
        # System name for metrics (from plugin config if available)
        self.system_name = self.plugin_config.get("system_name", "polaris_framework")
        
        self.logger.info(
            "Verification adapter initialized with optimized flow",
            extra={
                "system_name": self.system_name,
                "constraints_count": len(self.constraints),
                "policies_count": len(self.policies),
                "digital_twin_enabled": self.enable_digital_twin,
                "formal_verification_enabled": self.enable_formal_verification,
                "execution_subject": self.execution_subject,
                "direct_execution_enabled": True,
                "plugin_dir": str(self.plugin_dir) if self.plugin_dir else None
            }
        )

    async def _publish_metric(self, metric_name: str, payload: Dict[str, Any]):
        """Publish verification metrics to NATS."""
        try:
            metric_data = {
                "metric": metric_name,
                "timestamp": time.time(),
                "system": self.system_name,
                "component": "verification",
                **payload
            }
            
            await self.nats_client.publish_json(
                self.metrics_subject,
                metric_data
            )
            
        except Exception as e:
            self.logger.warning(
                "Metric publish failed",
                extra={
                    "metric": metric_name,
                    "error": str(e)
                }
            )

    async def verify_action(self, request: VerificationRequest) -> VerificationResult:
        """Verify a control action against all configured constraints and policies.
        
        Args:
            request: Verification request containing action and context
            
        Returns:
            Verification result with approval status and details
        """
        start_time = time.perf_counter()
        
        ctx = {
            "request_id": request.request_id,
            "action_id": request.action.action_id,
            "action_type": request.action.action_type,
            "verification_level": request.verification_level.value
        }
        
        self.logger.info(
            "Starting action verification",
            extra=ctx
        )
        
        violations = []
        recommendations = []
        confidence = 1.0
        
        try:
            # Phase 1: Basic constraint checking
            constraint_violations = await self._check_constraints(request.action, request.context)
            violations.extend(constraint_violations)
            
            # Phase 2: Policy compliance checking
            if request.verification_level in [VerificationLevel.POLICY, VerificationLevel.COMPREHENSIVE]:
                policy_violations = await self._check_policies(request.action, request.context)
                violations.extend(policy_violations)
            
            # Phase 3: Digital twin simulation (if enabled and requested)
            if (self.enable_digital_twin and 
                request.verification_level in [VerificationLevel.FORMAL, VerificationLevel.COMPREHENSIVE]):
                
                dt_violations, dt_recommendations = await self._check_digital_twin(request.action, request.context)
                violations.extend(dt_violations)
                recommendations.extend(dt_recommendations)
            
            # Phase 4: Formal verification (if enabled and requested)
            if (self.enable_formal_verification and 
                request.verification_level == VerificationLevel.COMPREHENSIVE):
                
                formal_violations = await self._check_formal_constraints(request.action, request.context)
                violations.extend(formal_violations)
            
            # Determine approval based on violations
            critical_violations = [v for v in violations if v.severity == "critical"]
            high_violations = [v for v in violations if v.severity == "high"]
            
            approved = len(critical_violations) == 0
            
            # Adjust confidence based on violations
            if critical_violations:
                confidence = 0.0
            elif high_violations:
                confidence = max(0.3, 1.0 - (len(high_violations) * 0.2))
            elif violations:
                confidence = max(0.7, 1.0 - (len(violations) * 0.1))
            
            # Generate recommendations for rejected actions
            if not approved:
                recommendations.extend(self._generate_fix_recommendations(violations))
            
            end_time = time.perf_counter()
            verification_time_ms = (end_time - start_time) * 1000
            
            result = VerificationResult(
                request_id=request.request_id,
                action_id=request.action.action_id,
                approved=approved,
                confidence=confidence,
                violations=violations,
                recommendations=recommendations,
                verification_time_ms=verification_time_ms,
                metadata={
                    "verification_level": request.verification_level.value,
                    "constraint_checks": len(self.constraints),
                    "policy_checks": len(self.policies),
                    "digital_twin_used": self.enable_digital_twin,
                    "formal_verification_used": self.enable_formal_verification
                },
                transformed_action=request.action.to_dict()
            )
            
            # Update metrics
            self._metrics["requests_processed"] += 1
            if approved:
                self._metrics["requests_approved"] += 1
            else:
                self._metrics["requests_rejected"] += 1
            
            self._metrics["constraint_violations"] += len([v for v in violations if v.constraint_type == ConstraintType.SAFETY])
            self._metrics["policy_violations"] += len([v for v in violations if v.constraint_type == ConstraintType.POLICY])
            
            # Update average verification time
            current_avg = self._metrics["average_verification_time_ms"]
            total_requests = self._metrics["requests_processed"]
            self._metrics["average_verification_time_ms"] = (
                (current_avg * (total_requests - 1) + verification_time_ms) / total_requests
            )
            
            # Publish metrics
            await self._publish_metric("verification_completed", {
                "request_id": request.request_id,
                "action_type": request.action.action_type,
                "approved": approved,
                "confidence": confidence,
                "verification_time_ms": verification_time_ms,
                "violations_count": len(violations),
                "critical_violations": len(critical_violations)
            })
            
            self.logger.info(
                "Action verification completed",
                extra={
                    **ctx,
                    "approved": approved,
                    "confidence": confidence,
                    "violations_count": len(violations),
                    "verification_time_ms": round(verification_time_ms, 3)
                }
            )
            
            return result
            
        except Exception as e:
            end_time = time.perf_counter()
            verification_time_ms = (end_time - start_time) * 1000
            
            error_msg = str(e)
            
            result = VerificationResult(
                request_id=request.request_id,
                action_id=request.action.action_id,
                approved=False,
                confidence=0.0,
                violations=[
                    ConstraintViolation(
                        constraint_id="verification_error",
                        constraint_type=ConstraintType.SAFETY,
                        severity="critical",
                        description=f"Verification failed: {error_msg}",
                        suggested_fix="Review action parameters and try again"
                    )
                ],
                recommendations=["Action rejected due to verification error"],
                verification_time_ms=verification_time_ms,
                metadata={"error": error_msg},
                transformed_action=request.action.to_dict() if hasattr(request, 'action') else None
            )
            
            self.logger.error(
                "Action verification failed",
                extra={
                    **ctx,
                    "error": error_msg,
                    "verification_time_ms": round(verification_time_ms, 3)
                }
            )
            
            return result

    async def _check_constraints(self, action: ControlAction, context: Dict[str, Any]) -> List[ConstraintViolation]:
        """Check action against configured safety and resource constraints."""
        violations = []
        
        for constraint_config in self.constraints:
            try:
                violation = await self._evaluate_constraint(constraint_config, action, context)
                if violation:
                    violations.append(violation)
            except Exception as e:
                self.logger.warning(
                    "Constraint evaluation failed",
                    extra={
                        "constraint_id": constraint_config.get("id"),
                        "error": str(e)
                    }
                )
                # Add a violation for the failed constraint check
                violations.append(
                    ConstraintViolation(
                        constraint_id=constraint_config.get("id", "unknown"),
                        constraint_type=ConstraintType.SAFETY,
                        severity="high",
                        description=f"Constraint evaluation failed: {str(e)}",
                        suggested_fix="Review constraint configuration"
                    )
                )
        
        return violations

    async def _evaluate_constraint(
        self, 
        constraint_config: Dict[str, Any], 
        action: ControlAction, 
        context: Dict[str, Any]
    ) -> Optional[ConstraintViolation]:
        """Evaluate a single constraint against an action."""
        constraint_id = constraint_config["id"]
        constraint_type = ConstraintType(constraint_config.get("type", "safety"))
        condition = constraint_config["condition"]
        severity = constraint_config.get("severity", "high")
        description = constraint_config.get("description", f"Constraint {constraint_id} violated")
        
        # Build evaluation context
        eval_context = {
            **context,
            "action": action.to_dict(),
            "action_type": action.action_type,
            "params": action.params,
            **action.params  # Allow direct access to action parameters
        }
        
        # Query current system state if needed
        if "current_state" in condition:
            try:
                # Use the context as current state since verification relies on provided context
                current_state = context.copy()  # Use context as current state
                eval_context["current_state"] = current_state
            except Exception as e:
                self.logger.warning(f"Failed to get current system state: {e}")
                eval_context["current_state"] = {}
        
        # Evaluate constraint condition
        try:
            result = safe_eval(condition, eval_context)
            
            # If condition evaluates to False, constraint is violated
            if not result:
                return ConstraintViolation(
                    constraint_id=constraint_id,
                    constraint_type=constraint_type,
                    severity=severity,
                    description=description,
                    suggested_fix=constraint_config.get("suggested_fix"),
                    metadata={
                        "condition": condition,
                        "evaluation_context": {k: str(v) for k, v in eval_context.items()}
                    }
                )
        
        except Exception as e:
            # Constraint evaluation error - treat as violation
            return ConstraintViolation(
                constraint_id=constraint_id,
                constraint_type=constraint_type,
                severity="high",
                description=f"Constraint evaluation error: {str(e)}",
                suggested_fix="Review constraint condition syntax",
                metadata={"condition": condition, "error": str(e)}
            )
        
        return None

    async def _check_policies(self, action: ControlAction, context: Dict[str, Any]) -> List[ConstraintViolation]:
        """Check action against organizational policies."""
        violations = []
        
        for policy_config in self.policies:
            try:
                violation = await self._evaluate_policy(policy_config, action, context)
                if violation:
                    violations.append(violation)
            except Exception as e:
                self.logger.warning(
                    "Policy evaluation failed",
                    extra={
                        "policy_id": policy_config.get("id"),
                        "error": str(e)
                    }
                )
        
        return violations

    async def _evaluate_policy(
        self, 
        policy_config: Dict[str, Any], 
        action: ControlAction, 
        context: Dict[str, Any]
    ) -> Optional[ConstraintViolation]:
        """Evaluate a single policy against an action."""
        policy_id = policy_config["id"]
        rules = policy_config.get("rules", [])
        
        for rule in rules:
            condition = rule["condition"]
            severity = rule.get("severity", "medium")
            description = rule.get("description", f"Policy {policy_id} violated")
            
            # Build evaluation context
            eval_context = {
                **context,
                "action": action.to_dict(),
                "action_type": action.action_type,
                "params": action.params,
                **action.params
            }
            
            try:
                result = safe_eval(condition, eval_context)
                
                if not result:
                    return ConstraintViolation(
                        constraint_id=f"{policy_id}_{rule.get('id', 'unknown')}",
                        constraint_type=ConstraintType.POLICY,
                        severity=severity,
                        description=description,
                        suggested_fix=rule.get("suggested_fix"),
                        metadata={
                            "policy_id": policy_id,
                            "rule_condition": condition
                        }
                    )
            
            except Exception as e:
                return ConstraintViolation(
                    constraint_id=f"{policy_id}_error",
                    constraint_type=ConstraintType.POLICY,
                    severity="medium",
                    description=f"Policy evaluation error: {str(e)}",
                    suggested_fix="Review policy rule syntax"
                )
        
        return None

    async def _check_digital_twin(
        self, 
        action: ControlAction, 
        context: Dict[str, Any]
    ) -> tuple[List[ConstraintViolation], List[str]]:
        """Use digital twin to simulate action effects and check for violations."""
        violations = []
        recommendations = []
        
        if not self.enable_digital_twin:
            return violations, recommendations
        
        try:
            # Create simulation request
            simulation_request = SimulationRequest(
                simulation_id=f"verification_{uuid.uuid4()}",
                simulation_type="what_if",
                actions=[action.to_dict()],
                horizon_minutes=30,  # Configurable simulation horizon
                parameters={"verification_mode": True}
            )
            
            # TODO: Integrate with Digital Twin gRPC service
            # For now, implement basic simulation logic
            
            # Simulate resource impact
            if action.action_type == "ADD_SERVER":
                # Check if adding server would exceed resource limits
                current_servers = context.get("current_servers", 0)
                max_servers = context.get("max_servers", 10)
                
                if current_servers >= max_servers:
                    violations.append(
                        ConstraintViolation(
                            constraint_id="max_servers_exceeded",
                            constraint_type=ConstraintType.RESOURCE,
                            severity="critical",
                            description="Adding server would exceed maximum server limit",
                            suggested_fix="Remove existing servers before adding new ones"
                        )
                    )
            
            elif action.action_type == "REMOVE_SERVER":
                # Check if removing server would cause service degradation
                current_servers = context.get("current_servers", 1)
                min_servers = context.get("min_servers", 1)
                
                if current_servers <= min_servers:
                    violations.append(
                        ConstraintViolation(
                            constraint_id="min_servers_violated",
                            constraint_type=ConstraintType.SAFETY,
                            severity="critical",
                            description="Removing server would violate minimum server requirement",
                            suggested_fix="Ensure minimum server count is maintained"
                        )
                    )
            
            # Add general recommendations
            if not violations:
                recommendations.append("Digital twin simulation indicates action is safe to execute")
            
        except Exception as e:
            self.logger.warning(f"Digital twin verification failed: {e}")
            violations.append(
                ConstraintViolation(
                    constraint_id="digital_twin_error",
                    constraint_type=ConstraintType.SAFETY,
                    severity="medium",
                    description=f"Digital twin verification failed: {str(e)}",
                    suggested_fix="Proceed with caution or retry verification"
                )
            )
        
        return violations, recommendations

    async def _check_formal_constraints(self, action: ControlAction, context: Dict[str, Any]) -> List[ConstraintViolation]:
        """Apply formal verification methods (placeholder for future implementation)."""
        violations = []
        
        # TODO: Implement formal verification integration
        # This could include:
        # - Temporal logic verification
        # - Model checking with tools like PRISM or TLA+
        # - Contract-based verification
        
        self.logger.debug("Formal verification not yet implemented")
        
        return violations

    def _generate_fix_recommendations(self, violations: List[ConstraintViolation]) -> List[str]:
        """Generate actionable recommendations based on violations."""
        recommendations = []
        
        critical_violations = [v for v in violations if v.severity == "critical"]
        
        if critical_violations:
            recommendations.append("Action contains critical violations and must be modified before execution")
            
            for violation in critical_violations:
                if violation.suggested_fix:
                    recommendations.append(f"Fix {violation.constraint_id}: {violation.suggested_fix}")
        
        high_violations = [v for v in violations if v.severity == "high"]
        if high_violations:
            recommendations.append("Consider addressing high-severity violations to improve action safety")
        
        return recommendations

    async def _get_current_system_state(self) -> Dict[str, Any]:
        """Get current system state from knowledge base or context.
        
        Since verification is an internal component, it doesn't directly
        query managed systems. Instead, it relies on context provided
        in verification requests or queries the knowledge base.
        """
        try:
            # For now, return empty state - context should be provided in requests
            # In the future, this could query the knowledge base for current state
            state = {}
            
            self.logger.debug("Current system state requested - using context from verification request")
            
            return state
            
        except Exception as e:
            self.logger.warning(f"Failed to get current system state: {e}")
            return {}

    async def _request_handler(self, msg: Msg):
        """Handle incoming verification requests from NATS."""
        try:
            # Parse the verification request using proper JSON parsing
            request_data = msg.data.decode()
            request_dict = json.loads(request_data)
            
            # Create VerificationRequest object
            action_dict = request_dict["action"]
            
            # Transform the action dict to match ControlAction model expectations
            # Handle the case where action has 'type' and 'decision' fields
            if "type" in action_dict and "decision" in action_dict:
                # Map the type/decision combination to standard action types
                decision = action_dict["decision"].lower()
                action_type_mapping = {
                    "scale_up": "SCALE_UP",
                    "scale_down": "SCALE_DOWN",
                    "add_server": "ADD_SERVER",
                    "remove_server": "REMOVE_SERVER",
                    "restart": "RESTART",
                    "set_dimmer": "SET_DIMMER",
                    "adjust_qos": "ADJUST_QOS"
                }
                
                # Use mapped action type or fallback to decision
                action_type = action_type_mapping.get(decision, decision.upper())
                
                transformed_action_dict = {
                    "action_type": action_type,
                    "params": {
                        "original_type": action_dict["type"],
                        "decision": action_dict["decision"]
                    }
                }
            elif "action_type" in action_dict:
                # Already in correct format
                transformed_action_dict = action_dict
            else:
                # Fallback: use the entire action dict as params
                transformed_action_dict = {
                    "action_type": "CUSTOM",
                    "params": action_dict
                }
            
            action = ControlAction(**transformed_action_dict)
            
            request = VerificationRequest(
                request_id=request_dict["request_id"],
                action=action,
                context=request_dict.get("context", {}),
                verification_level=VerificationLevel(request_dict.get("verification_level", "basic")),
                timeout_sec=request_dict.get("timeout_sec", self.default_timeout),
                requester=request_dict.get("requester", "unknown")
            )
            
            # Enqueue for processing
            await self.verification_queue.put(request)
            
        except Exception as e:
            self.logger.info(msg)
            self.logger.error(
                "Failed to parse verification request",
                extra={
                    "error": str(e),
                    "raw_data": msg.data[:256].decode(errors="replace")
                }
            )

    async def _handle_verification_result(self, result: VerificationResult):
        """Handle verification result with optimized flow."""
        try:
            if result.approved:
                # Approved actions: publish directly to execution
                await self._publish_to_execution(result)
            else:
                # Rejected actions: publish to results for logging/monitoring
                await self._publish_verification_result(result)
                
        except Exception as e:
            self.logger.error(
                "Failed to handle verification result",
                extra={
                    "request_id": result.request_id,
                    "action_id": result.action_id,
                    "approved": result.approved,
                    "error": str(e)
                }
            )

    async def _publish_to_execution(self, result: VerificationResult):
        """Publish approved action directly to execution adapter."""
        try:
            # Use transformed action if available, otherwise use original
            action_to_execute = result.transformed_action
            
            await self.nats_client.publish_json(
                self.execution_subject,
                action_to_execute
            )
            
            # Also publish verification result for monitoring/metrics
            await self._publish_verification_result(result)
            
            self.logger.info(
                "Approved action sent directly to execution",
                extra={
                    "request_id": result.request_id,
                    "action_id": result.action_id,
                    "action_type": action_to_execute.get("action_type"),
                    "confidence": result.confidence
                }
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to publish to execution",
                extra={
                    "request_id": result.request_id,
                    "action_id": result.action_id,
                    "error": str(e)
                }
            )
            # Fallback: publish verification result for manual handling
            await self._publish_verification_result(result)

    async def _publish_verification_result(self, result: VerificationResult):
        """Publish verification result to NATS for monitoring/logging."""
        try:
            await self.nats_client.publish_json(
                self.result_subject,
                result.to_dict()
            )
            
            self.logger.info(
                "Verification result published",
                extra={
                    "request_id": result.request_id,
                    "action_id": result.action_id,
                    "approved": result.approved,
                    "confidence": result.confidence
                }
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to publish verification result",
                extra={
                    "request_id": result.request_id,
                    "action_id": result.action_id,
                    "error": str(e)
                }
            )

    async def _worker(self):
        """Main worker loop that processes verification requests from the queue."""
        self.logger.info("Verification worker started")
        
        try:
            while self.running:
                # Get next request from queue
                request: VerificationRequest = await self.verification_queue.get()
                
                ctx = {
                    "request_id": request.request_id,
                    "action_id": request.action.action_id,
                    "action_type": request.action.action_type
                }
                
                # Execute with concurrency control and timeout
                async with self._verification_semaphore:
                    try:
                        # Apply timeout to verification
                        result = await asyncio.wait_for(
                            self.verify_action(request),
                            timeout=request.timeout_sec
                        )
                        
                        # Handle the result based on approval status
                        await self._handle_verification_result(result)
                        
                    except asyncio.TimeoutError:
                        # Handle verification timeout
                        timeout_result = VerificationResult(
                            request_id=request.request_id,
                            action_id=request.action.action_id,
                            approved=False,
                            confidence=0.0,
                            violations=[
                                ConstraintViolation(
                                    constraint_id="verification_timeout",
                                    constraint_type=ConstraintType.TEMPORAL,
                                    severity="high",
                                    description=f"Verification timed out after {request.timeout_sec} seconds",
                                    suggested_fix="Reduce verification complexity or increase timeout"
                                )
                            ],
                            recommendations=["Action rejected due to verification timeout"],
                            verification_time_ms=request.timeout_sec * 1000,
                            transformed_action=request.action.to_dict()
                        )
                        
                        await self._handle_verification_result(timeout_result)
                        
                        self.logger.warning(
                            "Verification request timed out",
                            extra={**ctx, "timeout_sec": request.timeout_sec}
                        )
                    
                    except Exception as e:
                        # Handle processing errors
                        error_result = VerificationResult(
                            request_id=request.request_id,
                            action_id=request.action.action_id,
                            approved=False,
                            confidence=0.0,
                            violations=[
                                ConstraintViolation(
                                    constraint_id="processing_error",
                                    constraint_type=ConstraintType.SAFETY,
                                    severity="critical",
                                    description=f"Verification processing failed: {str(e)}",
                                    suggested_fix="Review action and retry verification"
                                )
                            ],
                            recommendations=["Action rejected due to processing error"],
                            verification_time_ms=0.0,
                            metadata={"error": str(e)},
                            transformed_action=request.action.to_dict()
                        )
                        
                        try:
                            await self._handle_verification_result(error_result)
                        except Exception as publish_error:
                            self.logger.error(f"Failed to handle error result: {publish_error}")
                        
                        self.logger.exception(
                            "Unexpected error in verification processing",
                            extra={**ctx, "error": str(e)}
                        )
                    
                    finally:
                        # Mark task as done
                        self.verification_queue.task_done()
        
        except asyncio.CancelledError:
            self.logger.info("Verification worker cancelled")
        finally:
            self.logger.info("Verification worker stopped")



    async def _start_processing(self) -> None:
        """Start verification-specific processing."""
        # Subscribe to verification request messages
        await self.nats_client.subscribe(
            self.request_subject,
            self._request_handler
        )
        
        # Start worker task
        self.worker_task = asyncio.create_task(self._worker())
        self._tasks.append(self.worker_task)
        
        self.logger.info(
            "Verification processing started",
            extra={
                "request_subject": self.request_subject,
                "max_concurrent": self.max_concurrent,
                "default_timeout": self.default_timeout
            }
        )

    async def _stop_processing(self) -> None:
        """Stop verification-specific processing."""
        # Drain the verification queue with timeout
        try:
            await asyncio.wait_for(
                self.verification_queue.join(),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            self.logger.warning(
                "Verification queue drain timeout",
                extra={"remaining": self.verification_queue.qsize()}
            )
        
        # Cancel worker
        if self.worker_task and not self.worker_task.done():
            self.worker_task.cancel()
        
        self.logger.info("Verification processing stopped")
    
    def _get_default_verification_config(self) -> Dict[str, Any]:
        """Get built-in default verification configuration."""
        return {
            "constraints": [
                {
                    "id": "valid_action_type",
                    "type": "safety",
                    "condition": "action_type in ['ADD_SERVER', 'REMOVE_SERVER', 'SET_DIMMER', 'ADJUST_QOS', 'RESTART', 'SCALE_UP', 'SCALE_DOWN']",
                    "severity": "critical",
                    "description": "Action type must be recognized by the system",
                    "suggested_fix": "Use a valid action type from the supported list"
                },
                {
                    "id": "required_parameters",
                    "type": "safety", 
                    "condition": "params is not None",
                    "severity": "high",
                    "description": "Action must include required parameters",
                    "suggested_fix": "Ensure all required parameters are provided"
                }
            ],
            "policies": [
                {
                    "id": "framework_policy",
                    "name": "POLARIS Framework Policy",
                    "description": "Basic framework-level policies",
                    "rules": [
                        {
                            "id": "action_source_required",
                            "condition": "action.get('source') is not None and action.get('source') != ''",
                            "severity": "medium",
                            "description": "Actions should specify their source component",
                            "suggested_fix": "Add source field to action metadata"
                        }
                    ]
                }
            ],
            "settings": {
                "default_timeout_sec": 30,
                "max_concurrent": 5,
                "queue_size": 1000,
                "enable_digital_twin": False,
                "enable_formal_verification": False
            }
        }