"""
Execution Adapter Implementation

This module provides the implementation of the execution adapter, which is responsible
for executing adaptation actions on managed systems. The execution adapter is implemented
as a Chain of Responsibility pipeline that executes different stages of execution in order.
Each stage is responsible for a specific aspect of the execution pipeline, such as
validating preconditions, executing the action, performing post-condition checks, and so on.

The execution adapter publishes execution results via the event bus using the `PolarisEventBus`
interface. The execution results are published as `ExecutionResultEvent` objects.

"""


import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from ...domain.models import AdaptationAction, ExecutionResult, ExecutionStatus, SystemState
from ...domain.interfaces import ManagedSystemConnector
from ...framework.events import PolarisEventBus, ExecutionResultEvent, EventMetadata
from ...framework.plugin_management import PolarisPluginRegistry, ManagedSystemConnectorFactory
from ..base_adapter import (
    PolarisAdapter,
    AdapterConfiguration,
    AdapterValidationError,
)

from .execution_stages import ( 
    ExecutionStage, ValidationStage, PreConditionCheckStage, 
    ActionExecutionStage, PostExecutionVerificationStage
)

logger = logging.getLogger(__name__)

class ActionExecutionPipeline:
    """
    Pipeline for executing adaptation actions using Chain of Responsibility.
    
    Executes ordered stages, handling errors and composing an ExecutionResult.
    """
    
    def __init__(self, stages: List[ExecutionStage], event_bus: Optional[PolarisEventBus] = None, stage_timeouts: Optional[Dict[str, float]] = None):
        self.stages = stages
        self.event_bus = event_bus
        self.stage_timeouts = stage_timeouts or {}
    
    async def execute(self, action: AdaptationAction, initial_context: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """Execute an action through the pipeline."""
        start_time = datetime.utcnow()
        context: Dict[str, Any] = {
            "action": action,
            "start_time": start_time,
            "stage_results": {},
            "result_data": {},
        }
        if isinstance(initial_context, dict):
            # Merge initial context (e.g., connector) without overwriting core keys
            for k, v in initial_context.items():
                if k not in context:
                    context[k] = v
        
        try:
            for stage in self.stages:
                try:
                    # Determine per-stage timeout based on stage type
                    stage_key = None
                    if isinstance(stage, ValidationStage):
                        stage_key = "validation"
                    elif isinstance(stage, PreConditionCheckStage):
                        stage_key = "pre_condition"
                    elif isinstance(stage, ActionExecutionStage):
                        stage_key = "action_execution"
                    elif isinstance(stage, PostExecutionVerificationStage):
                        stage_key = "post_verification"

                    st_timeout = self.stage_timeouts.get(stage_key, None)
                    if st_timeout is not None:
                        context = await asyncio.wait_for(stage.execute(action, context), timeout=st_timeout)
                    else:
                        context = await stage.execute(action, context)
                except Exception as e:
                    logger.error(f"Stage {stage.__class__.__name__} failed: {e}")
                    err_result = ExecutionResult(
                        action_id=action.action_id,
                        status=ExecutionStatus.FAILED,
                        result_data=context.get("result_data", {}),
                        error_message=str(e),
                        execution_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000),
                    )
                    context["execution_result"] = err_result
                    break
            
            # If no stage produced a result, assume success with empty data
            if "execution_result" not in context:
                context["execution_result"] = ExecutionResult(
                    action_id=action.action_id,
                    status=ExecutionStatus.SUCCESS,
                    result_data=context.get("result_data", {}),
                    execution_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000),
                )
            else:
                # Ensure execution_time is set
                res: ExecutionResult = context["execution_result"]
                if res.execution_time_ms is None:
                    context["execution_result"] = ExecutionResult(
                        action_id=res.action_id,
                        status=res.status,
                        result_data=res.result_data,
                        error_message=res.error_message,
                        execution_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000),
                    )
            
            return context["execution_result"]
        except Exception as e:
            logger.error(f"Pipeline execution error: {e}")
            return ExecutionResult(
                action_id=action.action_id,
                status=ExecutionStatus.FAILED,
                result_data=context.get("result_data", {}),
                error_message=str(e),
                execution_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000),
            )


class ExecutionAdapter(PolarisAdapter):
    """
    Execution adapter for executing adaptation actions on managed systems.
    
    Implements a Chain of Responsibility pipeline for action execution and
    publishes execution results via the event bus.
    """
    
    def __init__(
        self,
        configuration: AdapterConfiguration,
        event_bus: Optional[PolarisEventBus] = None,
        plugin_registry: Optional[PolarisPluginRegistry] = None,
    ):
        super().__init__(configuration, event_bus)
        self._execution_pipeline: Optional[ActionExecutionPipeline] = None
        self.plugin_registry = plugin_registry
        self.connector_factory: Optional[ManagedSystemConnectorFactory] = None
        self._execution_lock = asyncio.Lock()
    
    async def _validate_configuration(self) -> None:
        """Validate execution adapter configuration."""
        errors = self.configuration.validate()
        validation_errors: List[str] = []
        if errors:
            validation_errors.extend(errors)
        
        cfg = self.configuration.config
        stages = cfg.get("pipeline_stages") if isinstance(cfg, dict) else None
        if not isinstance(stages, list) or not stages:
            validation_errors.append("config.pipeline_stages must be a non-empty list")
        
        # Ensure required four stages exist (order can be configured)
        required = {"validation", "pre_condition", "action_execution", "post_verification"}
        found = set([s.get("type") for s in stages]) if isinstance(stages, list) else set()
        missing = required - found
        if missing:
            validation_errors.append(f"Missing required pipeline stages: {sorted(list(missing))}")
        
        # Plugin registry is required to create connectors
        if not self.plugin_registry:
            validation_errors.append("plugin_registry must be provided to ExecutionAdapter")
        
        # Managed systems mapping is required to resolve connectors
        ms = cfg.get("managed_systems") if isinstance(cfg, dict) else None
        if not isinstance(ms, list) or not ms:
            validation_errors.append("config.managed_systems must be a non-empty list")
        else:
            for i, entry in enumerate(ms):
                if not isinstance(entry, dict):
                    validation_errors.append(f"managed_systems[{i}] must be a dict")
                    continue
                if not entry.get("system_id"):
                    validation_errors.append(f"managed_systems[{i}].system_id is required")
                if not entry.get("connector_type"):
                    validation_errors.append(f"managed_systems[{i}].connector_type is required")
                if not isinstance(entry.get("config", {}), dict):
                    validation_errors.append(f"managed_systems[{i}].config must be a dict")
        
        if validation_errors:
            raise AdapterValidationError(
                "ExecutionAdapter configuration validation failed",
                self.adapter_id,
                validation_errors,
            )
    
    async def _initialize_resources(self) -> None:
        """Initialize execution adapter resources."""
        # Initialize connector factory
        self.connector_factory = ManagedSystemConnectorFactory(self.plugin_registry)
        
        # Build pipeline from configuration
        cfg = self.configuration.config
        stage_cfgs: List[Dict[str, Any]] = cfg.get("pipeline_stages", [])
        stage_timeouts: Dict[str, int] = cfg.get("stage_timeouts", {}) if isinstance(cfg, dict) else {}
        verification_rules: Dict[str, Dict[str, Any]] = cfg.get("verification_rules", {}) if isinstance(cfg, dict) else {}
        pre_rules: Dict[str, Dict[str, Any]] = cfg.get("precondition_rules", {}) if isinstance(cfg, dict) else {}
        
        stages: List[ExecutionStage] = []
        for sc in stage_cfgs:
            stype = sc.get("type")
            if stype == "validation":
                stages.append(ValidationStage())
            elif stype == "pre_condition":
                stages.append(PreConditionCheckStage(rules=pre_rules))
            elif stype == "action_execution":
                stages.append(ActionExecutionStage(timeout_seconds=stage_timeouts.get("action_execution")))
            elif stype == "post_verification":
                stages.append(PostExecutionVerificationStage(rules=verification_rules))
            else:
                logger.warning(f"Unknown stage type in config: {stype}")
        
        self._execution_pipeline = ActionExecutionPipeline(
            stages=stages,
            event_bus=self.event_bus,
            stage_timeouts=stage_timeouts,
        )
        logger.info(f"Initialized ExecutionAdapter pipeline with {len(stages)} stages")
    
    async def _start_processing(self) -> None:
        """Start action execution processing."""
        logger.info(f"ExecutionAdapter {self.adapter_id} processing started")
    
    async def _stop_processing(self) -> None:
        """Stop action execution processing."""
        logger.info(f"ExecutionAdapter {self.adapter_id} processing stopped")
    
    async def _cleanup_resources(self) -> None:
        """Clean up execution adapter resources."""
        self._execution_pipeline = None
        self.connector_factory = None
        logger.info(f"ExecutionAdapter {self.adapter_id} resources cleaned up")

    async def execute_action(self, action: AdaptationAction) -> ExecutionResult:
        """
        Execute an adaptation action through the configured pipeline and publish result event.
        """
        if not self._execution_pipeline:
            raise RuntimeError("Execution pipeline not initialized")
        
        # Resolve and attach connector into context by creating it for the target system
        connector = self._resolve_connector_for(action)
        
        async with self._execution_lock:
            # Build initial context with connector for stages
            context: Dict[str, Any] = {"connector": connector}
            
            # Execute pipeline
            result = await self._execution_pipeline.execute(action, initial_context=context)
            
            # Publish execution result event
            if self.event_bus:
                try:
                    await self.event_bus.publish_execution_result(
                        ExecutionResultEvent(
                            execution_result=result,
                            metadata=EventMetadata(
                                source=f"execution_adapter:{self.adapter_id}",
                                tags={
                                    "action_type": action.action_type,
                                    "target_system": action.target_system,
                                    "status": result.status.value if hasattr(result.status, "value") else str(result.status),
                                },
                            ),
                        )
                    )
                except Exception as e:
                    logger.error(f"Failed to publish execution result event: {e}")
            
            # Update adapter metrics
            if result.status == ExecutionStatus.SUCCESS:
                self.update_metrics(processed=1)
            else:
                self.update_metrics(failed=1)
            
            return result
    
    def _resolve_connector_for(self, action: AdaptationAction) -> ManagedSystemConnector:
        """Create a connector for the action's target_system based on configuration."""
        if not self.connector_factory:
            raise RuntimeError("Connector factory not initialized")
        
        cfg = self.configuration.config
        systems = cfg.get("managed_systems", []) if isinstance(cfg, dict) else []
        for entry in systems:
            if entry.get("system_id") == action.target_system:
                connector_type = entry.get("connector_type")
                connector_cfg = entry.get("config", {})
                try:
                    return self.connector_factory.create_connector(connector_type, connector_cfg)
                except Exception as e:
                    raise RuntimeError(f"Failed to create connector for system '{action.target_system}': {e}")
        
        raise RuntimeError(f"No connector configuration found for target_system '{action.target_system}'")