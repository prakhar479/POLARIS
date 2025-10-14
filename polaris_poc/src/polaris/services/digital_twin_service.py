"""
Digital Twin gRPC Service Implementation.

This module implements the gRPC service endpoints for the Digital Twin,
providing synchronous query, simulation, diagnosis, and management capabilities.
"""

import asyncio
import logging
import uuid
from typing import Optional
from datetime import datetime, timezone

import grpc
from grpc import aio

from polaris.proto import digital_twin_pb2, digital_twin_pb2_grpc
from polaris.models.world_model import (
    WorldModel,
    QueryRequest,
    QueryResponse,
    SimulationRequest,
    SimulationResponse,
    DiagnosisRequest,
    DiagnosisResponse,
)


class DigitalTwinService(digital_twin_pb2_grpc.DigitalTwinServicer):
    """
    Digital Twin gRPC service implementation.

    This service provides synchronous access to the Digital Twin's World Model
    through gRPC endpoints for queries, simulations, diagnostics, and management.
    """

    def __init__(self, world_model: WorldModel, logger: Optional[logging.Logger] = None):
        """Initialize the Digital Twin gRPC service.

        Args:
            world_model: World Model instance to delegate operations to
            logger: Logger instance (created if not provided)
        """
        self.world_model = world_model
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        # Service metrics
        self._metrics = {
            "queries_processed": 0,
            "simulations_processed": 0,
            "diagnoses_processed": 0,
            "management_requests_processed": 0,
            "total_errors": 0,
            "service_start_time": datetime.now(timezone.utc).isoformat(),
        }

        self.logger.info("Digital Twin gRPC service initialized")

    async def Query(
        self, request: digital_twin_pb2.QueryRequest, context: aio.ServicerContext
    ) -> digital_twin_pb2.QueryResponse:
        """Handle query requests for current or historical system state.

        Args:
            request: gRPC query request
            context: gRPC service context

        Returns:
            gRPC query response
        """
        start_time = datetime.now(timezone.utc)
        query_id = request.query_id or str(uuid.uuid4())

        # Initial query received log
        self.logger.info(
            "Query received - starting processing",
            extra={
                "query_id": query_id,
                "query_type": request.query_type,
                "timestamp": start_time.isoformat(),
                "client_info": context.peer() if hasattr(context, "peer") else "unknown",
            },
        )

        try:
            self.logger.debug(
                "Processing query request - validating parameters",
                extra={
                    "query_id": query_id,
                    "query_type": request.query_type,
                    "query_content": (
                        request.query_content[:100] + "..."
                        if len(request.query_content) > 100
                        else request.query_content
                    ),
                    "parameters_count": len(request.parameters),
                },
            )

            # Convert gRPC request to internal format
            self.logger.debug(
                "Converting gRPC request to internal format", extra={"query_id": query_id}
            )

            query_request = QueryRequest(
                query_id=query_id,
                query_type=request.query_type,
                query_content=request.query_content,
                parameters=dict(request.parameters),
                timestamp=request.timestamp or datetime.now(timezone.utc).isoformat(),
            )

            # Execute query through World Model
            self.logger.info(
                "Delegating query to World Model",
                extra={"query_id": query_id, "world_model_type": type(self.world_model).__name__},
            )

            query_response = await self.world_model.query_state(query_request)

            self.logger.debug(
                "World Model query completed",
                extra={
                    "query_id": query_id,
                    "success": query_response.success,
                    "confidence": query_response.confidence,
                },
            )

            # Convert internal response to gRPC format
            self.logger.debug(
                "Converting World Model response to gRPC format",
                extra={
                    "query_id": query_id,
                    "result_length": len(query_response.result) if query_response.result else 0,
                },
            )

            grpc_response = digital_twin_pb2.QueryResponse(
                query_id=query_response.query_id,
                success=query_response.success,
                result=query_response.result,
                confidence=query_response.confidence,
                explanation=query_response.explanation,
                timestamp=query_response.timestamp,
                metadata=query_response.metadata,
            )

            # Update metrics
            self._metrics["queries_processed"] += 1

            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            self.logger.info(
                "Query completed successfully - returning response",
                extra={
                    "query_id": query_response.query_id,
                    "success": query_response.success,
                    "confidence": query_response.confidence,
                    "processing_time_sec": round(processing_time, 3),
                    "total_queries_processed": self._metrics["queries_processed"],
                },
            )

            return grpc_response

        except Exception as e:
            self._metrics["total_errors"] += 1
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            self.logger.error(
                "Query processing failed - exception occurred",
                extra={
                    "query_id": query_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "processing_time_sec": round(processing_time, 3),
                    "total_errors": self._metrics["total_errors"],
                },
            )

            # Return error response
            self.logger.debug(
                "Returning error response for failed query", extra={"query_id": query_id}
            )

            return digital_twin_pb2.QueryResponse(
                query_id=query_id,
                success=False,
                result="",
                confidence=0.0,
                explanation=f"Query failed: {str(e)}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                metadata={"error": str(e)},
            )

    async def Simulate(
        self, request: digital_twin_pb2.SimulationRequest, context: aio.ServicerContext
    ) -> digital_twin_pb2.SimulationResponse:
        """Handle simulation requests for predictive "what-if" analysis.

        Args:
            request: gRPC simulation request
            context: gRPC service context

        Returns:
            gRPC simulation response
        """
        start_time = datetime.now(timezone.utc)

        try:
            self.logger.debug(
                "Processing simulation request",
                extra={
                    "simulation_id": request.simulation_id,
                    "simulation_type": request.simulation_type,
                    "horizon_minutes": request.horizon_minutes,
                    "actions_count": len(request.actions),
                },
            )

            # Convert gRPC actions to internal format
            actions = []
            for action in request.actions:
                actions.append(
                    {
                        "action_id": action.action_id,
                        "action_type": action.action_type,
                        "target": action.target,
                        "params": dict(action.params),
                        "priority": action.priority,
                        "timeout": action.timeout,
                    }
                )

            # Convert gRPC request to internal format
            sim_request = SimulationRequest(
                simulation_id=request.simulation_id or str(uuid.uuid4()),
                simulation_type=request.simulation_type,
                actions=actions,
                horizon_minutes=request.horizon_minutes,
                parameters=dict(request.parameters),
                timestamp=request.timestamp or datetime.now(timezone.utc).isoformat(),
            )

            # Execute simulation through World Model
            sim_response = await self.world_model.simulate(sim_request)

            # Convert future states to gRPC format
            grpc_future_states = []
            for state in sim_response.future_states:
                if isinstance(state, dict):
                    grpc_state = digital_twin_pb2.PredictedState(
                        timestamp=state.get("timestamp", ""),
                        metrics=state.get("metrics", {}),
                        confidence=state.get("confidence", 0.0),
                        description=state.get("description", ""),
                    )
                    grpc_future_states.append(grpc_state)

            # Create simulation metrics
            grpc_metrics = digital_twin_pb2.SimulationMetrics(
                execution_time_sec=(datetime.now(timezone.utc) - start_time).total_seconds(),
                scenarios_processed=1,
                average_confidence=sim_response.confidence,
                custom_metrics={},
            )

            # Create impact estimates
            grpc_impact = digital_twin_pb2.CostPerformanceReliability(
                cost_impact=sim_response.impact_estimates.get("cost_impact", 0.0),
                performance_impact=sim_response.impact_estimates.get("performance_impact", 0.0),
                reliability_impact=sim_response.impact_estimates.get("reliability_impact", 0.0),
                cost_currency=sim_response.impact_estimates.get("cost_currency", "USD"),
                impact_description=sim_response.impact_estimates.get("impact_description", ""),
            )

            # Convert internal response to gRPC format
            grpc_response = digital_twin_pb2.SimulationResponse(
                simulation_id=sim_response.simulation_id,
                success=sim_response.success,
                future_states=grpc_future_states,
                metrics=grpc_metrics,
                confidence=sim_response.confidence,
                uncertainty_lower=sim_response.uncertainty_lower,
                uncertainty_upper=sim_response.uncertainty_upper,
                explanation=sim_response.explanation,
                impact_estimates=grpc_impact,
                timestamp=sim_response.timestamp,
                metadata=sim_response.metadata,
            )

            # Update metrics
            self._metrics["simulations_processed"] += 1

            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            self.logger.info(
                "Simulation processed successfully",
                extra={
                    "simulation_id": sim_response.simulation_id,
                    "success": sim_response.success,
                    "confidence": sim_response.confidence,
                    "processing_time_sec": round(processing_time, 3),
                },
            )

            return grpc_response

        except Exception as e:
            self._metrics["total_errors"] += 1

            self.logger.error(
                f"Simulation processing failed due to {str(e)}",
                extra={"simulation_id": request.simulation_id, "error": str(e)},
            )
            # print error e in a file
            with open("simulation_errors.log", "a") as f:
                f.write(f"{datetime.now(timezone.utc).isoformat()} - Simulation error: {str(e)}\n")

            # Return error response
            return digital_twin_pb2.SimulationResponse(
                simulation_id=request.simulation_id or str(uuid.uuid4()),
                success=False,
                future_states=[],
                metrics=digital_twin_pb2.SimulationMetrics(),
                confidence=0.0,
                uncertainty_lower=0.0,
                uncertainty_upper=0.0,
                explanation=f"Simulation failed: {str(e)}",
                impact_estimates=digital_twin_pb2.CostPerformanceReliability(),
                timestamp=datetime.now(timezone.utc).isoformat(),
                metadata={"error": str(e)},
            )

    async def Diagnose(
        self, request: digital_twin_pb2.DiagnosisRequest, context: aio.ServicerContext
    ) -> digital_twin_pb2.DiagnosisResponse:
        """Handle diagnosis requests for root cause analysis.

        Args:
            request: gRPC diagnosis request
            context: gRPC service context

        Returns:
            gRPC diagnosis response
        """
        start_time = datetime.now(timezone.utc)

        try:
            self.logger.debug(
                "Processing diagnosis request",
                extra={
                    "diagnosis_id": request.diagnosis_id,
                    "anomaly_description": (
                        request.anomaly_description[:100] + "..."
                        if len(request.anomaly_description) > 100
                        else request.anomaly_description
                    ),
                },
            )

            # Convert gRPC request to internal format
            diag_request = DiagnosisRequest(
                diagnosis_id=request.diagnosis_id or str(uuid.uuid4()),
                anomaly_description=request.anomaly_description,
                context=dict(request.context),
                timestamp=request.timestamp or datetime.now(timezone.utc).isoformat(),
            )

            # Execute diagnosis through World Model
            diag_response = await self.world_model.diagnose(diag_request)

            # Convert hypotheses to gRPC format
            grpc_hypotheses = []
            for i, hypothesis in enumerate(diag_response.hypotheses):
                if isinstance(hypothesis, dict):
                    grpc_hypothesis = digital_twin_pb2.CausalHypothesis(
                        hypothesis=hypothesis.get("hypothesis", ""),
                        probability=hypothesis.get("probability", 0.0),
                        reasoning=hypothesis.get("reasoning", ""),
                        evidence=hypothesis.get("evidence", []),
                        rank=hypothesis.get("rank", i + 1),
                    )
                elif isinstance(hypothesis, str):
                    grpc_hypothesis = digital_twin_pb2.CausalHypothesis(
                        hypothesis=hypothesis,
                        probability=1.0 / (i + 1),  # Simple ranking-based probability
                        reasoning="",
                        evidence=[],
                        rank=i + 1,
                    )
                else:
                    continue

                grpc_hypotheses.append(grpc_hypothesis)

            # Convert internal response to gRPC format
            grpc_response = digital_twin_pb2.DiagnosisResponse(
                diagnosis_id=diag_response.diagnosis_id,
                success=diag_response.success,
                hypotheses=grpc_hypotheses,
                causal_chain=diag_response.causal_chain,
                confidence=diag_response.confidence,
                explanation=diag_response.explanation,
                supporting_evidence=diag_response.supporting_evidence,
                timestamp=diag_response.timestamp,
                metadata=diag_response.metadata,
            )

            # Update metrics
            self._metrics["diagnoses_processed"] += 1

            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            self.logger.info(
                "Diagnosis processed successfully",
                extra={
                    "diagnosis_id": diag_response.diagnosis_id,
                    "success": diag_response.success,
                    "confidence": diag_response.confidence,
                    "hypotheses_count": len(grpc_hypotheses),
                    "processing_time_sec": round(processing_time, 3),
                },
            )

            return grpc_response

        except Exception as e:
            self._metrics["total_errors"] += 1

            self.logger.error(
                "Diagnosis processing failed",
                extra={"diagnosis_id": request.diagnosis_id, "error": str(e)},
            )

            # Return error response
            return digital_twin_pb2.DiagnosisResponse(
                diagnosis_id=request.diagnosis_id or str(uuid.uuid4()),
                success=False,
                hypotheses=[],
                causal_chain="",
                confidence=0.0,
                explanation=f"Diagnosis failed: {str(e)}",
                supporting_evidence=[],
                timestamp=datetime.now(timezone.utc).isoformat(),
                metadata={"error": str(e)},
            )

    async def Manage(
        self, request: digital_twin_pb2.ManagementRequest, context: aio.ServicerContext
    ) -> digital_twin_pb2.ManagementResponse:
        """Handle management requests for lifecycle control and introspection.

        Args:
            request: gRPC management request
            context: gRPC service context

        Returns:
            gRPC management response
        """
        start_time = datetime.now(timezone.utc)

        try:
            self.logger.debug(
                "Processing management request",
                extra={"request_id": request.request_id, "operation": request.operation},
            )

            success = True
            result = ""
            metrics = {}
            health_status = None

            # Handle different management operations
            if request.operation == "health_check":
                health_data = await self.world_model.get_health_status()
                health_status = digital_twin_pb2.HealthStatus(
                    status=health_data.get("status", "unknown"),
                    last_check=health_data.get(
                        "last_check", datetime.now(timezone.utc).isoformat()
                    ),
                    performance_metrics=health_data.get("metrics", {}),
                    issues=health_data.get("issues", []),
                    model_type=health_data.get("model_type", "unknown"),
                    model_version=health_data.get("model_version", "unknown"),
                )
                result = f"Health check completed: {health_data.get('status', 'unknown')}"

            elif request.operation == "reload_model":
                reload_success = await self.world_model.reload_model()
                success = reload_success
                result = "Model reloaded successfully" if reload_success else "Model reload failed"

            elif request.operation == "get_metrics":
                service_metrics = self._metrics.copy()
                service_metrics.update(
                    {
                        "current_time": datetime.now(timezone.utc).isoformat(),
                        "uptime_sec": (
                            datetime.now(timezone.utc)
                            - datetime.fromisoformat(
                                self._metrics["service_start_time"].replace("Z", "+00:00")
                            )
                        ).total_seconds(),
                    }
                )
                metrics = {k: str(v) for k, v in service_metrics.items()}
                result = "Service metrics retrieved"

            elif request.operation == "reset_state":
                # This would require World Model support for state reset
                result = "State reset not implemented in current World Model"
                success = False

            else:
                result = f"Unknown operation: {request.operation}"
                success = False

            # Create response
            grpc_response = digital_twin_pb2.ManagementResponse(
                request_id=request.request_id or str(uuid.uuid4()),
                success=success,
                result=result,
                metrics=metrics,
                health_status=health_status,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            # Update metrics
            self._metrics["management_requests_processed"] += 1

            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            self.logger.info(
                "Management request processed",
                extra={
                    "request_id": request.request_id,
                    "operation": request.operation,
                    "success": success,
                    "processing_time_sec": round(processing_time, 3),
                },
            )

            return grpc_response

        except Exception as e:
            self._metrics["total_errors"] += 1

            self.logger.error(
                "Management request processing failed",
                extra={
                    "request_id": request.request_id,
                    "operation": request.operation,
                    "error": str(e),
                },
            )

            # Return error response
            return digital_twin_pb2.ManagementResponse(
                request_id=request.request_id or str(uuid.uuid4()),
                success=False,
                result=f"Management operation failed: {str(e)}",
                metrics={"error": str(e)},
                health_status=None,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

    def get_service_metrics(self) -> dict:
        """Get current service metrics.

        Returns:
            Dictionary containing service performance metrics
        """
        return self._metrics.copy()
