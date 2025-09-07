"""
Gemini LLM-based World Model implementation for POLARIS Digital Twin.

This module implements the World Model interface using Google's Gemini LLM
via LangChain, providing conversational context management, state tracking,
predictive simulation, and diagnostic reasoning capabilities.
"""

import asyncio
import json
import logging
import os
import hashlib
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from collections import deque, defaultdict
import uuid

from .world_model import (
    WorldModel, WorldModelError, WorldModelInitializationError, WorldModelOperationError,
    QueryRequest, QueryResponse, SimulationRequest, SimulationResponse,
    DiagnosisRequest, DiagnosisResponse
)
from .digital_twin_events import KnowledgeEvent, CalibrationEvent


class GeminiWorldModel(WorldModel):
    """
    Gemini LLM-based World Model implementation.
    
    This implementation uses Google's Gemini LLM via LangChain to provide
    intelligent state tracking, predictive simulation, and diagnostic reasoning
    with conversational context management and uncertainty quantification.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize the Gemini World Model.
        
        Args:
            config: Configuration dictionary containing Gemini API settings
            logger: Logger instance (created if not provided)
        """
        super().__init__(config, logger)
        
        # Gemini API configuration
        self.api_key_env = config.get("api_key_env", "GEMINI_API_KEY")
        self.model_name = config.get("model", "gemini-1.5-pro")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 2000)
        self.concurrent_requests = config.get("concurrent_requests", 5)
        
        # Memory and context management
        self.max_history_events = config.get("max_history_events", 10000)
        self.retention_days = config.get("retention_days", 30)
        self.compression_threshold = config.get("compression_threshold", 1000)
        self.enable_causal_tracking = config.get("enable_causal_tracking", True)
        
        # Confidence scoring configuration
        self.min_confidence_threshold = config.get("min_confidence_threshold", 0.3)
        self.uncertainty_estimation = config.get("uncertainty_estimation", True)
        self.calibration_window_hours = config.get("calibration_window_hours", 24)
        
        # Diagnostic configuration
        self.max_hypotheses = config.get("max_hypotheses", 5)
        self.causal_depth_limit = config.get("causal_depth_limit", 10)
        self.evidence_weight_threshold = config.get("evidence_weight_threshold", 0.1)
        
        # Internal state
        self._llm_client = None
        self._conversation_memory = deque(maxlen=100)  # Recent conversation context
        self._system_state_history = deque(maxlen=self.max_history_events)
        self._calibration_history = deque(maxlen=1000)
        self._current_system_state = {}
        self._accuracy_metrics = {"overall": 0.8, "last_updated": None}
        
        # Enhanced state tracking components
        self._state_embeddings = {}  # Vector embeddings for state similarity
        self._state_metadata = {}  # Enhanced metadata tracking
        self._state_consistency_log = deque(maxlen=1000)  # Consistency validation history
        self._temporal_state_index = defaultdict(list)  # Time-based state indexing
        self._state_evolution_patterns = {}  # Learned state evolution patterns
        
        # Concurrent processing
        self._simulation_semaphore = asyncio.Semaphore(self.concurrent_requests)
        self._processing_lock = asyncio.Lock()
        
        self.logger.info(f"Initialized GeminiWorldModel with model: {self.model_name}")
    
    async def initialize(self) -> None:
        """Initialize the Gemini World Model implementation.
        
        Sets up the LangChain Gemini client and prepares the conversational context.
        
        Raises:
            WorldModelInitializationError: If initialization fails
        """
        try:
            self.logger.info("Initializing Gemini World Model...")
            
            # Check for API key
            api_key = os.getenv(self.api_key_env)
            if not api_key:
                raise WorldModelInitializationError(
                    f"Gemini API key not found in environment variable: {self.api_key_env}"
                )
            
            # For now, we'll simulate LangChain initialization
            # In a real implementation, this would initialize the actual LangChain Gemini client
            self._llm_client = self._create_mock_llm_client()
            
            # Initialize system prompt and context
            await self._initialize_system_context()
            
            # Set initialization status
            self._set_initialized(True)
            self._health_status.update({
                "status": "healthy",
                "model": self.model_name,
                "last_check": datetime.now(timezone.utc).isoformat(),
                "metrics": self._accuracy_metrics
            })
            
            self.logger.info("Gemini World Model initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini World Model: {str(e)}")
            raise WorldModelInitializationError(f"Initialization failed: {str(e)}") from e
    
    async def shutdown(self) -> None:
        """Shutdown the Gemini World Model implementation."""
        try:
            self.logger.info("Shutting down Gemini World Model...")
            
            # Clean up resources
            self._llm_client = None
            self._conversation_memory.clear()
            
            # Set shutdown status
            self._set_initialized(False)
            self._health_status.update({
                "status": "shutdown",
                "last_check": datetime.now(timezone.utc).isoformat()
            })
            
            self.logger.info("Gemini World Model shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
    
    def _create_mock_llm_client(self) -> Dict[str, Any]:
        """Create a mock LLM client for development purposes.
        
        In a real implementation, this would create the actual LangChain Gemini client.
        
        Returns:
            Mock client configuration
        """
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_key_configured": True,
            "mock": True  # Indicates this is a mock implementation
        }
    
    async def _initialize_system_context(self) -> None:
        """Initialize the system context and conversation memory."""
        system_prompt = """
        You are the World Model component of the POLARIS self-adaptive system framework.
        Your role is to maintain a comprehensive understanding of the managed software system
        and provide intelligent analysis, predictions, and diagnostics.
        
        Key responsibilities:
        1. Track system state evolution from telemetry and execution events
        2. Provide accurate predictions about future system behavior
        3. Perform root cause analysis for anomalies and issues
        4. Maintain uncertainty quantification for all predictions
        5. Learn from calibration feedback to improve accuracy
        
        Always provide:
        - Confidence scores (0.0 to 1.0) for all responses
        - Clear explanations for your reasoning
        - Uncertainty intervals for predictions
        - Evidence-based diagnostic hypotheses
        
        Current system context: Initializing...
        """
        
        self._conversation_memory.append({
            "role": "system",
            "content": system_prompt,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        self.logger.debug("System context initialized")    

    async def update_state(self, event: KnowledgeEvent) -> None:
        """Update the world model state with new knowledge.
        
        Processes incoming knowledge events and integrates them into the
        world model using LLM reasoning for state evolution.
        
        Args:
            event: Knowledge event containing system state updates
            
        Raises:
            WorldModelOperationError: If state update fails
        """
        try:
            async with self._processing_lock:
                self.logger.debug(f"Processing knowledge event: {event.event_type} from {event.source}")
                
                # Add event to history
                self._system_state_history.append({
                    "event": event.to_dict(),
                    "processed_at": datetime.now(timezone.utc).isoformat()
                })
                
                # Process different event types
                if event.event_type == "telemetry":
                    await self._process_telemetry_event(event)
                elif event.event_type == "execution_status":
                    await self._process_execution_event(event)
                elif event.event_type == "anomaly":
                    await self._process_anomaly_event(event)
                else:
                    self.logger.warning(f"Unknown event type: {event.event_type}")
                
                # Update conversation context
                self._add_to_conversation_memory(
                    "user",
                    f"System update: {event.event_type} event from {event.source} at {event.timestamp}"
                )
                
                self.logger.debug(f"Successfully processed knowledge event: {event.event_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to update state with event {event.event_id}: {str(e)}")
            raise WorldModelOperationError(f"State update failed: {str(e)}") from e
    
    async def calibrate(self, event: CalibrationEvent) -> None:
        """Calibrate the world model based on prediction accuracy feedback.
        
        Adjusts the model based on how accurate previous predictions were
        compared to actual outcomes, improving future predictions.
        
        Args:
            event: Calibration event with accuracy feedback
            
        Raises:
            WorldModelOperationError: If calibration fails
        """
        try:
            async with self._processing_lock:
                self.logger.debug(f"Processing calibration event: {event.calibration_id}")
                
                # Add calibration to history
                self._calibration_history.append({
                    "event": event.to_dict(),
                    "processed_at": datetime.now(timezone.utc).isoformat()
                })
                
                # Calculate accuracy score
                accuracy_score = event.calculate_accuracy_score()
                
                # Update accuracy metrics
                self._update_accuracy_metrics(accuracy_score, event.accuracy_metrics)
                
                # Add to conversation memory for learning
                calibration_context = (
                    f"Calibration feedback: Prediction {event.prediction_id} had accuracy {accuracy_score:.2f}. "
                    f"Predicted: {json.dumps(event.predicted_outcome, indent=None)}, "
                    f"Actual: {json.dumps(event.actual_outcome, indent=None)}"
                )
                
                self._add_to_conversation_memory("user", calibration_context)
                
                # Simulate LLM learning from feedback
                learning_response = await self._simulate_llm_learning(event)
                self._add_to_conversation_memory("assistant", learning_response)
                
                self.logger.info(f"Calibration complete: accuracy={accuracy_score:.2f}")
                
        except Exception as e:
            self.logger.error(f"Failed to calibrate with event {event.calibration_id}: {str(e)}")
            raise WorldModelOperationError(f"Calibration failed: {str(e)}") from e
    
    async def query_state(self, request: QueryRequest) -> QueryResponse:
        """Query the current or historical system state.
        
        Handles different types of queries using LLM reasoning with
        confidence scoring and explanation generation.
        
        Args:
            request: Query request with type and parameters
            
        Returns:
            Query response with results and confidence
            
        Raises:
            WorldModelOperationError: If query fails
        """
        try:
            self.logger.debug(f"Processing query: {request.query_type} - {request.query_content}")
            
            # Process different query types
            if request.query_type == "current_state":
                result, confidence, explanation = await self._query_current_state(request)
            elif request.query_type == "historical":
                result, confidence, explanation = await self._query_historical_state(request)
            elif request.query_type == "natural_language":
                result, confidence, explanation = await self._query_natural_language(request)
            else:
                raise WorldModelOperationError(f"Unsupported query type: {request.query_type}")
            
            # Create response
            response = QueryResponse(
                query_id=request.query_id,
                success=True,
                result=result,
                confidence=confidence,
                explanation=explanation,
                metadata={
                    "query_type": request.query_type,
                    "processing_time": "simulated",
                    "model": self.model_name
                }
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
                explanation=f"Query failed: {str(e)}"
            )
    
    async def simulate(self, request: SimulationRequest) -> SimulationResponse:
        """Perform predictive simulation ("what-if" analysis).
        
        Simulates future system states based on proposed actions using
        LLM reasoning with uncertainty quantification.
        
        Args:
            request: Simulation request with actions and parameters
            
        Returns:
            Simulation response with predictions and uncertainty
            
        Raises:
            WorldModelOperationError: If simulation fails
        """
        try:
            # Use semaphore for concurrent processing
            async with self._simulation_semaphore:
                self.logger.debug(f"Processing simulation: {request.simulation_type} - {request.simulation_id}")
                
                # Process different simulation types
                if request.simulation_type == "forecast":
                    future_states, confidence, uncertainty_bounds, explanation = await self._simulate_forecast(request)
                elif request.simulation_type == "what_if":
                    future_states, confidence, uncertainty_bounds, explanation = await self._simulate_what_if(request)
                elif request.simulation_type == "scenario":
                    future_states, confidence, uncertainty_bounds, explanation = await self._simulate_scenario(request)
                else:
                    raise WorldModelOperationError(f"Unsupported simulation type: {request.simulation_type}")
                
                # Generate impact estimates
                impact_estimates = await self._generate_impact_estimates(future_states, request)
                
                # Create response
                response = SimulationResponse(
                    simulation_id=request.simulation_id,
                    success=True,
                    future_states=future_states,
                    confidence=confidence,
                    uncertainty_lower=uncertainty_bounds[0],
                    uncertainty_upper=uncertainty_bounds[1],
                    explanation=explanation,
                    impact_estimates=impact_estimates,
                    metadata={
                        "simulation_type": request.simulation_type,
                        "horizon_minutes": request.horizon_minutes,
                        "actions_count": len(request.actions),
                        "model": self.model_name
                    }
                )
                
                self.logger.debug(f"Simulation completed: {request.simulation_id} (confidence: {confidence:.2f})")
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
                explanation=f"Simulation failed: {str(e)}"
            )
    
    async def diagnose(self, request: DiagnosisRequest) -> DiagnosisResponse:
        """Perform root cause analysis and diagnosis.
        
        Analyzes anomalies using LLM causal reasoning to identify likely
        root causes and provide ranked hypotheses with explanations.
        
        Args:
            request: Diagnosis request with anomaly description
            
        Returns:
            Diagnosis response with hypotheses and explanations
            
        Raises:
            WorldModelOperationError: If diagnosis fails
        """
        try:
            self.logger.debug(f"Processing diagnosis: {request.diagnosis_id}")
            
            # Perform causal analysis
            hypotheses = await self._generate_causal_hypotheses(request)
            causal_chain = await self._identify_causal_chain(request, hypotheses)
            confidence = await self._calculate_diagnostic_confidence(hypotheses)
            explanation = await self._generate_diagnostic_explanation(request, hypotheses, causal_chain)
            supporting_evidence = await self._gather_supporting_evidence(request, hypotheses)
            
            # Create response
            response = DiagnosisResponse(
                diagnosis_id=request.diagnosis_id,
                success=True,
                hypotheses=hypotheses,
                causal_chain=causal_chain,
                confidence=confidence,
                explanation=explanation,
                supporting_evidence=supporting_evidence,
                metadata={
                    "anomaly_timestamp": request.timestamp,
                    "context_keys": list(request.context.keys()),
                    "hypotheses_count": len(hypotheses),
                    "model": self.model_name
                }
            )
            
            self.logger.debug(f"Diagnosis completed: {request.diagnosis_id} (confidence: {confidence:.2f})")
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to process diagnosis {request.diagnosis_id}: {str(e)}")
            return DiagnosisResponse(
                diagnosis_id=request.diagnosis_id,
                success=False,
                hypotheses=[],
                causal_chain="",
                confidence=0.0,
                explanation=f"Diagnosis failed: {str(e)}"
            )
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get the current health status of the World Model.
        
        Returns:
            Dictionary containing health status information
        """
        # Calculate state tracking health metrics
        anomaly_count = sum(1 for state in self._current_system_state.values() 
                           if isinstance(state, dict) and state.get("metadata", {}).get("anomaly_score", 0) > 0.7)
        
        low_confidence_count = sum(1 for state in self._current_system_state.values() 
                                  if isinstance(state, dict) and state.get("metadata", {}).get("confidence", 1.0) < 0.6)
        
        consistency_issues = sum(1 for check in self._state_consistency_log 
                               if check.get("consistency_score", 1.0) < 0.7)
        
        return {
            "status": "healthy" if self.is_initialized else "not_initialized",
            "model_type": "gemini_llm",
            "model_name": self.model_name,
            "last_check": datetime.now(timezone.utc).isoformat(),
            "metrics": {
                "overall_accuracy": self._accuracy_metrics.get("overall", 0.0),
                "events_processed": len(self._system_state_history),
                "calibrations_received": len(self._calibration_history),
                "conversation_memory_size": len(self._conversation_memory),
                "concurrent_capacity": self.concurrent_requests,
                # Enhanced state tracking metrics
                "current_state_metrics": len(self._current_system_state),
                "state_embeddings_stored": len(self._state_embeddings),
                "temporal_index_keys": len(self._temporal_state_index),
                "evolution_patterns_tracked": len(self._state_evolution_patterns),
                "anomalies_detected": anomaly_count,
                "low_confidence_metrics": low_confidence_count,
                "consistency_issues": consistency_issues
            },
            "configuration": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "max_history_events": self.max_history_events,
                "enable_causal_tracking": self.enable_causal_tracking,
                # Enhanced state tracking configuration
                "vector_embeddings_enabled": True,
                "state_consistency_validation": True,
                "temporal_indexing_enabled": True,
                "llm_state_analysis_enabled": True
            },
            "state_tracking_status": {
                "embeddings_health": "healthy" if len(self._state_embeddings) > 0 else "no_data",
                "consistency_health": "healthy" if consistency_issues < 5 else "issues_detected",
                "anomaly_detection": "active" if anomaly_count < 10 else "high_anomaly_rate"
            }
        }
    
    async def reload_model(self) -> bool:
        """Reload the World Model implementation.
        
        Reloads the model configuration and resets the model state.
        
        Returns:
            True if reload was successful, False otherwise
        """
        try:
            self.logger.info("Reloading Gemini World Model...")
            
            # Shutdown current instance
            await self.shutdown()
            
            # Reinitialize
            await self.initialize()
            
            self.logger.info("Gemini World Model reloaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to reload model: {str(e)}")
            return False    
 
   # Helper methods for event processing
    
    async def _process_telemetry_event(self, event: KnowledgeEvent) -> None:
        """Process telemetry events and update system state with LLM reasoning."""
        try:
            telemetry_data = event.data
            
            # Extract key metrics from telemetry
            if hasattr(telemetry_data, 'name') and hasattr(telemetry_data, 'value'):
                original_metric_name = telemetry_data.name
                metric_value = telemetry_data.value
                
                # Normalize metric name for consistent storage
                metric_name = self._normalize_metric_name(original_metric_name)
                
                # Create enhanced state entry with metadata
                state_entry = {
                    "value": metric_value,
                    "timestamp": event.timestamp,
                    "source": event.source,
                    "event_id": event.event_id,
                    "original_name": original_metric_name,  # Keep original for reference
                    "canonical_name": metric_name,  # Normalized name
                    "metadata": {
                        "processing_time": datetime.now(timezone.utc).isoformat(),
                        "confidence": await self._calculate_telemetry_confidence(telemetry_data),
                        "anomaly_score": await self._calculate_anomaly_score(metric_name, metric_value),
                        "trend": await self._analyze_metric_trend(metric_name, metric_value)
                    }
                }
                
                # Update current system state using canonical name
                previous_value = self._current_system_state.get(metric_name, {}).get("value")
                self._current_system_state[metric_name] = state_entry
                
                # Generate and store state embedding for similarity search
                await self._generate_and_store_state_embedding(metric_name, state_entry)
                
                # Perform state consistency validation
                await self._validate_state_consistency(metric_name, state_entry, previous_value)
                
                # Update temporal index for historical queries
                await self._update_temporal_index(metric_name, state_entry)
                
                # Use LLM reasoning to analyze state evolution
                await self._analyze_state_evolution_with_llm(metric_name, state_entry, previous_value)
                
                self.logger.debug(f"Enhanced state update: {metric_name} = {metric_value} "
                                f"(confidence: {state_entry['metadata']['confidence']:.2f}, "
                                f"anomaly: {state_entry['metadata']['anomaly_score']:.2f})")
            
        except Exception as e:
            self.logger.warning(f"Failed to process telemetry event: {str(e)}")
    
    async def _process_execution_event(self, event: KnowledgeEvent) -> None:
        """Process execution status events."""
        try:
            execution_data = event.data
            
            # Track execution results
            if hasattr(execution_data, 'action_id') and hasattr(execution_data, 'status'):
                action_id = execution_data.action_id
                status = execution_data.status
                
                # Update execution tracking
                self._current_system_state[f"execution_{action_id}"] = {
                    "status": status,
                    "timestamp": event.timestamp,
                    "source": event.source
                }
                
                self.logger.debug(f"Updated execution status: {action_id} = {status}")
            
        except Exception as e:
            self.logger.warning(f"Failed to process execution event: {str(e)}")
    
    async def _process_anomaly_event(self, event: KnowledgeEvent) -> None:
        """Process anomaly events."""
        try:
            anomaly_data = event.data
            
            # Track anomalies for diagnostic purposes
            anomaly_key = f"anomaly_{event.event_id}"
            self._current_system_state[anomaly_key] = {
                "data": anomaly_data,
                "timestamp": event.timestamp,
                "source": event.source
            }
            
            self.logger.debug(f"Recorded anomaly: {anomaly_key}")
            
        except Exception as e:
            self.logger.warning(f"Failed to process anomaly event: {str(e)}")
    
    def _add_to_conversation_memory(self, role: str, content: str) -> None:
        """Add a message to conversation memory."""
        self._conversation_memory.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def _update_accuracy_metrics(self, accuracy_score: float, detailed_metrics: Dict[str, float]) -> None:
        """Update accuracy metrics based on calibration feedback."""
        # Simple exponential moving average
        alpha = 0.1  # Learning rate
        current_accuracy = self._accuracy_metrics.get("overall", 0.8)
        new_accuracy = alpha * accuracy_score + (1 - alpha) * current_accuracy
        
        self._accuracy_metrics.update({
            "overall": new_accuracy,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "recent_score": accuracy_score,
            "detailed": detailed_metrics
        })
    
    async def _simulate_llm_learning(self, event: CalibrationEvent) -> str:
        """Simulate LLM learning from calibration feedback."""
        accuracy = event.calculate_accuracy_score()
        
        if accuracy > 0.8:
            return f"Good prediction accuracy ({accuracy:.2f}). Maintaining current reasoning approach."
        elif accuracy > 0.6:
            return f"Moderate accuracy ({accuracy:.2f}). Will adjust confidence levels and consider additional factors."
        else:
            return f"Low accuracy ({accuracy:.2f}). Need to reconsider prediction methodology and incorporate more context."
    
    # Query processing methods
    
    async def _query_current_state(self, request: QueryRequest) -> tuple[str, float, str]:
        """Process current state queries with enhanced confidence scoring and metric name resolution."""
        try:
            # Get specific metric if requested
            requested_metric = request.parameters.get("metric", "")
            return_format = request.parameters.get("format", "json")  # json or summary
            
            if requested_metric:
                # Resolve metric name to actual stored key
                resolved_metric = self._resolve_metric_name(requested_metric)
                
                if resolved_metric and resolved_metric in self._current_system_state:
                    # Return specific metric with full metadata
                    state_entry = self._current_system_state[resolved_metric]
                    
                    # Create structured result
                    structured_result = {
                        "requested_metric": requested_metric,
                        "resolved_metric": resolved_metric,
                        "original_name": state_entry.get("original_name", resolved_metric),
                        "canonical_name": state_entry.get("canonical_name", resolved_metric),
                        "value": state_entry["value"],
                        "timestamp": state_entry["timestamp"],
                        "source": state_entry["source"],
                        "confidence": state_entry["metadata"]["confidence"],
                        "anomaly_score": state_entry["metadata"]["anomaly_score"],
                        "trend": state_entry["metadata"]["trend"],
                        "event_id": state_entry.get("event_id", "unknown")
                    }
                    
                    confidence = state_entry["metadata"]["confidence"]
                    
                    if return_format == "summary":
                        # Human-readable summary
                        result = (f"Metric '{requested_metric}' (resolved as '{resolved_metric}'): "
                                f"value={state_entry['value']}, confidence={confidence:.2f}, "
                                f"trend={state_entry['metadata']['trend']}, "
                                f"anomaly_score={state_entry['metadata']['anomaly_score']:.2f}")
                        explanation = f"Human-readable summary for {requested_metric}"
                    else:
                        # Structured JSON result
                        result = json.dumps(structured_result, indent=2)
                        explanation = f"Structured data for {requested_metric} with metric name resolution"
                    
                    return result, confidence, explanation
                
                else:
                    # Metric not found - provide helpful suggestions
                    available_metrics = list(self._current_system_state.keys())
                    suggestions = []
                    
                    # Find similar metrics
                    requested_normalized = self._normalize_metric_name(requested_metric)
                    for available in available_metrics:
                        available_normalized = self._normalize_metric_name(available)
                        if (requested_normalized in available_normalized or 
                            available_normalized in requested_normalized):
                            suggestions.append(available)
                    
                    error_result = {
                        "error": "metric_not_found",
                        "requested_metric": requested_metric,
                        "available_metrics": available_metrics[:10],  # Limit to first 10
                        "suggestions": suggestions[:5],  # Top 5 suggestions
                        "total_metrics_available": len(available_metrics)
                    }
                    
                    return (json.dumps(error_result, indent=2), 0.0, 
                           f"Metric '{requested_metric}' not found. Check available_metrics or suggestions.")
            
            else:
                # Get enhanced system state summary
                state_summary = await self._generate_enhanced_state_summary()
                
                # Calculate overall confidence based on individual metric confidences
                total_confidence = 0.0
                metric_count = 0
                
                for metric_data in self._current_system_state.values():
                    if isinstance(metric_data, dict) and "metadata" in metric_data:
                        total_confidence += metric_data["metadata"]["confidence"]
                        metric_count += 1
                
                overall_confidence = total_confidence / metric_count if metric_count > 0 else 0.8
                
                if return_format == "json":
                    # Structured overview
                    overview_result = {
                        "system_state_overview": {
                            "total_metrics": len(self._current_system_state),
                            "overall_confidence": overall_confidence,
                            "summary": state_summary,
                            "metrics": {
                                name: {
                                    "value": data["value"],
                                    "confidence": data["metadata"]["confidence"],
                                    "trend": data["metadata"]["trend"],
                                    "anomaly_score": data["metadata"]["anomaly_score"]
                                }
                                for name, data in self._current_system_state.items()
                                if isinstance(data, dict) and "metadata" in data
                            }
                        }
                    }
                    result = json.dumps(overview_result, indent=2)
                    explanation = f"Structured system overview with {metric_count} metrics"
                else:
                    # Human-readable summary
                    result = f"Current system state: {state_summary}"
                    explanation = f"Enhanced current state summary with confidence scoring from {metric_count} metrics"
                
                return result, overall_confidence, explanation
            
        except Exception as e:
            return f"Error querying current state: {str(e)}", 0.0, "Query processing failed"
    
    async def _query_historical_state(self, request: QueryRequest) -> tuple[str, float, str]:
        """Process historical state queries with enhanced embedding-based search."""
        try:
            # Use enhanced historical query with embeddings
            return await self._query_historical_state_with_embeddings(request)
            
        except Exception as e:
            return f"Error querying historical state: {str(e)}", 0.0, "Historical query failed"
    
    async def _query_natural_language(self, request: QueryRequest) -> tuple[str, float, str]:
        """Process natural language queries."""
        try:
            query_content = request.query_content.lower()
            
            # Simple keyword-based processing (mock implementation)
            if "cpu" in query_content or "performance" in query_content:
                result = "CPU usage is currently within normal ranges based on recent telemetry"
                confidence = 0.75
                explanation = "Analysis based on CPU-related telemetry data"
            elif "error" in query_content or "problem" in query_content:
                result = "No critical errors detected in recent system events"
                confidence = 0.7
                explanation = "Analysis based on error patterns in event history"
            else:
                result = f"Processed natural language query: '{request.query_content}'"
                confidence = 0.6
                explanation = "General natural language processing with limited context"
            
            return result, confidence, explanation
            
        except Exception as e:
            return f"Error processing natural language query: {str(e)}", 0.0, "Natural language processing failed"
    
    def _generate_state_summary(self) -> str:
        """Generate a summary of current system state."""
        if not self._current_system_state:
            return "No current state data available"
        
        # Create simple summary
        metrics = []
        for key, value in list(self._current_system_state.items())[:5]:  # Limit to 5 most recent
            if isinstance(value, dict) and "value" in value:
                metrics.append(f"{key}={value['value']}")
            else:
                metrics.append(f"{key}=present")
        
        return ", ".join(metrics) if metrics else "System state tracking active"
    
    async def _generate_enhanced_state_summary(self) -> str:
        """Generate an enhanced summary with confidence and anomaly information."""
        if not self._current_system_state:
            return "No current state data available"
        
        # Create enhanced summary with metadata
        summary_parts = []
        anomaly_count = 0
        low_confidence_count = 0
        
        for key, value in self._current_system_state.items():
            if isinstance(value, dict) and "value" in value and "metadata" in value:
                confidence = value["metadata"]["confidence"]
                anomaly_score = value["metadata"]["anomaly_score"]
                trend = value["metadata"]["trend"]
                
                # Count issues
                if anomaly_score > 0.7:
                    anomaly_count += 1
                if confidence < 0.6:
                    low_confidence_count += 1
                
                # Create metric summary with indicators
                indicators = []
                if anomaly_score > 0.7:
                    indicators.append("⚠️")
                if confidence < 0.6:
                    indicators.append("❓")
                if trend == "increasing":
                    indicators.append("📈")
                elif trend == "decreasing":
                    indicators.append("📉")
                
                indicator_str = "".join(indicators)
                summary_parts.append(f"{key}={value['value']}{indicator_str}")
            else:
                summary_parts.append(f"{key}=present")
        
        # Limit to most important metrics
        summary_parts = summary_parts[:8]
        
        # Add overall health indicators
        health_indicators = []
        if anomaly_count > 0:
            health_indicators.append(f"{anomaly_count} anomalies")
        if low_confidence_count > 0:
            health_indicators.append(f"{low_confidence_count} low-confidence")
        
        base_summary = ", ".join(summary_parts) if summary_parts else "System state tracking active"
        
        if health_indicators:
            return f"{base_summary} | Issues: {', '.join(health_indicators)}"
        else:
            return f"{base_summary} | System healthy"
    
    
    def _normalize_metric_name(self, metric_name: str) -> str:
        """Normalize metric name to canonical form for consistent storage."""
        if not metric_name:
            return metric_name
        
        # Convert to lowercase and replace common separators with dots
        normalized = metric_name.lower()
        normalized = normalized.replace('_', '.').replace('-', '.').replace(' ', '.')
        
        # Remove duplicate dots and leading/trailing dots
        while '..' in normalized:
            normalized = normalized.replace('..', '.')
        normalized = normalized.strip('.')
        
        return normalized
    
    def _resolve_metric_name(self, requested_name: str) -> Optional[str]:
        """Resolve a requested metric name to an actual stored metric key."""
        if not requested_name:
            return None
        
        # First try exact match
        if requested_name in self._current_system_state:
            return requested_name
        
        # Try normalized form
        normalized_requested = self._normalize_metric_name(requested_name)
        
        # Check all stored metrics for matches
        for stored_key in self._current_system_state.keys():
            stored_normalized = self._normalize_metric_name(stored_key)
            
            # Exact normalized match
            if normalized_requested == stored_normalized:
                return stored_key
            
            # Substring match (both directions)
            if (normalized_requested in stored_normalized or 
                stored_normalized in normalized_requested):
                return stored_key
            
            # Common aliases
            aliases = {
                'cpu': ['cpu.usage', 'cpu.utilization', 'processor'],
                'memory': ['mem', 'ram', 'memory.usage', 'memory.utilization'],
                'response.time': ['latency', 'response.latency', 'request.time'],
                'disk': ['storage', 'disk.usage', 'disk.utilization'],
                'network': ['net', 'bandwidth', 'network.usage']
            }
            
            for canonical, alias_list in aliases.items():
                if (normalized_requested == canonical and stored_normalized in alias_list) or \
                   (normalized_requested in alias_list and stored_normalized == canonical):
                    return stored_key
        
        return None
    
    def _get_metric_variants(self, base_name: str) -> List[str]:
        """Get common variants of a metric name for search purposes."""
        if not base_name:
            return []
        
        variants = [base_name]
        normalized = self._normalize_metric_name(base_name)
        
        # Add common separator variants
        variants.extend([
            normalized.replace('.', '_'),
            normalized.replace('.', '-'),
            normalized.replace('.', ' '),
            base_name.upper(),
            base_name.lower()
        ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variants = []
        for variant in variants:
            if variant not in seen:
                seen.add(variant)
                unique_variants.append(variant)
        
        return unique_variants
    
    def get_metric_state(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Get the state for a metric, handling name resolution.
        
        This is a helper method for tests and external access that handles
        metric name normalization and resolution.
        
        Args:
            metric_name: The metric name to look up (any variant)
            
        Returns:
            The metric state dictionary if found, None otherwise
        """
        resolved_name = self._resolve_metric_name(metric_name)
        if resolved_name and resolved_name in self._current_system_state:
            return self._current_system_state[resolved_name]
        return None
    
    async def _calculate_telemetry_confidence(self, telemetry_data: Any) -> float:
        """Calculate confidence score for telemetry data quality."""
        try:
            confidence = 0.8  # Base confidence
            
            # Adjust based on data completeness
            if hasattr(telemetry_data, 'name') and hasattr(telemetry_data, 'value'):
                confidence += 0.1
            
            # Adjust based on data type and range
            if hasattr(telemetry_data, 'value'):
                value = telemetry_data.value
                if isinstance(value, (int, float)) and 0 <= value <= 100:
                    confidence += 0.05  # Reasonable range for percentages
                elif isinstance(value, (int, float)) and value >= 0:
                    confidence += 0.03  # Non-negative numeric value
            
            # Adjust based on source reliability (mock implementation)
            if hasattr(telemetry_data, 'source'):
                source = getattr(telemetry_data, 'source', 'unknown')
                if 'monitor' in source.lower():
                    confidence += 0.05  # Monitor sources are more reliable
            
            return min(0.95, max(0.1, confidence))
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate telemetry confidence: {str(e)}")
            return 0.5
    
    async def _calculate_anomaly_score(self, metric_name: str, metric_value: Any) -> float:
        """Calculate anomaly score for a metric value based on historical patterns."""
        try:
            # Get historical values for this metric (handle metric name normalization)
            historical_values = []
            normalized_metric_name = self._normalize_metric_name(metric_name)
            
            for state_record in list(self._system_state_history)[-50:]:  # Last 50 events
                event_data = state_record.get("event", {}).get("data", {})
                if hasattr(event_data, 'name') and hasattr(event_data, 'value'):
                    event_metric_name = getattr(event_data, 'name')
                    normalized_event_name = self._normalize_metric_name(event_metric_name)
                    
                    # Match on normalized names
                    if normalized_event_name == normalized_metric_name:
                        historical_values.append(getattr(event_data, 'value'))
            
            if len(historical_values) < 2:  # Reduced threshold for testing
                # If we don't have enough historical data, use a simple heuristic
                if isinstance(metric_value, (int, float)):
                    # For CPU/Memory metrics, values > 90 are anomalous
                    if "cpu" in normalized_metric_name and metric_value > 90:
                        return 0.8
                    elif "memory" in normalized_metric_name and metric_value > 90:
                        return 0.8
                    # For response time, values > 500ms are anomalous
                    elif ("response" in normalized_metric_name or "latency" in normalized_metric_name) and metric_value > 500:
                        return 0.7
                return 0.0  # Not enough data for anomaly detection
            
            # Simple statistical anomaly detection
            if isinstance(metric_value, (int, float)):
                mean_val = sum(historical_values) / len(historical_values)
                variance = sum((x - mean_val) ** 2 for x in historical_values) / len(historical_values)
                std_dev = variance ** 0.5
                
                if std_dev > 0:
                    z_score = abs(metric_value - mean_val) / std_dev
                    # Convert z-score to anomaly score (0-1)
                    anomaly_score = min(1.0, z_score / 3.0)  # 3-sigma rule
                    return anomaly_score
                else:
                    # If std_dev is 0 (all values are the same), any different value is anomalous
                    if metric_value != mean_val:
                        return 1.0
            
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate anomaly score: {str(e)}")
            return 0.0
    
    async def _analyze_metric_trend(self, metric_name: str, metric_value: Any) -> str:
        """Analyze the trend for a metric based on recent history."""
        try:
            # Get recent values for trend analysis (handle metric name normalization)
            recent_values = []
            normalized_metric_name = self._normalize_metric_name(metric_name)
            
            for state_record in list(self._system_state_history)[-10:]:  # Last 10 events
                event_data = state_record.get("event", {}).get("data", {})
                if hasattr(event_data, 'name') and hasattr(event_data, 'value'):
                    event_metric_name = getattr(event_data, 'name')
                    normalized_event_name = self._normalize_metric_name(event_metric_name)
                    
                    # Match on normalized names
                    if normalized_event_name == normalized_metric_name:
                        recent_values.append(getattr(event_data, 'value'))
            
            if len(recent_values) < 2:
                return "insufficient_data"
            
            # Simple trend analysis
            if isinstance(metric_value, (int, float)) and all(isinstance(v, (int, float)) for v in recent_values):
                recent_values.append(metric_value)
                
                # Calculate trend over last few values
                if len(recent_values) >= 3:
                    first_half = sum(recent_values[:len(recent_values)//2]) / (len(recent_values)//2)
                    second_half = sum(recent_values[len(recent_values)//2:]) / (len(recent_values) - len(recent_values)//2)
                    
                    change_percent = ((second_half - first_half) / first_half * 100) if first_half != 0 else 0
                    
                    if change_percent > 10:
                        return "increasing"
                    elif change_percent < -10:
                        return "decreasing"
                    else:
                        return "stable"
            
            return "unknown"
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze metric trend: {str(e)}")
            return "error"
    
    async def _generate_and_store_state_embedding(self, metric_name: str, state_entry: Dict[str, Any]) -> None:
        """Generate and store vector embedding for state similarity search."""
        try:
            # Create a simple embedding based on metric characteristics
            # In a real implementation, this would use a proper embedding model
            embedding_features = []
            
            # Numeric value feature
            if isinstance(state_entry["value"], (int, float)):
                embedding_features.extend([
                    float(state_entry["value"]) / 100.0,  # Normalized value
                    state_entry["metadata"]["confidence"],
                    state_entry["metadata"]["anomaly_score"]
                ])
            else:
                # For non-numeric values, use hash-based features
                value_hash = hash(str(state_entry["value"])) % 1000 / 1000.0
                embedding_features.extend([value_hash, 0.5, 0.0])
            
            # Temporal features
            timestamp = datetime.fromisoformat(state_entry["timestamp"].replace('Z', '+00:00'))
            hour_of_day = timestamp.hour / 24.0
            day_of_week = timestamp.weekday() / 7.0
            embedding_features.extend([hour_of_day, day_of_week])
            
            # Source features
            source_hash = hash(state_entry["source"]) % 100 / 100.0
            embedding_features.append(source_hash)
            
            # Pad or truncate to fixed size
            target_size = 8
            if len(embedding_features) < target_size:
                embedding_features.extend([0.0] * (target_size - len(embedding_features)))
            else:
                embedding_features = embedding_features[:target_size]
            
            # Store embedding
            embedding_key = f"{metric_name}_{state_entry['event_id']}"
            self._state_embeddings[embedding_key] = {
                "embedding": embedding_features,
                "metric_name": metric_name,
                "timestamp": state_entry["timestamp"],
                "metadata": state_entry["metadata"]
            }
            
            # Limit embedding storage size
            if len(self._state_embeddings) > 1000:
                # Remove oldest embeddings
                oldest_keys = sorted(self._state_embeddings.keys())[:100]
                for key in oldest_keys:
                    del self._state_embeddings[key]
            
        except Exception as e:
            self.logger.warning(f"Failed to generate state embedding: {str(e)}")
    
    async def _validate_state_consistency(self, metric_name: str, state_entry: Dict[str, Any], previous_value: Any) -> None:
        """Validate state consistency and log inconsistencies."""
        try:
            consistency_check = {
                "metric_name": metric_name,
                "timestamp": state_entry["timestamp"],
                "current_value": state_entry["value"],
                "previous_value": previous_value,
                "consistency_score": 1.0,
                "issues": []
            }
            
            # Check for rapid value changes
            if (previous_value is not None and 
                isinstance(state_entry["value"], (int, float)) and 
                isinstance(previous_value, (int, float))):
                
                change_percent = abs(state_entry["value"] - previous_value) / max(abs(previous_value), 1) * 100
                if change_percent > 50:  # More than 50% change
                    consistency_check["consistency_score"] -= 0.3
                    consistency_check["issues"].append(f"Rapid value change: {change_percent:.1f}%")
            
            # Check for impossible values (domain-specific)
            if isinstance(state_entry["value"], (int, float)):
                if "cpu" in metric_name.lower() and (state_entry["value"] < 0 or state_entry["value"] > 100):
                    consistency_check["consistency_score"] -= 0.5
                    consistency_check["issues"].append("CPU value out of valid range (0-100)")
                elif "memory" in metric_name.lower() and (state_entry["value"] < 0 or state_entry["value"] > 100):
                    consistency_check["consistency_score"] -= 0.5
                    consistency_check["issues"].append("Memory value out of valid range (0-100)")
            
            # Store consistency check
            self._state_consistency_log.append(consistency_check)
            
            # Log significant consistency issues
            if consistency_check["consistency_score"] < 0.7:
                self.logger.warning(f"State consistency issue for {metric_name}: {consistency_check['issues']}")
            
        except Exception as e:
            self.logger.warning(f"Failed to validate state consistency: {str(e)}")
    
    async def _update_temporal_index(self, metric_name: str, state_entry: Dict[str, Any]) -> None:
        """Update temporal index for efficient historical queries."""
        try:
            timestamp = datetime.fromisoformat(state_entry["timestamp"].replace('Z', '+00:00'))
            
            # Create time-based keys for indexing
            hour_key = timestamp.strftime("%Y-%m-%d-%H")
            day_key = timestamp.strftime("%Y-%m-%d")
            
            # Update indices
            self._temporal_state_index[f"hour:{hour_key}"].append({
                "metric_name": metric_name,
                "value": state_entry["value"],
                "timestamp": state_entry["timestamp"],
                "event_id": state_entry["event_id"]
            })
            
            self._temporal_state_index[f"day:{day_key}"].append({
                "metric_name": metric_name,
                "value": state_entry["value"],
                "timestamp": state_entry["timestamp"],
                "event_id": state_entry["event_id"]
            })
            
            # Limit index size
            for key in list(self._temporal_state_index.keys()):
                if len(self._temporal_state_index[key]) > 100:
                    self._temporal_state_index[key] = self._temporal_state_index[key][-50:]
            
        except Exception as e:
            self.logger.warning(f"Failed to update temporal index: {str(e)}")
    
    async def _analyze_state_evolution_with_llm(self, metric_name: str, state_entry: Dict[str, Any], previous_value: Any) -> None:
        """Analyze state evolution using LLM reasoning to identify patterns."""
        try:
            if previous_value is None:
                return
            
            # Create evolution context for LLM analysis
            evolution_context = {
                "metric": metric_name,
                "previous_value": previous_value,
                "current_value": state_entry["value"],
                "timestamp": state_entry["timestamp"],
                "confidence": state_entry["metadata"]["confidence"],
                "anomaly_score": state_entry["metadata"]["anomaly_score"],
                "trend": state_entry["metadata"]["trend"]
            }
            
            # Simulate LLM analysis of state evolution
            pattern_analysis = await self._simulate_llm_pattern_analysis(evolution_context)
            
            # Store learned patterns
            pattern_key = f"{metric_name}_evolution"
            if pattern_key not in self._state_evolution_patterns:
                self._state_evolution_patterns[pattern_key] = []
            
            self._state_evolution_patterns[pattern_key].append({
                "analysis": pattern_analysis,
                "context": evolution_context,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            # Limit pattern storage
            if len(self._state_evolution_patterns[pattern_key]) > 50:
                self._state_evolution_patterns[pattern_key] = self._state_evolution_patterns[pattern_key][-25:]
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze state evolution with LLM: {str(e)}")
    
    async def _simulate_llm_pattern_analysis(self, evolution_context: Dict[str, Any]) -> str:
        """Simulate LLM analysis of state evolution patterns."""
        try:
            metric = evolution_context["metric"]
            prev_val = evolution_context["previous_value"]
            curr_val = evolution_context["current_value"]
            trend = evolution_context["trend"]
            anomaly_score = evolution_context["anomaly_score"]
            
            # Simple pattern analysis simulation
            if isinstance(prev_val, (int, float)) and isinstance(curr_val, (int, float)):
                change = curr_val - prev_val
                change_percent = (change / max(abs(prev_val), 1)) * 100
                
                if anomaly_score > 0.7:
                    return f"Anomalous change detected in {metric}: {change_percent:.1f}% change may indicate system stress"
                elif trend == "increasing" and change_percent > 20:
                    return f"Significant upward trend in {metric}: {change_percent:.1f}% increase suggests growing demand"
                elif trend == "decreasing" and change_percent < -20:
                    return f"Significant downward trend in {metric}: {change_percent:.1f}% decrease may indicate reduced load"
                else:
                    return f"Normal evolution pattern for {metric}: {change_percent:.1f}% change within expected range"
            else:
                return f"Non-numeric evolution in {metric}: changed from {prev_val} to {curr_val}"
            
        except Exception as e:
            return f"Pattern analysis failed: {str(e)}"
    
    async def _query_historical_state_with_embeddings(self, request: QueryRequest) -> tuple[str, float, str]:
        """Enhanced historical query using vector embeddings for similarity search."""
        try:
            # Get query parameters
            metric_name = request.parameters.get("metric", "")
            time_range = request.parameters.get("time_range", "1h")  # 1h, 1d, 1w
            similarity_threshold = float(request.parameters.get("similarity_threshold", "0.8"))
            
            if metric_name:
                # Query specific metric history with embeddings
                resolved_metric = self._resolve_metric_name(metric_name)
                if not resolved_metric:
                    return f"Metric '{metric_name}' not found", 0.0, "Metric resolution failed"
                
                # Find similar states using embeddings
                similar_states = await self._find_similar_states_by_embedding(
                    resolved_metric, similarity_threshold
                )
                
                # Get temporal history
                temporal_history = await self._get_temporal_history(resolved_metric, time_range)
                
                # Combine results
                result = {
                    "metric": metric_name,
                    "resolved_metric": resolved_metric,
                    "time_range": time_range,
                    "similar_states_count": len(similar_states),
                    "temporal_history_count": len(temporal_history),
                    "similar_states": similar_states[:5],  # Top 5 similar states
                    "recent_history": temporal_history[-10:]  # Last 10 temporal entries
                }
                
                confidence = 0.8 if len(similar_states) > 0 or len(temporal_history) > 0 else 0.3
                explanation = f"Historical analysis for {metric_name} using embedding similarity and temporal indexing"
                
                return json.dumps(result, indent=2), confidence, explanation
            
            else:
                # General historical overview
                total_embeddings = len(self._state_embeddings)
                total_temporal_entries = sum(len(entries) for entries in self._temporal_state_index.values())
                
                result = {
                    "historical_overview": {
                        "total_state_embeddings": total_embeddings,
                        "total_temporal_entries": total_temporal_entries,
                        "available_metrics": list(set(
                            emb["metric_name"] for emb in self._state_embeddings.values()
                        )),
                        "time_range_coverage": list(self._temporal_state_index.keys())[:10]
                    }
                }
                
                confidence = 0.7 if total_embeddings > 0 else 0.2
                explanation = "Historical overview using enhanced embedding and temporal indexing"
                
                return json.dumps(result, indent=2), confidence, explanation
            
        except Exception as e:
            return f"Error in enhanced historical query: {str(e)}", 0.0, "Enhanced historical query failed"
    
    async def _find_similar_states_by_embedding(self, metric_name: str, threshold: float) -> List[Dict[str, Any]]:
        """Find similar states using vector embedding similarity."""
        try:
            # Get current state embedding for comparison
            current_state = self._current_system_state.get(metric_name)
            if not current_state:
                return []
            
            # Find current state embedding
            current_embedding = None
            current_event_id = current_state.get("event_id")
            
            for emb_key, emb_data in self._state_embeddings.items():
                if (emb_data["metric_name"] == metric_name and 
                    current_event_id and current_event_id in emb_key):
                    current_embedding = emb_data["embedding"]
                    break
            
            if not current_embedding:
                return []
            
            # Calculate similarities
            similar_states = []
            for emb_key, emb_data in self._state_embeddings.items():
                if emb_data["metric_name"] == metric_name:
                    similarity = self._calculate_cosine_similarity(
                        current_embedding, emb_data["embedding"]
                    )
                    
                    if similarity >= threshold:
                        similar_states.append({
                            "embedding_key": emb_key,
                            "similarity": similarity,
                            "timestamp": emb_data["timestamp"],
                            "metadata": emb_data["metadata"]
                        })
            
            # Sort by similarity
            similar_states.sort(key=lambda x: x["similarity"], reverse=True)
            return similar_states
            
        except Exception as e:
            self.logger.warning(f"Failed to find similar states: {str(e)}")
            return []
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            if len(vec1) != len(vec2):
                return 0.0
            
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = sum(a * a for a in vec1) ** 0.5
            magnitude2 = sum(b * b for b in vec2) ** 0.5
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
            
        except Exception:
            return 0.0
    
    async def _get_temporal_history(self, metric_name: str, time_range: str) -> List[Dict[str, Any]]:
        """Get temporal history for a metric within specified time range."""
        try:
            now = datetime.now(timezone.utc)
            
            # Calculate time range
            if time_range == "1h":
                start_time = now - timedelta(hours=1)
                time_keys = [f"hour:{(now - timedelta(hours=i)).strftime('%Y-%m-%d-%H')}" for i in range(2)]
            elif time_range == "1d":
                start_time = now - timedelta(days=1)
                time_keys = [f"day:{(now - timedelta(days=i)).strftime('%Y-%m-%d')}" for i in range(2)]
            else:  # Default to 1h
                start_time = now - timedelta(hours=1)
                time_keys = [f"hour:{(now - timedelta(hours=i)).strftime('%Y-%m-%d-%H')}" for i in range(2)]
            
            # Collect matching entries
            matching_entries = []
            for time_key in time_keys:
                if time_key in self._temporal_state_index:
                    for entry in self._temporal_state_index[time_key]:
                        if entry["metric_name"] == metric_name:
                            entry_time = datetime.fromisoformat(entry["timestamp"].replace('Z', '+00:00'))
                            if entry_time >= start_time:
                                matching_entries.append(entry)
            
            # Sort by timestamp
            matching_entries.sort(key=lambda x: x["timestamp"])
            return matching_entries
            
        except Exception as e:
            self.logger.warning(f"Failed to get temporal history: {str(e)}")
            return []

    
    async def _simulate_forecast(self, request: SimulationRequest) -> tuple[list, float, tuple, str]:
        """Simulate forecast predictions using Gemini LLM reasoning.
        
        Generates predictions about future system states based on current trends
        and historical patterns without specific actions.
        
        Args:
            request: Simulation request with forecast parameters
            
        Returns:
            Tuple of (future_states, confidence, uncertainty_bounds, explanation)
        """
        try:
            self.logger.debug(f"Processing forecast simulation for {request.horizon_minutes} minutes")
            
            # Analyze current system trends
            current_trends = await self._analyze_current_trends()
            
            # Generate forecast based on trend extrapolation
            future_states = []
            horizon_steps = min(request.horizon_minutes // 5, 12)  # 5-minute intervals, max 12 steps
            
            for step in range(1, horizon_steps + 1):
                future_time = datetime.now(timezone.utc) + timedelta(minutes=step * 5)
                predicted_state = await self._extrapolate_state_from_trends(
                    current_trends, step * 5, request.parameters
                )
                
                future_states.append({
                    "timestamp": future_time.isoformat(),
                    "step": step,
                    "predicted_metrics": predicted_state,
                    "confidence": predicted_state.get("_confidence", 0.7)
                })
            
            # Calculate overall confidence based on trend stability
            overall_confidence = await self._calculate_forecast_confidence(current_trends, future_states)
            
            # Generate uncertainty bounds
            uncertainty_bounds = await self._calculate_forecast_uncertainty(current_trends, overall_confidence)
            
            # Generate explanation
            explanation = await self._generate_forecast_explanation(current_trends, future_states, overall_confidence)
            
            self.logger.debug(f"Forecast simulation completed with {len(future_states)} states, confidence: {overall_confidence:.2f}")
            
            return future_states, overall_confidence, uncertainty_bounds, explanation
            
        except Exception as e:
            self.logger.error(f"Forecast simulation failed: {str(e)}")
            return [], 0.0, (0.0, 1.0), f"Forecast simulation failed: {str(e)}"
    
    async def _simulate_what_if(self, request: SimulationRequest) -> tuple[list, float, tuple, str]:
        """Simulate "what-if" scenarios with specific actions using Gemini LLM reasoning.
        
        Predicts system behavior when specific actions are applied, considering
        action impacts and system responses.
        
        Args:
            request: Simulation request with actions to simulate
            
        Returns:
            Tuple of (future_states, confidence, uncertainty_bounds, explanation)
        """
        try:
            self.logger.debug(f"Processing what-if simulation with {len(request.actions)} actions")
            
            # Analyze current system state
            baseline_state = await self._capture_baseline_state()
            
            # Simulate action impacts
            future_states = []
            cumulative_effects = {}
            
            # Process actions sequentially with cumulative effects
            for i, action in enumerate(request.actions):
                action_time = datetime.now(timezone.utc) + timedelta(minutes=i * 2)  # 2-minute intervals
                
                # Predict immediate action impact
                action_impact = await self._predict_action_impact(action, baseline_state, cumulative_effects)
                
                # Apply cumulative effects
                cumulative_effects.update(action_impact.get("cumulative_changes", {}))
                
                future_states.append({
                    "timestamp": action_time.isoformat(),
                    "action_applied": {
                        "action_type": getattr(action, 'action_type', 'unknown'),
                        "action_id": getattr(action, 'action_id', f'action_{i}'),
                        "parameters": getattr(action, 'parameters', {})
                    },
                    "predicted_metrics": action_impact.get("predicted_state", {}),
                    "impact_summary": action_impact.get("impact_summary", ""),
                    "confidence": action_impact.get("confidence", 0.6)
                })
            
            # Extend simulation to show stabilization
            stabilization_steps = min((request.horizon_minutes - len(request.actions) * 2) // 5, 6)
            for step in range(1, stabilization_steps + 1):
                stabilization_time = future_states[-1]["timestamp"] if future_states else datetime.now(timezone.utc).isoformat()
                stabilization_time = datetime.fromisoformat(stabilization_time.replace('Z', '+00:00')) + timedelta(minutes=step * 5)
                
                stabilized_state = await self._predict_system_stabilization(
                    baseline_state, cumulative_effects, step * 5
                )
                
                future_states.append({
                    "timestamp": stabilization_time.isoformat(),
                    "phase": "stabilization",
                    "predicted_metrics": stabilized_state.get("predicted_state", {}),
                    "confidence": stabilized_state.get("confidence", 0.7)
                })
            
            # Calculate overall confidence
            overall_confidence = await self._calculate_what_if_confidence(request.actions, future_states)
            
            # Generate uncertainty bounds
            uncertainty_bounds = await self._calculate_what_if_uncertainty(request.actions, overall_confidence)
            
            # Generate explanation
            explanation = await self._generate_what_if_explanation(request.actions, future_states, overall_confidence)
            
            self.logger.debug(f"What-if simulation completed with {len(future_states)} states, confidence: {overall_confidence:.2f}")
            
            return future_states, overall_confidence, uncertainty_bounds, explanation
            
        except Exception as e:
            self.logger.error(f"What-if simulation failed: {str(e)}")
            return [], 0.0, (0.0, 1.0), f"What-if simulation failed: {str(e)}"
    
    async def _simulate_scenario(self, request: SimulationRequest) -> tuple[list, float, tuple, str]:
        """Simulate complex scenarios with multiple concurrent factors.
        
        Handles complex scenarios that may involve external factors,
        multiple concurrent actions, and system interactions.
        
        Args:
            request: Simulation request with scenario parameters
            
        Returns:
            Tuple of (future_states, confidence, uncertainty_bounds, explanation)
        """
        try:
            self.logger.debug(f"Processing scenario simulation: {request.parameters.get('scenario_name', 'unnamed')}")
            
            # Parse scenario parameters
            scenario_type = request.parameters.get("scenario_type", "load_test")
            external_factors = request.parameters.get("external_factors", {})
            concurrent_actions = request.parameters.get("concurrent_actions", "false").lower() == "true"
            
            # Initialize scenario context
            scenario_context = await self._initialize_scenario_context(scenario_type, external_factors)
            
            future_states = []
            
            if concurrent_actions and len(request.actions) > 1:
                # Process actions concurrently
                future_states = await self._simulate_concurrent_actions(
                    request.actions, scenario_context, request.horizon_minutes
                )
            else:
                # Process scenario with sequential actions and external factors
                future_states = await self._simulate_scenario_progression(
                    request.actions, scenario_context, request.horizon_minutes
                )
            
            # Calculate scenario-specific confidence
            overall_confidence = await self._calculate_scenario_confidence(scenario_type, future_states)
            
            # Generate uncertainty bounds considering external factors
            uncertainty_bounds = await self._calculate_scenario_uncertainty(
                scenario_type, external_factors, overall_confidence
            )
            
            # Generate comprehensive explanation
            explanation = await self._generate_scenario_explanation(
                scenario_type, external_factors, future_states, overall_confidence
            )
            
            self.logger.debug(f"Scenario simulation completed with {len(future_states)} states, confidence: {overall_confidence:.2f}")
            
            return future_states, overall_confidence, uncertainty_bounds, explanation
            
        except Exception as e:
            self.logger.error(f"Scenario simulation failed: {str(e)}")
            return [], 0.0, (0.0, 1.0), f"Scenario simulation failed: {str(e)}"
    
    async def _generate_impact_estimates(self, future_states: list, request: SimulationRequest) -> Dict[str, Any]:
        """Generate cost, performance, and reliability impact estimates.
        
        Analyzes predicted future states to estimate impacts on system
        cost, performance, and reliability metrics.
        
        Args:
            future_states: List of predicted future states
            request: Original simulation request
            
        Returns:
            Dictionary with impact estimates
        """
        try:
            if not future_states:
                return {"error": "No future states to analyze"}
            
            # Initialize impact categories
            impact_estimates = {
                "cost": {"current": 100, "predicted": 100, "change_percent": 0, "factors": []},
                "performance": {"current": 100, "predicted": 100, "change_percent": 0, "factors": []},
                "reliability": {"current": 95, "predicted": 95, "change_percent": 0, "factors": []}
            }
            
            # Analyze performance impacts from predicted metrics
            performance_impacts = await self._analyze_performance_impacts(future_states)
            impact_estimates["performance"].update(performance_impacts)
            
            # Analyze cost impacts based on resource usage predictions
            cost_impacts = await self._analyze_cost_impacts(future_states, request)
            impact_estimates["cost"].update(cost_impacts)
            
            # Analyze reliability impacts from anomaly predictions
            reliability_impacts = await self._analyze_reliability_impacts(future_states)
            impact_estimates["reliability"].update(reliability_impacts)
            
            # Add overall assessment
            impact_estimates["overall_assessment"] = await self._generate_overall_impact_assessment(impact_estimates)
            
            # Add confidence in impact estimates
            impact_estimates["confidence"] = await self._calculate_impact_estimate_confidence(future_states)
            
            return impact_estimates
            
        except Exception as e:
            self.logger.error(f"Failed to generate impact estimates: {str(e)}")
            return {"error": f"Impact estimation failed: {str(e)}"}
    
    # Helper methods for forecast simulation
    
    async def _analyze_current_trends(self) -> Dict[str, Any]:
        """Analyze current system trends for forecasting."""
        try:
            trends = {}
            
            for metric_name, state_data in self._current_system_state.items():
                if isinstance(state_data, dict) and "metadata" in state_data:
                    trend_info = {
                        "current_value": state_data["value"],
                        "trend_direction": state_data["metadata"]["trend"],
                        "anomaly_score": state_data["metadata"]["anomaly_score"],
                        "confidence": state_data["metadata"]["confidence"],
                        "stability": await self._calculate_metric_stability(metric_name)
                    }
                    trends[metric_name] = trend_info
            
            return trends
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze current trends: {str(e)}")
            return {}
    
    async def _calculate_metric_stability(self, metric_name: str) -> float:
        """Calculate stability score for a metric based on recent variance."""
        try:
            # Get recent values from history
            recent_values = []
            normalized_metric_name = self._normalize_metric_name(metric_name)
            
            for state_record in list(self._system_state_history)[-20:]:
                event_data = state_record.get("event", {}).get("data", {})
                if hasattr(event_data, 'name') and hasattr(event_data, 'value'):
                    event_metric_name = getattr(event_data, 'name')
                    if self._normalize_metric_name(event_metric_name) == normalized_metric_name:
                        recent_values.append(getattr(event_data, 'value'))
            
            if len(recent_values) < 3:
                return 0.5  # Neutral stability for insufficient data
            
            # Calculate coefficient of variation
            if all(isinstance(v, (int, float)) for v in recent_values):
                mean_val = sum(recent_values) / len(recent_values)
                if mean_val == 0:
                    return 1.0 if all(v == 0 for v in recent_values) else 0.0
                
                variance = sum((v - mean_val) ** 2 for v in recent_values) / len(recent_values)
                std_dev = variance ** 0.5
                cv = std_dev / abs(mean_val)
                
                # Convert CV to stability score (lower CV = higher stability)
                stability = max(0.0, min(1.0, 1.0 - cv))
                return stability
            
            return 0.5
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate metric stability: {str(e)}")
            return 0.5
    
    async def _extrapolate_state_from_trends(self, trends: Dict[str, Any], minutes_ahead: int, parameters: Dict[str, str]) -> Dict[str, Any]:
        """Extrapolate future state based on current trends."""
        try:
            predicted_state = {"_confidence": 0.7}
            
            for metric_name, trend_info in trends.items():
                current_value = trend_info["current_value"]
                trend_direction = trend_info["trend_direction"]
                stability = trend_info["stability"]
                
                if isinstance(current_value, (int, float)):
                    # Simple linear extrapolation based on trend
                    change_rate = 0.0
                    
                    if trend_direction == "increasing":
                        change_rate = 0.5 * (1 - stability)  # Less stable = more change
                    elif trend_direction == "decreasing":
                        change_rate = -0.5 * (1 - stability)
                    
                    # Apply time factor
                    time_factor = minutes_ahead / 60.0  # Convert to hours
                    predicted_change = change_rate * time_factor * current_value
                    
                    predicted_value = current_value + predicted_change
                    
                    # Apply bounds based on metric type
                    if "cpu" in metric_name.lower() or "memory" in metric_name.lower():
                        predicted_value = max(0, min(100, predicted_value))
                    elif "response" in metric_name.lower() or "latency" in metric_name.lower():
                        predicted_value = max(0, predicted_value)
                    
                    predicted_state[metric_name] = {
                        "value": predicted_value,
                        "change_from_current": predicted_change,
                        "confidence": trend_info["confidence"] * stability
                    }
                else:
                    # For non-numeric values, assume no change
                    predicted_state[metric_name] = {
                        "value": current_value,
                        "change_from_current": 0,
                        "confidence": 0.5
                    }
            
            return predicted_state
            
        except Exception as e:
            self.logger.warning(f"Failed to extrapolate state from trends: {str(e)}")
            return {"_confidence": 0.3, "error": str(e)}
    
    async def _calculate_forecast_confidence(self, trends: Dict[str, Any], future_states: list) -> float:
        """Calculate overall confidence for forecast simulation."""
        try:
            if not trends or not future_states:
                return 0.3
            
            # Base confidence on trend stability and data quality
            stability_scores = [trend["stability"] for trend in trends.values()]
            confidence_scores = [trend["confidence"] for trend in trends.values()]
            
            avg_stability = sum(stability_scores) / len(stability_scores) if stability_scores else 0.5
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
            
            # Reduce confidence for longer horizons
            horizon_penalty = max(0.1, 1.0 - len(future_states) * 0.05)
            
            overall_confidence = (avg_stability * 0.4 + avg_confidence * 0.4) * horizon_penalty + 0.2
            
            return max(0.1, min(0.95, overall_confidence))
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate forecast confidence: {str(e)}")
            return 0.5
    
    async def _calculate_forecast_uncertainty(self, trends: Dict[str, Any], confidence: float) -> tuple:
        """Calculate uncertainty bounds for forecast."""
        try:
            # Uncertainty increases with lower confidence and longer horizon
            base_uncertainty = 1.0 - confidence
            
            # Calculate uncertainty range
            uncertainty_range = base_uncertainty * 0.5  # ±50% of base uncertainty
            
            lower_bound = max(0.0, confidence - uncertainty_range)
            upper_bound = min(1.0, confidence + uncertainty_range)
            
            return (lower_bound, upper_bound)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate forecast uncertainty: {str(e)}")
            return (0.2, 0.8)
    
    async def _generate_forecast_explanation(self, trends: Dict[str, Any], future_states: list, confidence: float) -> str:
        """Generate human-readable explanation for forecast."""
        try:
            explanation_parts = [
                f"Forecast generated for {len(future_states)} time steps with {confidence:.1%} confidence."
            ]
            
            # Analyze key trends
            increasing_metrics = [name for name, trend in trends.items() if trend["trend_direction"] == "increasing"]
            decreasing_metrics = [name for name, trend in trends.items() if trend["trend_direction"] == "decreasing"]
            stable_metrics = [name for name, trend in trends.items() if trend["trend_direction"] == "stable"]
            
            if increasing_metrics:
                explanation_parts.append(f"Increasing trends detected in: {', '.join(increasing_metrics[:3])}")
            if decreasing_metrics:
                explanation_parts.append(f"Decreasing trends detected in: {', '.join(decreasing_metrics[:3])}")
            if stable_metrics:
                explanation_parts.append(f"Stable metrics: {len(stable_metrics)} metrics showing steady behavior")
            
            # Add confidence factors
            if confidence > 0.8:
                explanation_parts.append("High confidence due to stable trends and good data quality.")
            elif confidence > 0.6:
                explanation_parts.append("Moderate confidence with some trend variability.")
            else:
                explanation_parts.append("Lower confidence due to unstable trends or limited data.")
            
            return " ".join(explanation_parts)
            
        except Exception as e:
            return f"Forecast explanation generation failed: {str(e)}"
    
    # Helper methods for what-if simulation    
    async def _capture_baseline_state(self) -> Dict[str, Any]:
        """Capture current system state as baseline for what-if analysis."""
        try:
            baseline = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metrics": {},
                "metadata": {
                    "total_metrics": len(self._current_system_state),
                    "capture_method": "current_state_snapshot"
                }
            }
            
            for metric_name, state_data in self._current_system_state.items():
                if isinstance(state_data, dict) and "value" in state_data:
                    baseline["metrics"][metric_name] = {
                        "value": state_data["value"],
                        "confidence": state_data.get("metadata", {}).get("confidence", 0.7),
                        "trend": state_data.get("metadata", {}).get("trend", "unknown")
                    }
            
            return baseline
            
        except Exception as e:
            self.logger.warning(f"Failed to capture baseline state: {str(e)}")
            return {"timestamp": datetime.now(timezone.utc).isoformat(), "metrics": {}, "error": str(e)}
    
    async def _predict_action_impact(self, action: Any, baseline_state: Dict[str, Any], cumulative_effects: Dict[str, Any]) -> Dict[str, Any]:
        """Predict the impact of a specific action on system state."""
        try:
            # Extract action details
            action_type = getattr(action, 'action_type', 'unknown')
            action_params = getattr(action, 'parameters', {})
            
            impact_result = {
                "predicted_state": {},
                "cumulative_changes": cumulative_effects.copy(),
                "impact_summary": "",
                "confidence": 0.6
            }
            
            # Simulate different action types
            if action_type == "scale_up":
                impact_result = await self._simulate_scale_up_impact(action_params, baseline_state, cumulative_effects)
            elif action_type == "scale_down":
                impact_result = await self._simulate_scale_down_impact(action_params, baseline_state, cumulative_effects)
            elif action_type == "restart":
                impact_result = await self._simulate_restart_impact(action_params, baseline_state, cumulative_effects)
            elif action_type == "config_change":
                impact_result = await self._simulate_config_change_impact(action_params, baseline_state, cumulative_effects)
            else:
                # Generic action impact
                impact_result = await self._simulate_generic_action_impact(action_type, action_params, baseline_state, cumulative_effects)
            
            return impact_result
            
        except Exception as e:
            self.logger.warning(f"Failed to predict action impact: {str(e)}")
            return {
                "predicted_state": baseline_state.get("metrics", {}),
                "cumulative_changes": cumulative_effects,
                "impact_summary": f"Action impact prediction failed: {str(e)}",
                "confidence": 0.2
            }
    
    async def _simulate_scale_up_impact(self, params: Dict[str, Any], baseline: Dict[str, Any], cumulative: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate the impact of scaling up resources."""
        try:
            scale_factor = float(params.get("scale_factor", 1.5))
            
            predicted_state = {}
            impact_summary = f"Scaling up by factor of {scale_factor}"
            
            for metric_name, metric_data in baseline.get("metrics", {}).items():
                current_value = metric_data["value"]
                
                if isinstance(current_value, (int, float)):
                    if "cpu" in metric_name.lower():
                        # CPU usage should decrease with scale up
                        new_value = max(5, current_value / scale_factor)
                        predicted_state[metric_name] = {"value": new_value, "change": new_value - current_value}
                    elif "memory" in metric_name.lower():
                        # Memory usage should decrease with scale up
                        new_value = max(10, current_value / scale_factor)
                        predicted_state[metric_name] = {"value": new_value, "change": new_value - current_value}
                    elif "response" in metric_name.lower() or "latency" in metric_name.lower():
                        # Response time should improve (decrease)
                        new_value = max(10, current_value / scale_factor)
                        predicted_state[metric_name] = {"value": new_value, "change": new_value - current_value}
                    else:
                        # Other metrics remain relatively stable
                        predicted_state[metric_name] = {"value": current_value, "change": 0}
                else:
                    predicted_state[metric_name] = {"value": current_value, "change": 0}
            
            # Update cumulative effects
            cumulative["scale_factor"] = cumulative.get("scale_factor", 1.0) * scale_factor
            
            return {
                "predicted_state": predicted_state,
                "cumulative_changes": cumulative,
                "impact_summary": impact_summary,
                "confidence": 0.8
            }
            
        except Exception as e:
            return {
                "predicted_state": baseline.get("metrics", {}),
                "cumulative_changes": cumulative,
                "impact_summary": f"Scale up simulation failed: {str(e)}",
                "confidence": 0.3
            }
    
    async def _simulate_scale_down_impact(self, params: Dict[str, Any], baseline: Dict[str, Any], cumulative: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate the impact of scaling down resources."""
        try:
            scale_factor = float(params.get("scale_factor", 0.7))
            
            predicted_state = {}
            impact_summary = f"Scaling down by factor of {scale_factor}"
            
            for metric_name, metric_data in baseline.get("metrics", {}).items():
                current_value = metric_data["value"]
                
                if isinstance(current_value, (int, float)):
                    if "cpu" in metric_name.lower():
                        # CPU usage should increase with scale down
                        new_value = min(95, current_value / scale_factor)
                        predicted_state[metric_name] = {"value": new_value, "change": new_value - current_value}
                    elif "memory" in metric_name.lower():
                        # Memory usage should increase with scale down
                        new_value = min(90, current_value / scale_factor)
                        predicted_state[metric_name] = {"value": new_value, "change": new_value - current_value}
                    elif "response" in metric_name.lower() or "latency" in metric_name.lower():
                        # Response time should worsen (increase)
                        new_value = current_value / scale_factor
                        predicted_state[metric_name] = {"value": new_value, "change": new_value - current_value}
                    else:
                        predicted_state[metric_name] = {"value": current_value, "change": 0}
                else:
                    predicted_state[metric_name] = {"value": current_value, "change": 0}
            
            # Update cumulative effects
            cumulative["scale_factor"] = cumulative.get("scale_factor", 1.0) * scale_factor
            
            return {
                "predicted_state": predicted_state,
                "cumulative_changes": cumulative,
                "impact_summary": impact_summary,
                "confidence": 0.8
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to validate state consistency: {str(e)}")
            return {
                "predicted_state": baseline.get("metrics", {}),
                "cumulative_changes": cumulative,
                "impact_summary": f"Scale down simulation failed: {str(e)}",
                "confidence": 0.3
            }
    
    async def _simulate_llm_state_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate LLM analysis of state evolution (mock implementation)."""
        try:
            metric_name = context["metric"]
            current_value = context["current_value"]
            previous_value = context["previous_value"]
            anomaly_score = context["anomaly_score"]
            trend = context["trend"]
            
            analysis = {
                "significance": 0.5,  # Default significance
                "summary": "Normal state evolution",
                "insights": [],
                "recommendations": []
            }
            
            # Analyze significance based on various factors
            if anomaly_score > 0.7:
                analysis["significance"] = 0.9
                analysis["summary"] = f"Anomalous {metric_name} value detected"
                analysis["insights"].append(f"Value {current_value} is significantly different from historical pattern")
                analysis["recommendations"].append("Investigate potential system issues")
            
            elif trend == "increasing" and "cpu" in metric_name.lower():
                analysis["significance"] = 0.8
                analysis["summary"] = f"CPU utilization trending upward"
                analysis["insights"].append("System load appears to be increasing")
                analysis["recommendations"].append("Monitor for potential resource constraints")
            
            elif trend == "decreasing" and "response_time" in metric_name.lower():
                analysis["significance"] = 0.7
                analysis["summary"] = f"Response time improving"
                analysis["insights"].append("System performance optimization detected")
                analysis["recommendations"].append("Continue monitoring to confirm sustained improvement")
            
            # Add change magnitude analysis
            if (previous_value is not None and 
                isinstance(current_value, (int, float)) and 
                isinstance(previous_value, (int, float))):
                change_percent = abs(current_value - previous_value) / max(abs(previous_value), 1) * 100
                if change_percent > 25:
                    analysis["significance"] = max(analysis["significance"], 0.8)
                    analysis["insights"].append(f"Large change detected: {change_percent:.1f}%")
            
            return analysis
            
        except Exception as e:
            self.logger.warning(f"Failed to simulate LLM state analysis: {str(e)}")
            return {"significance": 0.0, "summary": "Analysis failed", "insights": [], "recommendations": []}
    
    # Simulation methods
    def _analyze_action_impact(self, actions: List[Any]) -> Dict[str, float]:
        """Analyze the potential impact of proposed actions."""
        impact = {"cpu_change": 0, "memory_change": 0, "response_time_change": 0}
        
        for action in actions:
            # Simple mock analysis based on action type
            if hasattr(action, 'action_type'):
                action_type = getattr(action, 'action_type', 'unknown')
                
                if action_type == "scale_up":
                    impact["cpu_change"] -= 10  # Reduce CPU usage
                    impact["memory_change"] += 5  # Increase memory usage
                    impact["response_time_change"] -= 20  # Improve response time
                elif action_type == "scale_down":
                    impact["cpu_change"] += 15  # Increase CPU usage
                    impact["memory_change"] -= 10  # Reduce memory usage
                    impact["response_time_change"] += 30  # Worsen response time
                elif action_type == "restart":
                    impact["cpu_change"] -= 5  # Temporary improvement
                    impact["memory_change"] -= 20  # Clear memory
                    impact["response_time_change"] += 10  # Temporary degradation
        
        return impact
    
    def _generate_high_load_scenario(self, horizon_minutes: int) -> List[Dict]:
        """Generate high load scenario states."""
        states = []
        for i in range(1, min(horizon_minutes // 10, 6)):
            time_offset = i * 10
            states.append({
                "time": f"+{time_offset}min",
                "cpu_usage": min(95, 85 + i * 2),
                "memory_usage": min(90, 75 + i * 3),
                "response_time": max(200, 150 + i * 25),
                "scenario": "high_load"
            })
        return states
    
    def _generate_failure_scenario(self, horizon_minutes: int) -> List[Dict]:
        """Generate failure scenario states."""
        states = []
        for i in range(1, min(horizon_minutes // 15, 4)):
            time_offset = i * 15
            states.append({
                "time": f"+{time_offset}min",
                "cpu_usage": 0 if i > 2 else max(0, 90 - i * 30),
                "memory_usage": 0 if i > 2 else max(0, 80 - i * 25),
                "response_time": 0 if i > 2 else max(500, 200 + i * 100),
                "scenario": "failure",
                "status": "failed" if i > 2 else "degrading"
            })
        return states
    
    def _generate_default_scenario(self, horizon_minutes: int) -> List[Dict]:
        """Generate default scenario states."""
        states = []
        for i in range(1, min(horizon_minutes // 20, 4)):
            time_offset = i * 20
            states.append({
                "time": f"+{time_offset}min",
                "cpu_usage": max(30, 60 - i * 5),
                "memory_usage": max(40, 55 + i * 2),
                "response_time": max(80, 110 - i * 5),
                "scenario": "default"
            })
        return states

    # Diagnostic methods    
    async def _generate_causal_hypotheses(self, request: DiagnosisRequest) -> List[str]:
        """Generate ranked causal hypotheses for the anomaly."""
        try:
            anomaly_desc = request.anomaly_description.lower()
            context = request.context
            
            # Simple rule-based hypothesis generation (mock LLM reasoning)
            hypotheses = []
            
            # CPU-related issues
            if "cpu" in anomaly_desc or "performance" in anomaly_desc:
                hypotheses.extend([
                    "High CPU utilization due to resource-intensive processes",
                    "CPU throttling due to thermal constraints",
                    "Inefficient algorithm causing excessive CPU usage",
                    "Memory leak leading to increased CPU overhead"
                ])
            
            # Memory-related issues
            if "memory" in anomaly_desc or "oom" in anomaly_desc:
                hypotheses.extend([
                    "Memory leak in application code",
                    "Insufficient memory allocation for workload",
                    "Memory fragmentation issues",
                    "Garbage collection pressure"
                ])
            
            # Network-related issues
            if "network" in anomaly_desc or "timeout" in anomaly_desc or "connection" in anomaly_desc:
                hypotheses.extend([
                    "Network congestion or bandwidth limitations",
                    "DNS resolution issues",
                    "Firewall or security policy blocking connections",
                    "Service discovery failures"
                ])
            
            # Response time issues
            if "slow" in anomaly_desc or "latency" in anomaly_desc or "response" in anomaly_desc:
                hypotheses.extend([
                    "Database query performance degradation",
                    "External service dependency slowdown",
                    "Resource contention in shared infrastructure",
                    "Inefficient caching strategy"
                ])
            
            # Error-related issues
            if "error" in anomaly_desc or "exception" in anomaly_desc or "failure" in anomaly_desc:
                hypotheses.extend([
                    "Configuration mismatch or invalid settings",
                    "Dependency version incompatibility",
                    "Resource exhaustion causing service failures",
                    "Race condition in concurrent processing"
                ])
            
            # Generic fallback hypotheses
            if not hypotheses:
                hypotheses = [
                    "System resource exhaustion",
                    "External dependency failure",
                    "Configuration drift",
                    "Unexpected workload pattern"
                ]
            
            # Limit to max hypotheses and add context-based ranking
            hypotheses = hypotheses[:self.max_hypotheses]
            
            # Add context-based weighting (mock implementation)
            if "high_priority" in context:
                hypotheses.insert(0, "Critical system component failure")
            
            return hypotheses
            
        except Exception as e:
            self.logger.error(f"Failed to generate hypotheses: {str(e)}")
            return ["Unknown root cause - analysis failed"]
    
    async def _identify_causal_chain(self, request: DiagnosisRequest, hypotheses: List[str]) -> str:
        """Identify the most likely causal chain."""
        try:
            if not hypotheses:
                return "No causal chain identified"
            
            # Use the top hypothesis to build a causal chain
            primary_hypothesis = hypotheses[0]
            
            # Simple causal chain construction based on hypothesis type
            if "cpu" in primary_hypothesis.lower():
                return "Resource demand increase → CPU saturation → Performance degradation → User impact"
            elif "memory" in primary_hypothesis.lower():
                return "Memory allocation growth → Memory pressure → GC overhead → System slowdown"
            elif "network" in primary_hypothesis.lower():
                return "Network issue → Connection failures → Service unavailability → Cascading failures"
            elif "database" in primary_hypothesis.lower():
                return "Query complexity increase → Database load → Response time degradation → Application slowdown"
            else:
                return f"Root cause: {primary_hypothesis} → System impact → Performance degradation"
                
        except Exception as e:
            self.logger.error(f"Failed to identify causal chain: {str(e)}")
            return "Causal chain analysis failed"
    
    async def _calculate_diagnostic_confidence(self, hypotheses: List[str]) -> float:
        """Calculate confidence score for diagnostic results."""
        try:
            if not hypotheses:
                return 0.0
            
            # Base confidence depends on number of hypotheses and available data
            base_confidence = 0.8
            
            # Reduce confidence if we have many hypotheses (less certain)
            hypothesis_penalty = min(0.2, len(hypotheses) * 0.05)
            
            # Adjust based on available historical data
            history_bonus = min(0.1, len(self._system_state_history) * 0.001)
            
            # Adjust based on calibration accuracy
            accuracy_bonus = (self._accuracy_metrics.get("overall", 0.8) - 0.8) * 0.2
            
            confidence = base_confidence - hypothesis_penalty + history_bonus + accuracy_bonus
            return max(0.1, min(0.95, confidence))
            
        except Exception as e:
            self.logger.error(f"Failed to calculate diagnostic confidence: {str(e)}")
            return 0.5
    
    async def _generate_diagnostic_explanation(
        self, 
        request: DiagnosisRequest, 
        hypotheses: List[str], 
        causal_chain: str
    ) -> str:
        """Generate human-readable explanation for diagnostic results."""
        try:
            anomaly_desc = request.anomaly_description
            timestamp = request.timestamp
            
            explanation_parts = [
                f"Analysis of anomaly: '{anomaly_desc}' at {timestamp}",
                f"Identified {len(hypotheses)} potential root causes:",
            ]
            
            # Add top hypotheses
            for i, hypothesis in enumerate(hypotheses[:3], 1):
                explanation_parts.append(f"  {i}. {hypothesis}")
            
            # Add causal chain
            explanation_parts.append(f"Most likely causal chain: {causal_chain}")
            
            # Add context information
            if request.context:
                context_info = ", ".join([f"{k}={v}" for k, v in request.context.items()])
                explanation_parts.append(f"Context factors: {context_info}")
            
            # Add recommendation
            explanation_parts.append(
                "Recommendation: Focus investigation on the primary hypothesis and monitor "
                "the identified causal chain for validation."
            )
            
            return "\n".join(explanation_parts)
            
        except Exception as e:
            self.logger.error(f"Failed to generate diagnostic explanation: {str(e)}")
            return f"Diagnostic explanation generation failed: {str(e)}"
    
    async def _gather_supporting_evidence(self, request: DiagnosisRequest, hypotheses: List[str]) -> List[str]:
        """Gather supporting evidence for diagnostic hypotheses."""
        try:
            evidence = []
            
            # Look for relevant evidence in system state history
            recent_events = list(self._system_state_history)[-10:]  # Last 10 events
            
            for event_record in recent_events:
                event = event_record.get("event", {})
                event_type = event.get("event_type", "")
                
                if event_type == "telemetry":
                    data = event.get("data", {})
                    if hasattr(data, 'name') and hasattr(data, 'value'):
                        metric_name = data.name
                        metric_value = data.value
                        
                        # Check if metric supports any hypothesis
                        if "cpu" in str(hypotheses).lower() and "cpu" in metric_name.lower():
                            evidence.append(f"CPU metric: {metric_name} = {metric_value}")
                        elif "memory" in str(hypotheses).lower() and "memory" in metric_name.lower():
                            evidence.append(f"Memory metric: {metric_name} = {metric_value}")
                        elif "response" in str(hypotheses).lower() and "response" in metric_name.lower():
                            evidence.append(f"Response metric: {metric_name} = {metric_value}")
                
                elif event_type == "anomaly":
                    evidence.append(f"Related anomaly detected at {event.get('timestamp', 'unknown')}")
            
            # Add calibration evidence
            if self._calibration_history:
                recent_accuracy = self._accuracy_metrics.get("overall", 0.8)
                evidence.append(f"Model accuracy: {recent_accuracy:.2f} based on recent calibrations")
            
            # Ensure we have some evidence
            if not evidence:
                evidence = [
                    "Limited historical data available for evidence gathering",
                    "Diagnosis based on pattern matching and rule-based analysis"
                ]
            
            return evidence[:5]  # Limit to 5 pieces of evidence
            
        except Exception as e:
            self.logger.error(f"Failed to gather supporting evidence: {str(e)}")
            return [f"Evidence gathering failed: {str(e)}"]

    # What-if simulation methods
    async def _simulate_restart_impact(self, params: Dict[str, Any], baseline: Dict[str, Any], cumulative: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate the impact of restarting a service or component."""
        try:
            component = params.get("component", "service")
            
            predicted_state = {}
            impact_summary = f"Restarting {component}"
            
            for metric_name, metric_data in baseline.get("metrics", {}).items():
                current_value = metric_data["value"]
                
                if isinstance(current_value, (int, float)):
                    if "cpu" in metric_name.lower():
                        # CPU usage spikes briefly then normalizes
                        new_value = min(100, current_value * 1.2)
                        predicted_state[metric_name] = {"value": new_value, "change": new_value - current_value}
                    elif "memory" in metric_name.lower():
                        # Memory usage resets to lower baseline
                        new_value = max(20, current_value * 0.6)
                        predicted_state[metric_name] = {"value": new_value, "change": new_value - current_value}
                    elif "response" in metric_name.lower() or "latency" in metric_name.lower():
                        # Response time temporarily increases during restart
                        new_value = current_value * 2.0
                        predicted_state[metric_name] = {"value": new_value, "change": new_value - current_value}
                    else:
                        predicted_state[metric_name] = {"value": current_value, "change": 0}
                else:
                    predicted_state[metric_name] = {"value": current_value, "change": 0}
            
            # Update cumulative effects
            cumulative["restarts"] = cumulative.get("restarts", 0) + 1
            
            return {
                "predicted_state": predicted_state,
                "cumulative_changes": cumulative,
                "impact_summary": impact_summary,
                "confidence": 0.7
            }
            
        except Exception as e:
            return {
                "predicted_state": baseline.get("metrics", {}),
                "cumulative_changes": cumulative,
                "impact_summary": f"Restart simulation failed: {str(e)}",
                "confidence": 0.3
            }
    
    async def _simulate_config_change_impact(self, params: Dict[str, Any], baseline: Dict[str, Any], cumulative: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate the impact of configuration changes."""
        try:
            config_type = params.get("config_type", "performance")
            change_magnitude = float(params.get("change_magnitude", 0.1))
            
            predicted_state = {}
            impact_summary = f"Configuration change: {config_type} (magnitude: {change_magnitude})"
            
            for metric_name, metric_data in baseline.get("metrics", {}).items():
                current_value = metric_data["value"]
                
                if isinstance(current_value, (int, float)):
                    if config_type == "performance":
                        if "cpu" in metric_name.lower() or "memory" in metric_name.lower():
                            # Performance config changes affect resource usage
                            new_value = current_value * (1 + change_magnitude)
                            predicted_state[metric_name] = {"value": new_value, "change": new_value - current_value}
                        elif "response" in metric_name.lower() or "latency" in metric_name.lower():
                            # Performance improvements reduce response time
                            new_value = current_value * (1 - change_magnitude)
                            predicted_state[metric_name] = {"value": new_value, "change": new_value - current_value}
                        else:
                            predicted_state[metric_name] = {"value": current_value, "change": 0}
                    else:
                        # Generic config change has minimal impact
                        new_value = current_value * (1 + change_magnitude * 0.5)
                        predicted_state[metric_name] = {"value": new_value, "change": new_value - current_value}
                else:
                    predicted_state[metric_name] = {"value": current_value, "change": 0}
            
            # Update cumulative effects
            cumulative["config_changes"] = cumulative.get("config_changes", 0) + 1
            
            return {
                "predicted_state": predicted_state,
                "cumulative_changes": cumulative,
                "impact_summary": impact_summary,
                "confidence": 0.6
            }
            
        except Exception as e:
            return {
                "predicted_state": baseline.get("metrics", {}),
                "cumulative_changes": cumulative,
                "impact_summary": f"Config change simulation failed: {str(e)}",
                "confidence": 0.3
            }
    
    async def _simulate_generic_action_impact(self, action_type: str, params: Dict[str, Any], baseline: Dict[str, Any], cumulative: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate the impact of generic actions."""
        try:
            impact_factor = float(params.get("impact_factor", 0.05))
            
            predicted_state = {}
            impact_summary = f"Generic action: {action_type} (impact: {impact_factor})"
            
            for metric_name, metric_data in baseline.get("metrics", {}).items():
                current_value = metric_data["value"]
                
                if isinstance(current_value, (int, float)):
                    # Apply small random impact
                    change = current_value * impact_factor * (1 if hash(action_type) % 2 else -1)
                    new_value = max(0, current_value + change)
                    predicted_state[metric_name] = {"value": new_value, "change": change}
                else:
                    predicted_state[metric_name] = {"value": current_value, "change": 0}
            
            # Update cumulative effects
            cumulative["generic_actions"] = cumulative.get("generic_actions", 0) + 1
            
            return {
                "predicted_state": predicted_state,
                "cumulative_changes": cumulative,
                "impact_summary": impact_summary,
                "confidence": 0.4
            }
            
        except Exception as e:
            return {
                "predicted_state": baseline.get("metrics", {}),
                "cumulative_changes": cumulative,
                "impact_summary": f"Generic action simulation failed: {str(e)}",
                "confidence": 0.2
            }
    
    async def _predict_system_stabilization(self, baseline_state: Dict[str, Any], cumulative_effects: Dict[str, Any], minutes_elapsed: int) -> Dict[str, Any]:
        """Predict how the system stabilizes after actions."""
        try:
            stabilization_factor = max(0.1, 1.0 - minutes_elapsed / 60.0)  # Stabilizes over 1 hour
            
            predicted_state = {}
            
            for metric_name, metric_data in baseline_state.get("metrics", {}).items():
                baseline_value = metric_data["value"]
                
                if isinstance(baseline_value, (int, float)):
                    # Apply stabilization toward baseline with cumulative effects
                    scale_effect = cumulative_effects.get("scale_factor", 1.0)
                    
                    if "cpu" in metric_name.lower():
                        target_value = baseline_value / scale_effect
                        stabilized_value = target_value + (baseline_value - target_value) * stabilization_factor
                    elif "memory" in metric_name.lower():
                        target_value = baseline_value / scale_effect
                        stabilized_value = target_value + (baseline_value - target_value) * stabilization_factor
                    elif "response" in metric_name.lower() or "latency" in metric_name.lower():
                        target_value = baseline_value / scale_effect
                        stabilized_value = target_value + (baseline_value - target_value) * stabilization_factor
                    else:
                        stabilized_value = baseline_value
                    
                    predicted_state[metric_name] = {
                        "value": stabilized_value,
                        "stabilization_progress": 1.0 - stabilization_factor
                    }
                else:
                    predicted_state[metric_name] = {"value": baseline_value, "stabilization_progress": 1.0}
            
            return {
                "predicted_state": predicted_state,
                "confidence": 0.7 * (1.0 - stabilization_factor) + 0.3  # Higher confidence as system stabilizes
            }
            
        except Exception as e:
            return {
                "predicted_state": baseline_state.get("metrics", {}),
                "confidence": 0.4,
                "error": str(e)
            }
    
    async def _calculate_what_if_confidence(self, actions: list, future_states: list) -> float:
        """Calculate confidence for what-if simulation."""
        try:
            if not actions or not future_states:
                return 0.3
            
            # Base confidence on action complexity and number of states
            action_complexity = len(actions) / 10.0  # Normalize to 0-1 range
            state_consistency = len(future_states) / max(len(actions) * 2, 1)  # Expected states per action
            
            # Higher confidence for simpler scenarios
            complexity_penalty = max(0.1, 1.0 - action_complexity)
            consistency_bonus = min(1.0, state_consistency)
            
            confidence = (complexity_penalty * 0.6 + consistency_bonus * 0.4) * 0.8 + 0.2
            
            return max(0.2, min(0.9, confidence))
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate what-if confidence: {str(e)}")
            return 0.5
    
    async def _calculate_what_if_uncertainty(self, actions: list, confidence: float) -> tuple:
        """Calculate uncertainty bounds for what-if simulation."""
        try:
            # Uncertainty increases with action complexity
            action_complexity = len(actions) / 5.0
            base_uncertainty = (1.0 - confidence) + action_complexity * 0.1
            
            uncertainty_range = base_uncertainty * 0.4
            
            lower_bound = max(0.0, confidence - uncertainty_range)
            upper_bound = min(1.0, confidence + uncertainty_range)
            
            return (lower_bound, upper_bound)
            
        except Exception as e:
            return (0.1, 0.9)
    
    async def _generate_what_if_explanation(self, actions: list, future_states: list, confidence: float) -> str:
        """Generate explanation for what-if simulation."""
        try:
            explanation_parts = [
                f"What-if simulation with {len(actions)} actions over {len(future_states)} time steps."
            ]
            
            # Summarize actions
            action_types = [getattr(action, 'action_type', 'unknown') for action in actions]
            action_summary = {}
            for action_type in action_types:
                action_summary[action_type] = action_summary.get(action_type, 0) + 1
            
            if action_summary:
                action_desc = ", ".join([f"{count} {action_type}" for action_type, count in action_summary.items()])
                explanation_parts.append(f"Actions simulated: {action_desc}.")
            
            # Add confidence assessment
            if confidence > 0.7:
                explanation_parts.append("High confidence in predictions due to well-understood action impacts.")
            elif confidence > 0.5:
                explanation_parts.append("Moderate confidence with some uncertainty in action interactions.")
            else:
                explanation_parts.append("Lower confidence due to complex action interactions or limited historical data.")
            
            return " ".join(explanation_parts)
            
        except Exception as e:
            return f"What-if explanation generation failed: {str(e)}"
    
    # Helper methods for scenario simulation    
    async def _initialize_scenario_context(self, scenario_type: str, external_factors: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize context for scenario simulation."""
        try:
            context = {
                "scenario_type": scenario_type,
                "external_factors": external_factors,
                "start_time": datetime.now(timezone.utc).isoformat(),
                "baseline_metrics": {}
            }
            
            # Capture baseline for scenario
            for metric_name, state_data in self._current_system_state.items():
                if isinstance(state_data, dict) and "value" in state_data:
                    context["baseline_metrics"][metric_name] = state_data["value"]
            
            # Apply external factors
            if "load_multiplier" in external_factors:
                context["load_multiplier"] = float(external_factors["load_multiplier"])
            if "network_latency" in external_factors:
                context["network_latency"] = float(external_factors["network_latency"])
            if "failure_rate" in external_factors:
                context["failure_rate"] = float(external_factors["failure_rate"])
            
            return context
            
        except Exception as e:
            return {
                "scenario_type": scenario_type,
                "external_factors": external_factors,
                "error": str(e)
            }
    
    async def _simulate_concurrent_actions(self, actions: list, scenario_context: Dict[str, Any], horizon_minutes: int) -> list:
        """Simulate multiple actions running concurrently."""
        try:
            future_states = []
            
            # Process actions in parallel time slices
            time_slices = min(horizon_minutes // 2, 10)  # 2-minute slices, max 10 slices
            
            for slice_idx in range(time_slices):
                slice_time = datetime.now(timezone.utc) + timedelta(minutes=slice_idx * 2)
                
                # Apply all actions concurrently for this time slice
                combined_effects = {}
                action_summaries = []
                
                for action in actions:
                    action_effect = await self._calculate_concurrent_action_effect(
                        action, scenario_context, slice_idx
                    )
                    
                    # Combine effects
                    for metric, effect in action_effect.get("metric_effects", {}).items():
                        if metric not in combined_effects:
                            combined_effects[metric] = {"value": 0, "factors": []}
                        combined_effects[metric]["value"] += effect.get("change", 0)
                        combined_effects[metric]["factors"].append(effect.get("factor", "unknown"))
                    
                    action_summaries.append(action_effect.get("summary", "unknown action"))
                
                # Apply external factors
                combined_effects = await self._apply_external_factors_to_effects(
                    combined_effects, scenario_context, slice_idx
                )
                
                # Create state for this time slice
                future_states.append({
                    "timestamp": slice_time.isoformat(),
                    "slice": slice_idx,
                    "concurrent_actions": action_summaries,
                    "combined_effects": combined_effects,
                    "confidence": 0.6 - slice_idx * 0.05  # Decreasing confidence over time
                })
            
            return future_states
            
        except Exception as e:
            self.logger.error(f"Concurrent action simulation failed: {str(e)}")
            return []
    
    async def _simulate_scenario_progression(self, actions: list, scenario_context: Dict[str, Any], horizon_minutes: int) -> list:
        """Simulate scenario progression with sequential actions and external factors."""
        try:
            future_states = []
            current_state = scenario_context.get("baseline_metrics", {}).copy()
            
            # Process scenario in phases
            phases = min(horizon_minutes // 5, 8)  # 5-minute phases, max 8 phases
            
            for phase_idx in range(phases):
                phase_time = datetime.now(timezone.utc) + timedelta(minutes=phase_idx * 5)
                
                # Apply actions for this phase
                if phase_idx < len(actions):
                    action = actions[phase_idx]
                    action_impact = await self._calculate_scenario_action_impact(
                        action, current_state, scenario_context, phase_idx
                    )
                    current_state.update(action_impact.get("new_state", {}))
                
                # Apply external factors progression
                external_impact = await self._calculate_external_factors_progression(
                    scenario_context, phase_idx, current_state
                )
                current_state.update(external_impact.get("modified_state", {}))
                
                # Create phase state
                future_states.append({
                    "timestamp": phase_time.isoformat(),
                    "phase": phase_idx,
                    "scenario_type": scenario_context["scenario_type"],
                    "predicted_metrics": current_state.copy(),
                    "external_factors_applied": external_impact.get("factors_applied", []),
                    "confidence": max(0.3, 0.8 - phase_idx * 0.06)
                })
            
            return future_states
            
        except Exception as e:
            self.logger.error(f"Scenario progression simulation failed: {str(e)}")
            return []
    
    async def _calculate_concurrent_action_effect(self, action: Any, context: Dict[str, Any], time_slice: int) -> Dict[str, Any]:
        """Calculate the effect of an action in concurrent execution."""
        try:
            action_type = getattr(action, 'action_type', 'unknown')
            action_params = getattr(action, 'parameters', {})
            
            # Time-based effect scaling
            time_factor = 1.0 - time_slice * 0.1  # Effects diminish over time
            
            metric_effects = {}
            
            if action_type == "scale_up":
                scale_factor = float(action_params.get("scale_factor", 1.5))
                metric_effects = {
                    "cpu.usage": {"change": -10 * time_factor, "factor": f"scale_up_{scale_factor}"},
                    "memory.usage": {"change": -8 * time_factor, "factor": f"scale_up_{scale_factor}"},
                    "response.time": {"change": -20 * time_factor, "factor": f"scale_up_{scale_factor}"}
                }
            elif action_type == "scale_down":
                scale_factor = float(action_params.get("scale_factor", 0.7))
                metric_effects = {
                    "cpu.usage": {"change": 15 * time_factor, "factor": f"scale_down_{scale_factor}"},
                    "memory.usage": {"change": 12 * time_factor, "factor": f"scale_down_{scale_factor}"},
                    "response.time": {"change": 30 * time_factor, "factor": f"scale_down_{scale_factor}"}
                }
            else:
                # Generic action
                metric_effects = {
                    "cpu.usage": {"change": 2 * time_factor, "factor": f"generic_{action_type}"},
                    "memory.usage": {"change": 1 * time_factor, "factor": f"generic_{action_type}"}
                }
            
            return {
                "metric_effects": metric_effects,
                "summary": f"{action_type} (concurrent, slice {time_slice})"
            }
            
        except Exception as e:
            return {
                "metric_effects": {},
                "summary": f"Action effect calculation failed: {str(e)}"
            }
    
    async def _apply_external_factors_to_effects(self, effects: Dict[str, Any], context: Dict[str, Any], time_slice: int) -> Dict[str, Any]:
        """Apply external factors to combined action effects."""
        try:
            external_factors = context.get("external_factors", {})
            
            # Apply load multiplier
            if "load_multiplier" in external_factors:
                load_mult = float(external_factors["load_multiplier"])
                for metric in effects:
                    if "cpu" in metric or "memory" in metric:
                        effects[metric]["value"] *= load_mult
                        effects[metric]["factors"].append(f"load_mult_{load_mult}")
            
            # Apply network latency
            if "network_latency" in external_factors:
                latency_mult = float(external_factors["network_latency"])
                for metric in effects:
                    if "response" in metric or "latency" in metric:
                        effects[metric]["value"] += latency_mult * 10  # 10ms per unit
                        effects[metric]["factors"].append(f"network_latency_{latency_mult}")
            
            # Apply failure rate
            if "failure_rate" in external_factors:
                failure_rate = float(external_factors["failure_rate"])
                if failure_rate > 0.1:  # Significant failure rate
                    for metric in effects:
                        effects[metric]["value"] *= (1 + failure_rate)
                        effects[metric]["factors"].append(f"failure_rate_{failure_rate}")
            
            return effects
            
        except Exception as e:
            self.logger.warning(f"Failed to apply external factors: {str(e)}")
            return effects
    
    async def _calculate_scenario_action_impact(self, action: Any, current_state: Dict[str, Any], context: Dict[str, Any], phase: int) -> Dict[str, Any]:
        """Calculate action impact within scenario context."""
        try:
            action_type = getattr(action, 'action_type', 'unknown')
            action_params = getattr(action, 'parameters', {})
            
            new_state = current_state.copy()
            
            # Apply action based on scenario type
            scenario_type = context.get("scenario_type", "generic")
            
            if scenario_type == "load_test":
                # In load test scenarios, actions have amplified effects
                amplification = 1.5
            elif scenario_type == "failure_recovery":
                # In failure recovery, actions are less predictable
                amplification = 0.8
            else:
                amplification = 1.0
            
            # Apply action with scenario amplification
            if action_type == "scale_up":
                scale_factor = float(action_params.get("scale_factor", 1.5)) * amplification
                for metric in new_state:
                    if isinstance(new_state[metric], (int, float)):
                        if "cpu" in metric.lower() or "memory" in metric.lower():
                            new_state[metric] = max(5, new_state[metric] / scale_factor)
                        elif "response" in metric.lower():
                            new_state[metric] = max(10, new_state[metric] / scale_factor)
            
            return {"new_state": new_state}
            
        except Exception as e:
            return {"new_state": current_state, "error": str(e)}
    
    async def _calculate_external_factors_progression(self, context: Dict[str, Any], phase: int, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate how external factors evolve over scenario phases."""
        try:
            external_factors = context.get("external_factors", {})
            modified_state = current_state.copy()
            factors_applied = []
            
            # Progressive load increase
            if "load_multiplier" in external_factors:
                base_load = float(external_factors["load_multiplier"])
                progressive_load = base_load * (1 + phase * 0.1)  # 10% increase per phase
                
                for metric in modified_state:
                    if isinstance(modified_state[metric], (int, float)):
                        if "cpu" in metric.lower() or "memory" in metric.lower():
                            modified_state[metric] *= progressive_load
                            factors_applied.append(f"progressive_load_{progressive_load:.1f}")
            
            # Network degradation over time
            if "network_latency" in external_factors:
                base_latency = float(external_factors["network_latency"])
                progressive_latency = base_latency * (1 + phase * 0.05)  # 5% increase per phase
                
                for metric in modified_state:
                    if isinstance(modified_state[metric], (int, float)):
                        if "response" in metric.lower() or "latency" in metric.lower():
                            modified_state[metric] += progressive_latency * 5
                            factors_applied.append(f"progressive_latency_{progressive_latency:.1f}")
            
            return {
                "modified_state": modified_state,
                "factors_applied": factors_applied
            }
            
        except Exception as e:
            return {
                "modified_state": current_state,
                "factors_applied": [f"error: {str(e)}"]
            }
    
    async def _calculate_scenario_confidence(self, scenario_type: str, future_states: list) -> float:
        """Calculate confidence for scenario simulation."""
        try:
            if not future_states:
                return 0.2
            
            # Base confidence varies by scenario type
            base_confidence = {
                "load_test": 0.8,      # Well-understood scenario
                "failure_recovery": 0.6, # More unpredictable
                "capacity_planning": 0.7, # Moderately predictable
                "generic": 0.5         # Unknown scenario type
            }.get(scenario_type, 0.5)
            
            # Reduce confidence for longer scenarios
            length_penalty = max(0.1, 1.0 - len(future_states) * 0.05)
            
            overall_confidence = base_confidence * length_penalty
            
            return max(0.2, min(0.9, overall_confidence))
            
        except Exception as e:
            return 0.4
    
    async def _calculate_scenario_uncertainty(self, scenario_type: str, external_factors: Dict[str, Any], confidence: float) -> tuple:
        """Calculate uncertainty bounds for scenario simulation."""
        try:
            # Base uncertainty from scenario complexity
            complexity_factor = len(external_factors) * 0.1
            scenario_uncertainty = {
                "load_test": 0.1,
                "failure_recovery": 0.3,
                "capacity_planning": 0.2,
                "generic": 0.25
            }.get(scenario_type, 0.25)
            
            total_uncertainty = scenario_uncertainty + complexity_factor
            uncertainty_range = total_uncertainty * 0.5
            
            lower_bound = max(0.0, confidence - uncertainty_range)
            upper_bound = min(1.0, confidence + uncertainty_range)
            
            return (lower_bound, upper_bound)
            
        except Exception as e:
            return (0.1, 0.9)
    
    async def _generate_scenario_explanation(self, scenario_type: str, external_factors: Dict[str, Any], future_states: list, confidence: float) -> str:
        """Generate explanation for scenario simulation."""
        try:
            explanation_parts = [
                f"Scenario simulation: {scenario_type} with {len(future_states)} phases."
            ]
            
            # Describe external factors
            if external_factors:
                factor_desc = ", ".join([f"{k}={v}" for k, v in external_factors.items()])
                explanation_parts.append(f"External factors: {factor_desc}.")
            
            # Scenario-specific insights
            if scenario_type == "load_test":
                explanation_parts.append("Load test scenario with progressive load increases and resource scaling.")
            elif scenario_type == "failure_recovery":
                explanation_parts.append("Failure recovery scenario with unpredictable system responses.")
            elif scenario_type == "capacity_planning":
                explanation_parts.append("Capacity planning scenario analyzing resource requirements over time.")
            
            # Confidence assessment
            if confidence > 0.7:
                explanation_parts.append("High confidence in scenario predictions.")
            elif confidence > 0.5:
                explanation_parts.append("Moderate confidence with some scenario complexity.")
            else:
                explanation_parts.append("Lower confidence due to scenario unpredictability.")
            
            return " ".join(explanation_parts)
            
        except Exception as e:
            return f"Scenario explanation generation failed: {str(e)}"
    
    # Helper methods for impact estimation    
    async def _analyze_performance_impacts(self, future_states: list) -> Dict[str, Any]:
        """Analyze performance impacts from predicted states."""
        try:
            if not future_states:
                return {"predicted": 100, "change_percent": 0, "factors": ["no_data"]}
            
            # Extract performance metrics from future states
            response_times = []
            cpu_usage = []
            
            for state in future_states:
                metrics = state.get("predicted_metrics", {})
                for metric_name, metric_data in metrics.items():
                    if isinstance(metric_data, dict) and "value" in metric_data:
                        value = metric_data["value"]
                    else:
                        value = metric_data
                    
                    if isinstance(value, (int, float)):
                        if "response" in metric_name.lower() or "latency" in metric_name.lower():
                            response_times.append(value)
                        elif "cpu" in metric_name.lower():
                            cpu_usage.append(value)
            
            # Calculate performance score (lower response time and CPU = better performance)
            current_performance = 100  # Baseline
            
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                # Assume baseline response time of 100ms, performance degrades linearly
                response_impact = max(0, 100 - (avg_response_time - 100) * 0.5)
            else:
                response_impact = 100
            
            if cpu_usage:
                avg_cpu = sum(cpu_usage) / len(cpu_usage)
                # Performance degrades as CPU usage increases
                cpu_impact = max(0, 100 - avg_cpu * 0.5)
            else:
                cpu_impact = 100
            
            predicted_performance = (response_impact + cpu_impact) / 2
            change_percent = ((predicted_performance - current_performance) / current_performance) * 100
            
            factors = []
            if response_times:
                factors.append(f"avg_response_time_{sum(response_times)/len(response_times):.1f}ms")
            if cpu_usage:
                factors.append(f"avg_cpu_{sum(cpu_usage)/len(cpu_usage):.1f}%")
            
            return {
                "current": current_performance,
                "predicted": predicted_performance,
                "change_percent": change_percent,
                "factors": factors
            }
            
        except Exception as e:
            return {
                "current": 100,
                "predicted": 100,
                "change_percent": 0,
                "factors": [f"analysis_error: {str(e)}"]
            }
    
    async def _analyze_cost_impacts(self, future_states: list, request: SimulationRequest) -> Dict[str, Any]:
        """Analyze cost impacts based on resource usage predictions."""
        try:
            if not future_states:
                return {"predicted": 100, "change_percent": 0, "factors": ["no_data"]}
            
            # Extract resource usage metrics
            cpu_usage = []
            memory_usage = []
            scale_factors = []
            
            for state in future_states:
                # Check for scaling actions
                if "action_applied" in state:
                    action = state["action_applied"]
                    if action.get("action_type") == "scale_up":
                        scale_factor = action.get("parameters", {}).get("scale_factor", 1.0)
                        scale_factors.append(float(scale_factor))
                
                # Extract resource metrics
                metrics = state.get("predicted_metrics", {})
                for metric_name, metric_data in metrics.items():
                    if isinstance(metric_data, dict) and "value" in metric_data:
                        value = metric_data["value"]
                    else:
                        value = metric_data
                    
                    if isinstance(value, (int, float)):
                        if "cpu" in metric_name.lower():
                            cpu_usage.append(value)
                        elif "memory" in metric_name.lower():
                            memory_usage.append(value)
            
            # Calculate cost impact
            current_cost = 100  # Baseline cost
            
            # Scale-up increases cost
            if scale_factors:
                avg_scale_factor = sum(scale_factors) / len(scale_factors)
                scale_cost_impact = avg_scale_factor * 100  # Linear cost scaling
            else:
                scale_cost_impact = 100
            
            # High resource usage may trigger auto-scaling (cost increase)
            if cpu_usage or memory_usage:
                avg_cpu = sum(cpu_usage) / len(cpu_usage) if cpu_usage else 50
                avg_memory = sum(memory_usage) / len(memory_usage) if memory_usage else 50
                
                # If average usage > 80%, assume auto-scaling triggers
                if avg_cpu > 80 or avg_memory > 80:
                    usage_cost_impact = 120  # 20% cost increase
                else:
                    usage_cost_impact = 100
            else:
                usage_cost_impact = 100
            
            predicted_cost = max(scale_cost_impact, usage_cost_impact)
            change_percent = ((predicted_cost - current_cost) / current_cost) * 100
            
            factors = []
            if scale_factors:
                factors.append(f"scaling_factor_{sum(scale_factors)/len(scale_factors):.1f}")
            if cpu_usage:
                factors.append(f"avg_cpu_{sum(cpu_usage)/len(cpu_usage):.1f}%")
            if memory_usage:
                factors.append(f"avg_memory_{sum(memory_usage)/len(memory_usage):.1f}%")
            
            return {
                "current": current_cost,
                "predicted": predicted_cost,
                "change_percent": change_percent,
                "factors": factors
            }
            
        except Exception as e:
            return {
                "current": 100,
                "predicted": 100,
                "change_percent": 0,
                "factors": [f"cost_analysis_error: {str(e)}"]
            }
    
    async def _analyze_reliability_impacts(self, future_states: list) -> Dict[str, Any]:
        """Analyze reliability impacts from anomaly predictions."""
        try:
            if not future_states:
                return {"predicted": 95, "change_percent": 0, "factors": ["no_data"]}
            
            # Count reliability risk factors
            high_cpu_count = 0
            high_memory_count = 0
            high_response_time_count = 0
            restart_count = 0
            
            for state in future_states:
                # Check for restart actions (temporary reliability impact)
                if "action_applied" in state:
                    action = state["action_applied"]
                    if "restart" in action.get("action_type", "").lower():
                        restart_count += 1
                
                # Check for high resource usage (reliability risk)
                metrics = state.get("predicted_metrics", {})
                for metric_name, metric_data in metrics.items():
                    if isinstance(metric_data, dict) and "value" in metric_data:
                        value = metric_data["value"]
                    else:
                        value = metric_data
                    
                    if isinstance(value, (int, float)):
                        if "cpu" in metric_name.lower() and value > 90:
                            high_cpu_count += 1
                        elif "memory" in metric_name.lower() and value > 85:
                            high_memory_count += 1
                        elif ("response" in metric_name.lower() or "latency" in metric_name.lower()) and value > 1000:
                            high_response_time_count += 1
            
            # Calculate reliability score
            current_reliability = 95  # Baseline 95% reliability
            
            # Penalties for risk factors
            reliability_penalty = 0
            reliability_penalty += high_cpu_count * 2      # 2% per high CPU state
            reliability_penalty += high_memory_count * 2   # 2% per high memory state
            reliability_penalty += high_response_time_count * 1  # 1% per slow response
            reliability_penalty += restart_count * 3       # 3% per restart (temporary)
            
            predicted_reliability = max(70, current_reliability - reliability_penalty)
            change_percent = ((predicted_reliability - current_reliability) / current_reliability) * 100
            
            factors = []
            if high_cpu_count > 0:
                factors.append(f"high_cpu_states_{high_cpu_count}")
            if high_memory_count > 0:
                factors.append(f"high_memory_states_{high_memory_count}")
            if high_response_time_count > 0:
                factors.append(f"slow_response_states_{high_response_time_count}")
            if restart_count > 0:
                factors.append(f"restart_actions_{restart_count}")
            
            if not factors:
                factors.append("no_reliability_risks")
            
            return {
                "current": current_reliability,
                "predicted": predicted_reliability,
                "change_percent": change_percent,
                "factors": factors
            }
            
        except Exception as e:
            return {
                "current": 95,
                "predicted": 95,
                "change_percent": 0,
                "factors": [f"reliability_analysis_error: {str(e)}"]
            }
    
    async def _generate_overall_impact_assessment(self, impact_estimates: Dict[str, Any]) -> str:
        """Generate overall assessment of predicted impacts."""
        try:
            assessments = []
            
            # Performance assessment
            perf_change = impact_estimates.get("performance", {}).get("change_percent", 0)
            if perf_change > 10:
                assessments.append("significant performance improvement expected")
            elif perf_change < -10:
                assessments.append("performance degradation expected")
            else:
                assessments.append("minimal performance impact")
            
            # Cost assessment
            cost_change = impact_estimates.get("cost", {}).get("change_percent", 0)
            if cost_change > 20:
                assessments.append("substantial cost increase expected")
            elif cost_change > 5:
                assessments.append("moderate cost increase expected")
            else:
                assessments.append("minimal cost impact")
            
            # Reliability assessment
            rel_change = impact_estimates.get("reliability", {}).get("change_percent", 0)
            if rel_change < -5:
                assessments.append("reliability concerns identified")
            elif rel_change > 2:
                assessments.append("reliability improvement expected")
            else:
                assessments.append("reliability maintained")
            
            return "; ".join(assessments).capitalize() + "."
            
        except Exception as e:
            return f"Overall impact assessment failed: {str(e)}"
    
    async def _calculate_impact_estimate_confidence(self, future_states: list) -> float:
        """Calculate confidence in impact estimates."""
        try:
            if not future_states:
                return 0.3
            
            # Base confidence on number of states and data quality
            state_count_factor = min(1.0, len(future_states) / 5.0)  # Optimal around 5 states
            
            # Check data completeness
            states_with_metrics = sum(1 for state in future_states if state.get("predicted_metrics"))
            completeness_factor = states_with_metrics / len(future_states) if future_states else 0
            
            confidence = (state_count_factor * 0.4 + completeness_factor * 0.6) * 0.8 + 0.2
            
            return max(0.2, min(0.9, confidence))
            
        except Exception as e:
            return 0.5
        
# Register the Gemini World Model with the factory
def register_gemini_world_model():
    """Register the Gemini World Model implementation with the factory."""
    from .world_model import WorldModelFactory
    WorldModelFactory.register("gemini", GeminiWorldModel)
    WorldModelFactory.register("llm", GeminiWorldModel)  # Alias for LLM-based models


# Auto-register when module is imported
register_gemini_world_model()   
 