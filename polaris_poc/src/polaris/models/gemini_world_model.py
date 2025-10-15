"""
Gemini LLM-based World Model implementation for POLARIS Digital Twin.

This module implements the World Model interface using Google's Gemini LLM
with proper API integration, providing intelligent state tracking, predictive
simulation, and diagnostic reasoning capabilities.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from collections import deque, defaultdict
import uuid

# Google Generative AI imports
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from .world_model import (
    WorldModel, WorldModelError, WorldModelInitializationError, WorldModelOperationError,
    QueryRequest, QueryResponse, SimulationRequest, SimulationResponse,
    DiagnosisRequest, DiagnosisResponse
)
from .digital_twin_events import KnowledgeEvent, CalibrationEvent


class GeminiWorldModel(WorldModel):
    """
    Gemini LLM-based World Model implementation.
    
    This implementation uses Google's Gemini LLM API to provide intelligent
    state tracking, predictive simulation, and diagnostic reasoning with
    proper API integration and error handling.
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
        self.model_name = config.get("model", "gemini-2.5-flash")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 2048)
        self.top_p = config.get("top_p", 0.9)
        self.top_k = config.get("top_k", 40)
        
        # Request configuration
        self.concurrent_requests = config.get("concurrent_requests", 5)
        self.request_timeout_sec = config.get("request_timeout_sec", 30)
        self.retry_attempts = config.get("retry_attempts", 3)
        self.retry_delay_sec = config.get("retry_delay_sec", 1)
        
        # Prompt templates
        prompts = config.get("prompts", {})
        self.system_prompt = prompts.get("system_prompt", self._get_default_system_prompt())
        self.query_prompt = prompts.get("query_prompt", self._get_default_query_prompt())
        self.simulation_prompt = prompts.get("simulation_prompt", self._get_default_simulation_prompt())
        self.diagnosis_prompt = prompts.get("diagnosis_prompt", self._get_default_diagnosis_prompt())
        
        # State management
        self.max_history_events = config.get("max_history_events", 1000)
        self.max_conversation_memory = config.get("max_conversation_memory", 50)
        
        # Internal state
        self._model = None
        self._generation_config = None
        self._safety_settings = None
        self._conversation_memory = deque(maxlen=self.max_conversation_memory)
        self._system_state_history = deque(maxlen=self.max_history_events)
        self._calibration_history = deque(maxlen=100)
        self._current_system_state = {}
        self._accuracy_metrics = {"overall": 0.8, "last_updated": None}
        
        # Rate limiting and concurrency
        self._request_semaphore = asyncio.Semaphore(self.concurrent_requests)
        self._last_request_time = 0
        self._min_request_interval = 1.0 / self.concurrent_requests  # Prevent rate limiting
        
        self.logger.info(f"Initialized GeminiWorldModel with model: {self.model_name}")
    
    async def initialize(self) -> None:
        """Initialize the Gemini World Model implementation.
        
        Sets up the Gemini API client and prepares the model configuration.
        
        Raises:
            WorldModelInitializationError: If initialization fails
        """
        try:
            self.logger.info("Initializing Gemini World Model...")
            
            # Check for API key with interactive prompt
            api_key = os.getenv(self.api_key_env)
            if not api_key:
                # Try to get API key interactively
                try:
                    from polaris.common.api_key_manager import get_gemini_api_key_interactive
                    self.logger.info("API key not found in environment, prompting user...")
                    api_key = get_gemini_api_key_interactive("Gemini World Model")
                    if not api_key:
                        raise WorldModelInitializationError(
                            f"Gemini API key not found in environment variable: {self.api_key_env}"
                        )
                    # Set the API key in environment for this session
                    os.environ[self.api_key_env] = api_key
                    self.logger.info("✅ Gemini API key obtained interactively")
                except ImportError:
                    raise WorldModelInitializationError(
                        f"Gemini API key not found in environment variable: {self.api_key_env}. "
                        f"Install API key manager dependencies: pip install keyring cryptography"
                    )
            
            # Configure Gemini API
            genai.configure(api_key=api_key)
            
            # Initialize the model
            self._model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=self.system_prompt
            )
            
            # Configure generation parameters
            self._generation_config = genai.types.GenerationConfig(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_output_tokens=self.max_tokens,
                response_mime_type="text/plain"
            )
            
            # Configure safety settings (allow most content for system analysis)
            self._safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            }
            
            # Test the connection
            await self._test_api_connection()
            
            # Initialize conversation memory
            self._add_to_conversation_memory("system", "Gemini World Model initialized and ready for system analysis.")
            
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
    
    async def _test_api_connection(self) -> None:
        """Test the Gemini API connection."""
        try:
            test_prompt = "Test connection. Respond with 'OK' if you can process this message."
            
            # Use a shorter timeout for connection test
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._model.generate_content,
                    test_prompt,
                    generation_config=self._generation_config,
                    safety_settings=self._safety_settings
                ),
                timeout=10.0  # Shorter timeout for connection test
            )
            
            if not response or not response.text or len(response.text.strip()) == 0:
                raise WorldModelInitializationError("API test returned empty response")
            
            self.logger.info(f"API connection test successful: {response.text[:50]}...")
            
        except Exception as e:
            raise WorldModelInitializationError(f"API connection test failed: {str(e)}") from e
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt."""
        return """You are an expert system analyst for the POLARIS adaptive system framework.
You have deep knowledge of system monitoring, performance analysis, and root cause diagnosis.
You understand distributed systems, performance metrics, and adaptation strategies.

Your responses should be:
- Accurate and based on provided data
- Actionable with specific recommendations
- Clear and concise for technical audiences
- Include confidence levels for your assessments

Always consider system safety and stability in your recommendations.
Provide confidence scores (0.0 to 1.0) for all your assessments."""
    
    def _get_default_query_prompt(self) -> str:
        """Get the default query prompt template."""
        return """Based on the following system information, answer this query: {query}

System Context:
{context}

Recent Telemetry Data:
{telemetry}

Please provide:
1. A clear, concise answer to the query
2. Supporting evidence from the telemetry data
3. Your confidence level (0.0-1.0)
4. Any relevant warnings or considerations"""
    
    def _get_default_simulation_prompt(self) -> str:
        """Get the default simulation prompt template."""
        return """Simulate the following scenario for the adaptive system:

Proposed Actions: {actions}
Time Horizon: {horizon_minutes} minutes
Current System State: {current_state}

Predict the likely outcomes and provide:
1. Expected system performance changes
2. Resource utilization impacts
3. Potential risks and mitigation strategies
4. Expected benefits and improvements
5. Confidence level in predictions (0.0-1.0)
6. Recommended monitoring points during execution

Consider both immediate effects and longer-term implications."""
    
    def _get_default_diagnosis_prompt(self) -> str:
        """Get the default diagnosis prompt template."""
        return """You are a system diagnostics engine. Analyze the following technical anomaly.

Anomaly Description: {anomaly}
System Context: {context}
Recent System Events: {events}

Provide a root cause analysis based *only* on the provided data. Your response must be in JSON format.
1.  **Likely Root Causes**: List the most probable technical causes (e.g., "database connection pool exhaustion," "memory leak in service X").
2.  **Supporting Evidence**: Link each hypothesis to specific data points from the context or events.
3.  **Recommended Next Steps**: Suggest concrete technical actions to confirm the diagnosis (e.g., "check database logs for errors," "inspect heap dump of service X").
4.  **Confidence Score**: Provide an overall confidence level (0.0-1.0) in the analysis.

Focus strictly on technical, data-driven hypotheses."""    

    async def _generate_response(self, prompt: str, max_retries: Optional[int] = None) -> str:
        """Generate a response from the Gemini model with rate limiting and retries.
        
        Args:
            prompt: The prompt to send to the model
            max_retries: Maximum number of retries (uses config default if None)
            
        Returns:
            Generated response text
            
        Raises:
            WorldModelOperationError: If generation fails after retries
        """
        if not self._model:
            raise WorldModelOperationError("Model not initialized")
        
        max_retries = max_retries or self.retry_attempts
        
        async with self._request_semaphore:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            if time_since_last < self._min_request_interval:
                await asyncio.sleep(self._min_request_interval - time_since_last)
            
            for attempt in range(max_retries):
                try:
                    self._last_request_time = time.time()
                    
                    # Generate response using asyncio.to_thread to avoid blocking
                    response = await asyncio.wait_for(
                        asyncio.to_thread(
                            self._model.generate_content,
                            prompt,
                            generation_config=self._generation_config,
                            safety_settings=self._safety_settings
                        ),
                        timeout=self.request_timeout_sec
                    )
                    
                    if not response.parts:
                        finish_reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"
                        safety_ratings = response.candidates[0].safety_ratings if response.candidates else "N/A"
                        error_message = (
                            f"Gemini API returned no content. Finish Reason: {finish_reason}. "
                            f"Safety Ratings: {safety_ratings}"
                        )
                        self.logger.error(error_message)
                        raise WorldModelOperationError(error_message)

                    if not response or not response.text:
                        raise WorldModelOperationError("Empty response from Gemini API")
                    
                    return response.text.strip()
                    
                except asyncio.TimeoutError:
                    self.logger.warning(f"Gemini API timeout on attempt {attempt + 1}")
                    if attempt == max_retries - 1:
                        raise WorldModelOperationError("Gemini API timeout after retries")
                    
                except Exception as e:
                    self.logger.warning(f"Gemini API error on attempt {attempt + 1}: {str(e)}")
                    if attempt == max_retries - 1:
                        raise WorldModelOperationError(f"Gemini API failed after retries: {str(e)}")
                    
                    # Exponential backoff
                    await asyncio.sleep(self.retry_delay_sec * (2 ** attempt))
            
            raise WorldModelOperationError("Failed to generate response after all retries")
    
    def _add_to_conversation_memory(self, role: str, content: str) -> None:
        """Add a message to conversation memory."""
        self._conversation_memory.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def _get_conversation_context(self, max_messages: int = 10) -> str:
        """Get recent conversation context as a formatted string."""
        recent_messages = list(self._conversation_memory)[-max_messages:]
        context_parts = []
        
        for msg in recent_messages:
            role = msg["role"]
            content = msg["content"][:500]  # Limit content length
            timestamp = msg["timestamp"]
            context_parts.append(f"[{timestamp}] {role.upper()}: {content}")
        
        return "\n".join(context_parts)
    
    def _format_system_state(self) -> str:
        """Format current system state for LLM consumption."""
        if not self._current_system_state:
            return "No current system state available."
        
        state_parts = []
        for key, value in self._current_system_state.items():
            if isinstance(value, dict):
                if "value" in value and "timestamp" in value:
                    state_parts.append(f"{key}: {value['value']} (at {value['timestamp']})")
                else:
                    state_parts.append(f"{key}: {json.dumps(value)}")
            else:
                state_parts.append(f"{key}: {value}")
        
        return "\n".join(state_parts[:20])  # Limit to 20 most recent items
    
    def _format_recent_events(self, max_events: int = 10) -> str:
        """Format recent system events for LLM consumption."""
        if not self._system_state_history:
            return "No recent system events available."
        
        recent_events = list(self._system_state_history)[-max_events:]
        event_parts = []
        
        for event_record in recent_events:
            event = event_record.get("event", {})
            timestamp = event.get("timestamp", "unknown")
            event_type = event.get("event_type", "unknown")
            source = event.get("source", "unknown")
            
            event_parts.append(f"[{timestamp}] {event_type} from {source}")
        
        return "\n".join(event_parts)
    
    async def update_state(self, event: KnowledgeEvent) -> None:
        """Update the world model state with new knowledge.
        
        Args:
            event: Knowledge event containing system state updates
        """
        try:
            self.logger.debug(f"Processing knowledge event: {event.event_type} from {event.source}")
            
            # Add event to history
            self._system_state_history.append({
                "event": event.to_dict(),
                "processed_at": datetime.now(timezone.utc).isoformat()
            })
            
            # Update current system state based on event type
            if event.event_type == "telemetry":
                await self._process_telemetry_event(event)
            elif event.event_type == "execution_status":
                await self._process_execution_event(event)
            elif event.event_type == "anomaly":
                await self._process_anomaly_event(event)
            
            # Add to conversation memory for context
            self._add_to_conversation_memory(
                "user",
                f"System update: {event.event_type} from {event.source}"
            )
            
            self.logger.debug(f"Successfully processed knowledge event: {event.event_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to update state with event {event.event_id}: {str(e)}")
            raise WorldModelOperationError(f"State update failed: {str(e)}") from e
    
    async def calibrate(self, event: CalibrationEvent) -> None:
        """Calibrate the world model based on prediction accuracy feedback.
        
        Args:
            event: Calibration event with accuracy feedback
        """
        try:
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
            
            # Add to conversation memory for learning context
            calibration_context = (
                f"Calibration feedback: Prediction accuracy {accuracy_score:.2f}. "
                f"Learning from prediction vs actual outcome differences."
            )
            
            self._add_to_conversation_memory("user", calibration_context)
            
            self.logger.info(f"Calibration complete: accuracy={accuracy_score:.2f}")
            
        except Exception as e:
            self.logger.error(f"Failed to calibrate with event {event.calibration_id}: {str(e)}")
            raise WorldModelOperationError(f"Calibration failed: {str(e)}") from e
    
    async def query_state(self, request: QueryRequest) -> QueryResponse:
        """Query the current or historical system state using Gemini LLM.
        
        Args:
            request: Query request with type and parameters
            
        Returns:
            Query response with results and confidence
        """
        try:
            self.logger.debug(f"Processing query: {request.query_type} - {request.query_content}")
            
            # Build context for the query
            context = self._get_conversation_context()
            telemetry = self._format_system_state()
            
            # Format the prompt based on query type
            if request.query_type == "natural_language":
                prompt = self.query_prompt.format(
                    query=request.query_content,
                    context=context,
                    telemetry=telemetry
                )
            else:
                # For current_state and historical queries
                prompt = f"""Query Type: {request.query_type}
Query: {request.query_content}
Parameters: {json.dumps(request.parameters)}

Current System State:
{telemetry}

Recent Context:
{context}

Please provide a detailed response including:
1. Direct answer to the query
2. Supporting evidence from the data
3. Confidence level (0.0-1.0)
4. Any relevant observations or warnings

Format your response as JSON with fields: answer, evidence, confidence, warnings"""
            
            # Generate response using Gemini
            llm_response = await self._generate_response(prompt)
            
            # Parse the response
            result, confidence, explanation = self._parse_query_response(llm_response, request.query_type)
            
            # Add to conversation memory
            self._add_to_conversation_memory("user", f"Query: {request.query_content}")
            self._add_to_conversation_memory("assistant", result[:200] + "..." if len(result) > 200 else result)
            
            response = QueryResponse(
                query_id=request.query_id,
                success=True,
                result=result,
                confidence=confidence,
                explanation=explanation,
                metadata={
                    "query_type": request.query_type,
                    "model": self.model_name,
                    "llm_response_length": len(llm_response)
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
        """Perform predictive simulation using Gemini LLM.
        
        Args:
            request: Simulation request with actions and parameters
            
        Returns:
            Simulation response with predictions and uncertainty
        """
        try:
            async with self._request_semaphore:
                self.logger.debug(f"Processing simulation: {request.simulation_type} - {request.simulation_id}")
                
                # Format actions for the prompt
                actions_text = []
                for i, action in enumerate(request.actions):
                    if hasattr(action, 'action_type'):
                        action_text = f"Action {i+1}: {action.action_type}"
                        if hasattr(action, 'parameters'):
                            action_text += f" with parameters: {json.dumps(action.parameters)}"
                    else:
                        action_text = f"Action {i+1}: {json.dumps(action)}"
                    actions_text.append(action_text)
                
                actions_str = "\n".join(actions_text) if actions_text else "No specific actions provided"
                
                # Build the simulation prompt
                prompt = self.simulation_prompt.format(
                    actions=actions_str,
                    horizon_minutes=request.horizon_minutes,
                    current_state=self._format_system_state()
                )
                
                # Add simulation type specific instructions
                if request.simulation_type == "forecast":
                    prompt += "\n\nFocus on predicting natural system evolution without external interventions."
                elif request.simulation_type == "what_if":
                    prompt += "\n\nFocus on the specific impacts of the proposed actions."
                elif request.simulation_type == "scenario":
                    prompt += "\n\nConsider complex interactions and multiple factors in this scenario."
                
                prompt += "\n\nProvide your response in JSON format with fields: future_states, confidence, uncertainty_lower, uncertainty_upper, explanation, impact_estimates"
                
                # Generate response using Gemini
                llm_response = await self._generate_response(prompt)
                
                # Parse the simulation response
                future_states, confidence, uncertainty_bounds, explanation, impact_estimates = self._parse_simulation_response(llm_response)
                
                # Add to conversation memory
                self._add_to_conversation_memory("user", f"Simulation: {request.simulation_type} with {len(request.actions)} actions")
                self._add_to_conversation_memory("assistant", f"Predicted {len(future_states)} future states with {confidence:.2f} confidence")
                
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
                        "model": self.model_name,
                        "llm_response_length": len(llm_response)
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
        """Perform root cause analysis using Gemini LLM.
        
        Args:
            request: Diagnosis request with anomaly description
            
        Returns:
            Diagnosis response with hypotheses and explanations
        """
        try:
            self.logger.debug(f"Processing diagnosis: {request.diagnosis_id}")
            
            # Format context information
            context_str = json.dumps(request.context, indent=2) if request.context else "No additional context provided"
            events_str = self._format_recent_events()
            
            # Build the diagnosis prompt
            prompt = self.diagnosis_prompt.format(
                anomaly=request.anomaly_description,
                context=context_str,
                events=events_str
            )
            
            prompt += "\n\nProvide your response in JSON format with fields: hypotheses, causal_chain, confidence, explanation, supporting_evidence"
            
            # Generate response using Gemini
            llm_response = await self._generate_response(prompt)
            
            # Parse the diagnosis response
            hypotheses, causal_chain, confidence, explanation, supporting_evidence = self._parse_diagnosis_response(llm_response)
            
            # Add to conversation memory
            self._add_to_conversation_memory("user", f"Diagnosis request: {request.anomaly_description}")
            self._add_to_conversation_memory("assistant", f"Identified {len(hypotheses)} hypotheses with {confidence:.2f} confidence")
            
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
                    "model": self.model_name,
                    "llm_response_length": len(llm_response)
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
    
    def _parse_query_response(self, llm_response: str, query_type: str) -> Tuple[str, float, str]:
        """Parse LLM response for query operations."""
        try:
            # Try to parse as JSON first
            if llm_response.strip().startswith('{'):
                response_data = json.loads(llm_response)
                result = response_data.get("answer", llm_response)
                confidence = float(response_data.get("confidence", 0.7))
                explanation = response_data.get("evidence", "") + " " + response_data.get("warnings", "")
            else:
                # Parse structured text response
                result = llm_response
                confidence = self._extract_confidence_from_text(llm_response)
                explanation = f"LLM analysis for {query_type} query"
            
            return result, confidence, explanation.strip()
            
        except Exception as e:
            self.logger.warning(f"Failed to parse query response: {e}")
            return llm_response, 0.6, "Response parsing failed, using raw LLM output"
    
    def _parse_simulation_response(self, llm_response: str) -> Tuple[List[Dict], float, Tuple[float, float], str, Dict]:
        """Parse LLM response for simulation operations."""
        try:
            # Try to parse as JSON
            if llm_response.strip().startswith('{'):
                response_data = json.loads(llm_response)
                
                future_states = response_data.get("future_states", [])
                confidence = float(response_data.get("confidence", 0.7))
                uncertainty_lower = float(response_data.get("uncertainty_lower", confidence - 0.1))
                uncertainty_upper = float(response_data.get("uncertainty_upper", confidence + 0.1))
                explanation = response_data.get("explanation", "LLM simulation analysis")
                impact_estimates = response_data.get("impact_estimates", {})
                
            else:
                # Parse structured text and create default response
                future_states = self._extract_future_states_from_text(llm_response)
                confidence = self._extract_confidence_from_text(llm_response)
                uncertainty_lower = max(0.0, confidence - 0.15)
                uncertainty_upper = min(1.0, confidence + 0.15)
                explanation = llm_response[:500] + "..." if len(llm_response) > 500 else llm_response
                impact_estimates = {"cost_impact": 0, "performance_impact": 0, "reliability_impact": 0}
            
            return future_states, confidence, (uncertainty_lower, uncertainty_upper), explanation, impact_estimates
            
        except Exception as e:
            self.logger.warning(f"Failed to parse simulation response: {e}")
            return [], 0.5, (0.3, 0.7), llm_response, {}
    
    def _parse_diagnosis_response(self, llm_response: str) -> Tuple[List[str], str, float, str, List[str]]:
        """Parse LLM response for diagnosis operations."""
        try:
            # Try to parse as JSON
            if llm_response.strip().startswith('{'):
                response_data = json.loads(llm_response)
                
                hypotheses = response_data.get("hypotheses", [])
                causal_chain = response_data.get("causal_chain", "")
                confidence = float(response_data.get("confidence", 0.7))
                explanation = response_data.get("explanation", "LLM diagnosis analysis")
                supporting_evidence = response_data.get("supporting_evidence", [])
                
            else:
                # Parse structured text
                hypotheses = self._extract_hypotheses_from_text(llm_response)
                causal_chain = self._extract_causal_chain_from_text(llm_response)
                confidence = self._extract_confidence_from_text(llm_response)
                explanation = llm_response[:500] + "..." if len(llm_response) > 500 else llm_response
                supporting_evidence = ["LLM analysis based on system data"]
            
            return hypotheses, causal_chain, confidence, explanation, supporting_evidence
            
        except Exception as e:
            self.logger.warning(f"Failed to parse diagnosis response: {e}")
            return ["Analysis failed"], "Unknown", 0.3, llm_response, []
    
    def _extract_confidence_from_text(self, text: str) -> float:
        """Extract confidence score from text response."""
        import re
        
        # Look for confidence patterns
        patterns = [
            r'confidence[:\s]+([0-9]*\.?[0-9]+)',
            r'([0-9]*\.?[0-9]+)\s*confidence',
            r'([0-9]*\.?[0-9]+)%\s*confident',
            r'confidence.*?([0-9]*\.?[0-9]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    value = float(match.group(1))
                    # Convert percentage to decimal if needed
                    if value > 1.0:
                        value = value / 100.0
                    return max(0.0, min(1.0, value))
                except ValueError:
                    continue
        
        # Default confidence based on response quality
        if len(text) > 100 and "analysis" in text.lower():
            return 0.7
        elif len(text) > 50:
            return 0.6
        else:
            return 0.5
    
    def _extract_future_states_from_text(self, text: str) -> List[Dict]:
        """Extract future states from text response."""
        # Simple extraction - look for time-based predictions
        states = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            if any(time_word in line.lower() for time_word in ['minute', 'hour', 'time', 'future', 'predict']):
                states.append({
                    "timestamp": f"+{i*5}min",
                    "description": line.strip(),
                    "confidence": 0.7
                })
        
        # Ensure we have at least one state
        if not states:
            states.append({
                "timestamp": "+10min",
                "description": "System state prediction based on LLM analysis",
                "confidence": 0.6
            })
        
        return states[:5]  # Limit to 5 states
    
    def _extract_hypotheses_from_text(self, text: str) -> List[str]:
        """Extract hypotheses from text response."""
        hypotheses = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for numbered lists or bullet points
            if (line and (line[0].isdigit() or line.startswith('-') or line.startswith('*')) 
                and len(line) > 10):
                # Clean up the hypothesis
                hypothesis = re.sub(r'^[\d\.\-\*\s]+', '', line).strip()
                if hypothesis:
                    hypotheses.append(hypothesis)
        
        # If no structured hypotheses found, create from content
        if not hypotheses and text:
            sentences = text.split('.')
            for sentence in sentences[:3]:  # Take first 3 sentences
                sentence = sentence.strip()
                if len(sentence) > 20:
                    hypotheses.append(sentence)
        
        return hypotheses[:5]  # Limit to 5 hypotheses
    
    def _extract_causal_chain_from_text(self, text: str) -> str:
        """Extract causal chain from text response."""
        # Look for causal indicators
        causal_patterns = [
            r'causal chain[:\s]+(.*?)(?:\n|$)',
            r'root cause[:\s]+(.*?)(?:\n|$)',
            r'leads to[:\s]+(.*?)(?:\n|$)',
            r'causes?[:\s]+(.*?)(?:\n|$)'
        ]
        
        for pattern in causal_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1).strip()
        
        # Default causal chain
        return "Analysis indicates system issue requiring investigation"
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get the current health status of the World Model."""
        try:
            # Test API connectivity
            api_healthy = False
            api_error = None
            
            if self._model:
                try:
                    test_response = await asyncio.wait_for(
                        self._generate_response("Health check - respond with 'OK'"),
                        timeout=10.0
                    )
                    api_healthy = len(test_response.strip()) > 0
                except Exception as e:
                    api_error = str(e)
            
            return {
                "status": "healthy" if (self.is_initialized and api_healthy) else "unhealthy",
                "model_type": "gemini_llm_real",
                "model_name": self.model_name,
                "last_check": datetime.now(timezone.utc).isoformat(),
                "api_connectivity": {
                    "healthy": api_healthy,
                    "error": api_error,
                    "last_request_time": self._last_request_time
                },
                "metrics": {
                    "overall_accuracy": self._accuracy_metrics.get("overall", 0.8),
                    "events_processed": len(self._system_state_history),
                    "calibrations_received": len(self._calibration_history),
                    "conversation_memory_size": len(self._conversation_memory),
                    "current_state_metrics": len(self._current_system_state),
                    "concurrent_capacity": self.concurrent_requests,
                    "request_semaphore_available": self._request_semaphore._value
                },
                "configuration": {
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "request_timeout_sec": self.request_timeout_sec,
                    "retry_attempts": self.retry_attempts,
                    "max_history_events": self.max_history_events
                },
                "api_status": {
                    "model_initialized": self._model is not None,
                    "generation_config_set": self._generation_config is not None,
                    "safety_settings_configured": self._safety_settings is not None
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "model_type": "gemini_llm_real",
                "error": str(e),
                "last_check": datetime.now(timezone.utc).isoformat()
            }
    
    async def _process_telemetry_event(self, event: KnowledgeEvent) -> None:
        """Process telemetry events and update system state."""
        try:
            telemetry_data = event.data
            
            # Extract key metrics from telemetry
            if hasattr(telemetry_data, 'name') and hasattr(telemetry_data, 'value'):
                metric_name = telemetry_data.name
                metric_value = telemetry_data.value
                
                # Update current system state
                self._current_system_state[metric_name] = {
                    "value": metric_value,
                    "timestamp": event.timestamp,
                    "source": event.source,
                    "event_id": event.event_id
                }
                
                self.logger.debug(f"Updated metric: {metric_name} = {metric_value}")
            
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
    
    def _set_initialized(self, initialized: bool) -> None:
        """Set the initialization status."""
        self._initialized = initialized
    
    @property
    def is_initialized(self) -> bool:
        """Check if the model is initialized."""
        return self._initialized
    
    async def reload_model(self) -> bool:
        """Reload the World Model implementation.
        
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
 
# Register the Gemini World Model with the factory
def register_gemini_world_model():
    """Register the Gemini World Model implementation with the factory."""
    from .world_model import WorldModelFactory
    WorldModelFactory.register("gemini", GeminiWorldModel)
    WorldModelFactory.register("llm", GeminiWorldModel)  # Alias for LLM-based models


# Auto-register when module is imported
register_gemini_world_model()