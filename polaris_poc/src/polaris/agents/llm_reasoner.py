"""
LLM-based Reasoner Implementation

Single class that adds LLM-powered reasoning implementation to process
telemetry data and generate action outputs without relying on KB or Digital Twin.
Includes convenient prompt modification capabilities.
"""

import json
import asyncio
import time
from typing import Any, Dict, List, Optional, Union
import logging
import openai
from openai import AsyncOpenAI
from google import genai
from pathlib import Path

# Add these imports to the existing reasoner_agent.py file
from .reasoner_core import (
    ReasoningInterface, 
    ReasoningContext, 
    ReasoningResult, 
    ReasoningType,
)

from .reasoner_agent import ReasonerAgent
from google.genai import types


class LLMReasoningImplementation(ReasoningInterface):
    """LLM-powered reasoning implementation with customizable prompts."""
    
    def __init__(self, 
                 api_key: str,
                 reasoning_type: ReasoningType,
                 model: str = "gemini-2.5-pro",
                 max_tokens: int = 1000,
                 temperature: float = 0.7,
                 timeout: float = 30.0,
                 base_url: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        
        self.api_key = api_key
        self.reasoning_type = reasoning_type
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.logger = logger or logging.getLogger(f"LLMReasoner.{reasoning_type.value}")
        self.logger.info(f"Initializing LLMReasoningImplementation with model {model}")
        # Initialize OpenAI client
        self.client = genai.Client(api_key=api_key)
        
        # Prompt templates - easily customizable
        self.system_prompt_template = """You are an intelligent self adaptive system controller.
Your task is to analyze telemetry data and generate appropriate control actions. You are a proactive reasoner, not handling sudden spikes but long term system goals. Dont take 
actions too frequently, only when really needed.

Context: {context_description}
Reasoning Type: {reasoning_type}
Current Timestamp: {timestamp}

Guidelines:
1. Analyze the provided telemetry data carefully
2. Consider system state and trends
3. Generate specific, actionable control decisions
4. Provide clear reasoning for your decisions
5. Output must be a valid JSON action format

Available Action Types:
- SET_DIMMER: Control lighting dimmer values (0.0 to 1.0)
- ADD_SERVER: Add server instance
- REMOVE_SERVER: Remove server instance

Output Format:
{{
    "action_type": "ACTION_TYPE",
    "target": "device_id_or_system",
    "params": {{"key": "value"}},
    "reasoning": "Clear explanation of why this action was chosen",
    "confidence": 0.95
}}

Examples:

{{
                    "action_type": "SET_DIMMER",
                    "source": "fast_controller",
                    "action_id": str(uuid.uuid4()),
                    "params": {{"value": new_dimmer}},
                    "priority": "low",
}}

{{
                    "action_type": "REMOVE_SERVER",
                    "source": "fast_controller",
                    "action_id": str(uuid.uuid4()),
                    "params": {{"server_type": "compute", "count": 1}},
                    "priority": "low",
}}

{{
                    "action_type": "ADD_SERVER",
                    "source": "fast_controller",
                    "action_id": str(uuid.uuid4()),
                    "params": {{"server_type": "compute", "count": 1}},
                    "priority": "low",
}}




"""

        self.user_prompt_template = """Please analyze the following telemetry data and generate an appropriate control action:

Telemetry Data:
{telemetry_data}

Input Context:
{input_data}

Additional Metadata:
{metadata}

Generate a control action based on this information. """
        
        # Customizable fields
        self.custom_instructions = ""
        self.domain_context = ""
        self.safety_constraints = [
            "Do not generate actions that could cause system damage",
            "Prioritize safety over efficiency",
            "Validate all parameter ranges before output"
        ]
        
        self.call_history: List[Dict[str, Any]] = []
    
    async def reason(self, 
                    context: ReasoningContext, 
                    knowledge: Optional[List[Dict[str, Any]]] = None) -> ReasoningResult:
        """Execute LLM-based reasoning on the telemetry data."""
        start_time = time.time()
        reasoning_steps = ["Starting LLM-based reasoning"]
        
        try:
            # Prepare telemetry data from context
            telemetry_data = self._extract_telemetry_data(context)
            reasoning_steps.append(f"Extracted {len(telemetry_data)} telemetry points")
            
            # Build prompts
            system_prompt = self._build_system_prompt(context)
            user_prompt = self._build_user_prompt(context, telemetry_data)
            reasoning_steps.append("Built LLM prompts")
            
            # Make LLM call
            llm_response = await self._call_llm(system_prompt, user_prompt)
            reasoning_steps.append("Received LLM response")
            
            # Parse and validate response
            action_result = self._parse_llm_response(llm_response)
            reasoning_steps.append("Parsed and validated LLM output")
            
            # Store call history for debugging
            self._store_call_history(context, system_prompt, user_prompt, llm_response, action_result)
            
            execution_time = time.time() - start_time
            
            return ReasoningResult(
                result=action_result,
                confidence=action_result.get("confidence", 0.8),
                reasoning_steps=reasoning_steps,
                context=context,
                execution_time=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"LLM reasoning failed: {e}", exc_info=True)
            execution_time = time.time() - start_time
            
            return ReasoningResult(
                result={
                    "error": str(e),
                    "action_type": "ALERT",
                    "target": "system",
                    "params": {"message": f"LLM reasoning failed: {e}"},
                    "reasoning": "Fallback due to LLM failure"
                },
                confidence=0.1,
                reasoning_steps=reasoning_steps + [f"Error: {str(e)}"],
                context=context,
                execution_time=execution_time
            )
    
    async def validate_input(self, context: ReasoningContext) -> bool:
        """Validate that we have sufficient telemetry data for LLM reasoning."""
        
        
        return True
    
    def get_required_knowledge_types(self, context: ReasoningContext) -> List[str]:
        """LLM reasoner doesn't require KB knowledge."""
        return []
    
    def extract_search_terms(self, context: ReasoningContext) -> List[str]:
        """LLM reasoner doesn't require KB search."""
        return []
    
    # ==== Prompt Customization Methods ====
    
    def set_system_prompt(self, template: str):
        """Update the system prompt template."""
        self.system_prompt_template = template
        return self
    
    def set_user_prompt(self, template: str):
        """Update the user prompt template."""
        self.user_prompt_template = template
        return self
    
    def add_custom_instructions(self, instructions: str):
        """Add custom instructions to the system prompt."""
        self.custom_instructions = instructions
        return self
    
    def set_domain_context(self, context: str):
        """Set domain-specific context."""
        self.domain_context = context
        return self
    
    def add_safety_constraint(self, constraint: str):
        """Add a safety constraint."""
        self.safety_constraints.append(constraint)
        return self
    
    def set_safety_constraints(self, constraints: List[str]):
        """Replace all safety constraints."""
        self.safety_constraints = constraints
        return self
    
    
    def configure_basic(self):
        """Generic safe defaults."""
        self.set_domain_context("Proactive adaptive control of SWIM exemplar")
        self.set_safety_constraints([
            "Only use: ADD_SERVER, REMOVE_SERVER, SET_DIMMER",
            "Dimmer values must be between 0.0 and 1.0",
            "Validate parameter ranges",
            "Prefer minimal, reversible actions"
        ])
        self.temperature = 0.3
        return self

    
    # ==== Debugging and Monitoring ====
    
    def get_call_history(self) -> List[Dict[str, Any]]:
        """Get recent LLM call history for debugging."""
        return self.call_history.copy()
    
    def clear_call_history(self):
        """Clear call history."""
        self.call_history.clear()
        return self
    
    def get_current_prompts(self) -> Dict[str, str]:
        """Get current prompt templates for review."""
        return {
            "system_template": self.system_prompt_template,
            "user_template": self.user_prompt_template,
            "custom_instructions": self.custom_instructions,
            "domain_context": self.domain_context,
            "safety_constraints": self.safety_constraints
        }
    
    # ==== Private Methods ====
    
    def _extract_telemetry_data(self, context: ReasoningContext) -> Dict[str, Any]:
        """Extract and format telemetry data from context."""
        telemetry = {}
        
        # Direct telemetry data
        if "telemetry" in context.input_data:
            telemetry.update(context.input_data["telemetry"])
        
        # Sensor data
        if "sensors" in context.input_data:
            telemetry.update(context.input_data["sensors"])
        
        # Any numerical measurements
        for key, value in context.input_data.items():
            if isinstance(value, (int, float)):
                telemetry[key] = value
            elif isinstance(value, dict):
                telemetry[key] = value
        
        return telemetry
    
    def _build_system_prompt(self, context: ReasoningContext) -> str:
        """Build the system prompt with context and customizations."""
        context_description = context.metadata.get("description", "IoT system control")
        
        base_prompt = self.system_prompt_template.format(
            context_description=context_description,
            reasoning_type=self.reasoning_type.value,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(context.timestamp))
        )
        
        # Add custom instructions
        if self.custom_instructions:
            base_prompt += f"\n\nCustom Instructions:\n{self.custom_instructions}"
        
        # Add domain context
        if self.domain_context:
            base_prompt += f"\n\nDomain Context:\n{self.domain_context}"
        
        # Add safety constraints
        if self.safety_constraints:
            constraints = "\n".join(f"- {constraint}" for constraint in self.safety_constraints)
            base_prompt += f"\n\nSafety Constraints:\n{constraints}"
        
        return base_prompt
    
    def _build_user_prompt(self, context: ReasoningContext, telemetry_data: Dict[str, Any]) -> str:
        """Build the user prompt with telemetry data."""
        return self.user_prompt_template.format(
            telemetry_data=json.dumps(telemetry_data, indent=2),
            input_data=json.dumps(context.input_data, indent=2),
            metadata=json.dumps(context.metadata or {}, indent=2)
        )
    
 

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Make the actual Gemini API call and log prompts/responses to JSONL."""
        try:
            # Combine system and user prompts for Gemini
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=full_prompt),
                    ],
                ),
            ]

            generate_content_config = types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,  # <-- fixed bug (was self.max_output_tokens)
                thinking_config=types.ThinkingConfig(
                    thinking_budget=-1,  # Use default thinking budget
                ),
            )

            # Collect streamed response
            response_text = ""
            for chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.text:
                    response_text += chunk.text

            response_text = response_text.strip()

            # === Log prompts & responses to JSONL ===
            log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "model": self.model,
            "system_prompt": str(system_prompt),
            "user_prompt": str(user_prompt),
            "response": str(response_text),
            "reasoning_type": getattr(getattr(self, "reasoning_type", None), "value", None),  # safely to string
        }

            log_file = Path("llm_calls.jsonl")
            with log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            return response_text

        except Exception as e:
            self.logger.error(f"Gemini API call failed: {e}")
            raise


    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate the LLM response."""
        try:
            # Try to extract JSON from response
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
            else:
                raise ValueError("No JSON found in response")
            
            action = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["action_type", "target", "params"]
            for field in required_fields:
                if field not in action:
                    raise ValueError(f"Missing required field: {field}")
            
            # Add default values
            action.setdefault("reasoning", "LLM-generated action")
            action.setdefault("confidence", 0.8)
            action.setdefault("source", "llm_reasoner")
            action.setdefault("action_id", f"llm_{int(time.time())}")
            
            return action
            
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            # Return a safe fallback action
            return {
                "action_type": "ALERT",
                "target": "system",
                "params": {"message": f"Failed to parse LLM response: {e}"},
                "reasoning": "Fallback due to parsing error",
                "confidence": 0.1,
                "source": "llm_reasoner",
                "action_id": f"fallback_{int(time.time())}"
            }
    
    def _store_call_history(self, context: ReasoningContext, system_prompt: str, 
                           user_prompt: str, llm_response: str, parsed_result: Dict[str, Any]):
        """Store call history for debugging and improvement."""
        call_record = {
            "timestamp": context.timestamp,
            "session_id": context.session_id,
            "reasoning_type": self.reasoning_type.value,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "llm_response": llm_response,
            "parsed_result": parsed_result,
            "model": self.model
        }
        
        self.call_history.append(call_record)
        
        # Keep only last 100 calls
        if len(self.call_history) > 100:
            self.call_history.pop(0)


def create_llm_reasoner_agent(agent_id: str,
                             config_path: str,
                             llm_api_key: str,
                             nats_url: Optional[str] = None,
                             logger: Optional[logging.Logger] = None) -> 'ReasonerAgent':
    """
    Create a reasoner agent with LLM-based reasoning implementations.
    
    Usage:
        agent = create_llm_reasoner_agent(
            agent_id="smart_reasoner",
            config_path="config.yaml", 
            llm_api_key="your-openai-key"
        )
        
        # Customize prompts
        llm_reasoner = agent.reasoning_implementations[ReasoningType.DECISION]
        llm_reasoner.add_custom_instructions("Always prioritize system stability")
        llm_reasoner.add_safety_constraint("Never exceed 80% server capacity")
    """
    from .reasoner_agent import ReasonerAgent, ReasoningType
    
    reasoning_implementations = {}
    
    # Create LLM reasoners for all reasoning types
    for reasoning_type in ReasoningType:
        llm_reasoner = LLMReasoningImplementation(
            api_key=llm_api_key,
            reasoning_type=reasoning_type,
            logger=logger
        )
        
        # Configure with basic settings
        llm_reasoner.configure_basic()
        
        reasoning_implementations[reasoning_type] = llm_reasoner
    
    return ReasonerAgent(
        agent_id=agent_id,
        reasoning_implementations=reasoning_implementations,
        config_path=config_path,
        nats_url=nats_url,
        logger=logger
    )