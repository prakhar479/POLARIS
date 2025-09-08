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
from google import genai
import google
from pathlib import Path
import random
# Add these imports to the existing reasoner_agent.py file
from .reasoner_core import (
    ReasoningInterface, 
    ReasoningContext, 
    ReasoningResult, 
    ReasoningType,
)

from .reasoner_agent import KnowledgeQueryInterface
from .reasoner_agent import ReasonerAgent
from google.genai import types


class LLMReasoningImplementation(ReasoningInterface):
    """LLM-powered reasoning implementation with customizable prompts."""
    
    def __init__(self, 
                 api_key: str,
                 reasoning_type: ReasoningType,
                 kb_query_interface: Optional[KnowledgeQueryInterface] = None,
                 model: str = "gemini-2.5-flash",
                 max_tokens: int = 4096,
                 temperature: float = 0.5,
                 timeout: float = 30.0,
                 base_url: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        
        self.api_key = api_key
        self.reasoning_type = reasoning_type
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries= 5
        self.timeout = timeout
        self.initial_backoff = 2.0  # seconds
        self.logger = logger or logging.getLogger(f"LLMReasoner.{reasoning_type.value}")
        self.logger.info(f"Initializing LLMReasoningImplementation with model {model}")
        # Initialize OpenAI client
        self.client = genai.Client(api_key=api_key)

        # NEW: In-memory storage for the last action
        self.last_action_memory: Optional[Dict[str, Any]] = None

        self.kb_query = kb_query_interface
        self.context_builder = ContextBuilder(self.logger)
        self.system_prompt_template = """You are an intelligent self-adaptive system controller. Your task is to analyze telemetry data, evaluate the outcome of your last action, and generate the next control action. You are a proactive reasoner, aiming for long-term system stability and performance.

System Goals & Constraints:
- **Primary Goal:** Keep `response_time_ms_weighted` below the target.
- **Secondary Goal:** Maximize user experience by keeping the `dimmer` as high as possible without violating the primary goal.
- **Efficiency Goal:** Avoid using more servers than necessary. Remove servers if utilization is low and performance is healthy.
- **Constraint:** Always respect `max_servers` and `min_servers`.
- **Constraint:** Actions should be conservative. Avoid oscillating (e.g., adding then immediately removing a server).

**REASONING PROCESS (Chain-of-Thought):**

You MUST follow these steps in your reasoning process:
1.  **Analysis:** Briefly summarize the current system state, focusing on key metrics compared to their targets (e.g., "Response time is 150ms, which is 50ms above the 100ms target.").
2.  **Critique of Last Action:** Review the `feedback_on_last_action`. Was it successful, failed, or neutral? Explain why. This is the most important step for learning.
3.  **Strategy Formulation:** Based on your analysis and critique, decide on a high-level strategy. Examples: "Aggressively reduce response time," "Slightly increase user experience," "Optimize costs," "Maintain current state."
4.  **Action Selection:** Choose the single best action to implement your strategy, explaining your choice. (e.g., "The system is overloaded and the last dimmer change failed. The best action is to add a server.").
5.  **Final JSON Output:** Generate the final JSON action. Follow the exact format. Don't deviate. Format involves following keys - action_type, source, action_id, params, priority. 

**OUTPUT FORMAT:**
You MUST provide your reasoning within a `<thinking>` block. After the block, provide the final JSON action in a `<json_output>` block.

**EXAMPLE:**
<thinking>
* **Analysis:** Current response time is 250ms, far above the 100ms target. Utilization is critical at 98%. The trends show both metrics are worsening.
* **Critique of Last Action:** The last action was `SET_DIMMER` to 0.6. The feedback indicates this action `FAILED` as response time continued to climb. This means load shedding is not enough.
* **Strategy Formulation:** The system is critically overloaded. The strategy must be to `Aggressively increase capacity`.
* **Action Selection:** The only action that increases capacity is `ADD_SERVER`. This will immediately address the high utilization and begin to lower the response time.
* **Final JSON Output:** I will now generate the JSON for the ADD_SERVER action.
</thinking>
<json_output>
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
</json_output>
"""
        # Prompt templates - easily customizable
        
        self.user_prompt_template =  """Please analyze the following system state and generate an appropriate control action.

System State & Goals:
{telemetry_data}

Feedback on Your Last Action:
{feedback}

Follow the reasoning process and output format defined in your instructions.
"""
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
            # 1. Perform KB queries needed for this specific reasoner
            reasoning_steps.append("Querying KB for recent monitor snapshots.")
            snapshots = await self.kb_query.query_structured(
                data_types=["observation"],
                filters={"source": "swim_snapshotter", "tags": ["snapshot"]},
                limit=120
            )
            

            # self.logger.info(f"Retrieved {snapshots} snapshots from KB.")  
            reasoning_steps.append("Querying KB for the last control action.")
            # last_action_list = await self.kb_query.query_structured(
            #     data_types=["observation"],
            #     filters={"source": "polaris.reasoning_engine", "tags": ["control_action"]},
            #     limit=1
            # )
            reasoning_steps.append("Retrieving last action from internal memory to critique.")
            last_action = self.last_action_memory
            
            # 2. Build the rich context using its internal builder
            reasoning_steps.append("Processing KB results with internal ContextBuilder.")
            enriched_context_data = self.context_builder.build_context_from_kb_results(snapshots or [], last_action)
            
            # 3. Prepare prompts using the enriched context
            telemetry_data = enriched_context_data # The payload is the telemetry
            system_prompt = self._build_system_prompt(context)
            user_prompt = self._build_user_prompt(context, telemetry_data)
            reasoning_steps.append("Built LLM prompts with enriched context")
            
            # 4. Call the LLM
            llm_response = await self._call_llm(system_prompt, user_prompt)
            reasoning_steps.append("Received LLM response")
            
            # 5. Parse and return the result
            action_result = self._parse_llm_response(llm_response)
            reasoning_steps.append("Parsed and validated LLM output")

            if "error" not in action_result and action_result.get("action_type") != "ALERT":
                reasoning_steps.append("Storing generated action in internal memory for future reflection.")
                # The content must contain a timestamp, just like the KB version did
                action_to_store = {
                    'content': action_result.copy()
                }
                action_to_store['content']['timestamp'] = datetime.now(timezone.utc).isoformat()
                self.last_action_memory = action_to_store
                self.logger.info(f"Successfully stored action '{action_result['action_type']}' in internal memory.")

            
            self._store_call_history(context, system_prompt, user_prompt, llm_response, action_result)
            
            execution_time = time.time() - start_time
            return ReasoningResult(
                result=action_result,
                confidence=action_result.get("confidence", 0.8),
                reasoning_steps=reasoning_steps,
                context=context,
                execution_time=execution_time,
                kb_queries_made=2 # We made 2 queries
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
        self.set_domain_context('''Proactive adaptive control of SWIM exemplar.
                                
                                About SWIM: The system manages servers and content delivery.

                                Actions: Add/remove servers, adjust a dimmer (0.0–1.0 for optional content), or take no action.

                                Metrics: Track dimmer level, active/max servers, utilization, response times (overall, basic, optional), arrival rate, and throughput (basic/optional).

                                Behavior: High utilization → exponential rise in response time; adding servers lowers per-server load but raises cost; lowering dimmer reduces load/response time.

                                Thresholds:  normal response time = 1.0, recommended servers ≈ arrival_rate/10.
                                
                                ''')
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
        """
        Extract and format telemetry data from context, with specific handling
        for the SWIM 'events' list structure.
        """
        # telemetry = {}

        # # --- NEW LOGIC TO PARSE THE 'events' LIST ---
        # # Check if 'events' key exists and is a list in the input data
        # if "events" in context.input_data and isinstance(context.input_data["events"], list):
        #     # Iterate through each event dictionary in the list
        #     for event in context.input_data["events"]:
        #         # Check if the event has a 'name' and 'value'
        #         if "name" in event and "value" in event:
        #             # Add the metric to our telemetry dictionary.
        #             # If a metric appears multiple times, this will keep the latest one.
        #             telemetry[event["name"]] = event["value"]

        # # --- Keep old logic as a fallback for other data formats ---
        # # Direct telemetry data
        # if "telemetry" in context.input_data:
        #     telemetry.update(context.input_data["telemetry"])
            
        # # Any other top-level numerical measurements
        # for key, value in context.input_data.items():
        #     if key not in telemetry and isinstance(value, (int, float)):
        #         telemetry[key] = value

        # return telemetry
        return context.input_data
    

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

        feedback = telemetry_data.pop("feedback_on_last_action", {})
        return self.user_prompt_template.format(
            telemetry_data=json.dumps(telemetry_data, indent=2),
            feedback=json.dumps(feedback, indent=2)
        )

 

    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Make a direct Gemini API call with retry logic, without using any tools.
        """
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"


        # self.logger.info(f"Making LLM call with prompt {full_prompt}")

        # self.logger.info(f"Making LLM call with prompt: {full_prompt}")
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=full_prompt)])]

        # Simplified generation config without tool-related settings
        generate_content_config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens
        )

        for attempt in range(self.max_retries):
            try:
                # Direct API call without the 'tools' parameter
                response = self.client.models.generate_content(
                    model=self.model,  # Ensure self.model is set to "gemini-1.5-flash"
                    contents=contents,
                    config=generate_content_config,
                )

                response_text = response.text.strip()
                self.logger.info(f"LLM response: {response_text}")

                # --- Success Case: Log and return ---
                log_entry = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "model": self.model,
                    "system_prompt": str(system_prompt),
                    "user_prompt": str(user_prompt),
                    "response": str(response_text),
                    "reasoning_type": getattr(self.reasoning_type, "value", None),
                    "status": "success",
                    "attempt": attempt + 1
                }
                log_file = Path("llm_calls.jsonl")
                with log_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                
                return response_text

            except google.genai.errors.ServerError as e:
                # --- Failure Case (retry logic remains unchanged) ---
                self.logger.warning(f"Gemini API call failed with ServerError: {e}. This is attempt {attempt + 1}/{self.max_retries}.")
                if attempt + 1 == self.max_retries:
                    self.logger.error("Max retries reached. Failing the operation.")
                    # ... (logging code for final failure) ...
                    raise
                
                backoff_time = self.initial_backoff * (2 ** attempt)
                jitter = random.uniform(0, backoff_time * 0.1)
                delay = backoff_time + jitter
                self.logger.info(f"Model is overloaded. Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)

        raise Exception("LLM call failed after all retries.")
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate the LLM response, expecting a <json_output> block."""
        try:
            # Look for the <json_output> block
            if "<json_output>" in response:
                json_start = response.find("<json_output>") + len("<json_output>")
                json_end = response.find("</json_output>", json_start)
                if json_end == -1:
                    raise ValueError("Found <json_output> but no closing </json_output> tag.")
                json_str = response[json_start:json_end].strip()
            else:
                # Fallback for cases where the model might forget the tags
                self.logger.warning("LLM response did not contain <json_output> tags. Falling back to extracting first JSON object.")
                if "{" in response and "}" in response:
                    json_start = response.find("{")
                    json_end = response.rfind("}") + 1
                    json_str = response[json_start:json_end]
                else:
                    raise ValueError("No JSON object found in response")

            action = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["action_type", "source", "params","action_id"]
            for field in required_fields:
                if field not in action:
                    raise ValueError(f"Missing required field in parsed JSON: {field}")
            
            action.setdefault("source", "llm_reasoner")
            
            return action
            
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {e}. Response was: {response}")
            return {
                "action_type": "ALERT",
                "source": "llm_reasoner_parser",
                "params": {"message": f"Failed to parse LLM response: {e}"},
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
                             logger: Optional[logging.Logger] = None,
                             mode='llm') -> 'ReasonerAgent':
    """
    Create a reasoner agent with LLM-based reasoning implementations.
    This factory correctly injects the agent's knowledge base query interface
    into the LLM reasoner for its own specialized data processing.
    
    Usage:
        agent = create_llm_reasoner_agent(
            agent_id="smart_reasoner",
            config_path="config.yaml", 
            llm_api_key="your-gemini-key"
        )
        
        # Customize prompts
        llm_reasoner = agent.reasoning_implementations[ReasoningType.DECISION]
        llm_reasoner.add_custom_instructions("Always prioritize system stability")
        llm_reasoner.add_safety_constraint("Never exceed 80% server capacity")
    """
    from .reasoner_agent import ReasonerAgent, ReasoningType
    # This import is now needed here
    from .llm_reasoner import LLMReasoningImplementation
    from .multi_agent_reasoner import MultiAgentReasoner

    # Step 1: Create the agent instance first, with an empty set of implementations.
    # Its __init__ method will create and own the self.kb_query interface.
    agent = ReasonerAgent(
        agent_id=agent_id,
        reasoning_implementations={}, # Start with an empty dictionary
        config_path=config_path,
        nats_url=nats_url,
        logger=logger
    )
    
    # Step 2: Create the LLM reasoners for all reasoning types and inject the dependency.
    for reasoning_type in ReasoningType:


        if mode == 'multi':
            llm_reasoner = MultiAgentReasoner(
                api_key=llm_api_key,
                reasoning_type=reasoning_type,
                # This is the crucial dependency injection step:
                kb_query_interface=agent.kb_query,
                logger=logger
            )
            agent.add_reasoning_implementation(reasoning_type, llm_reasoner)
        else:
            llm_reasoner = LLMReasoningImplementation(
                api_key=llm_api_key,
                reasoning_type=reasoning_type,
                # This is the crucial dependency injection step:
                kb_query_interface=agent.kb_query,
                logger=logger
            )
        
        # Configure with basic settings
        llm_reasoner.configure_basic()
        
        # Step 3: Add the fully configured implementation back into the agent.
        agent.add_reasoning_implementation(reasoning_type, llm_reasoner)
    
    # Return the fully assembled and configured agent
    return agent


from datetime import datetime, timezone

class ContextBuilder:
    """
    Processes lists of KB entries (snapshots, actions) to build the
    rich context payload required by the LLM.
    """

    def __init__(self,logger) -> None:
        self.logger = logger or logging.getLogger("ContextBuilder")

    def calculate_ewma(self, data: List[float], alpha: float = 0.5) -> Optional[float]:
        if not data: return None
        ewma = data[0]
        for value in data[1:]:
            ewma = alpha * value + (1 - alpha) * ewma
        return ewma

    def calculate_rate_of_change(self, sorted_snapshots: List[Dict[str, Any]], metric: str, time_window_seconds: int = 300) -> Optional[str]:
        if len(sorted_snapshots) < 2: return None
        
        # We need to extract the value and timestamp for the given metric
        # from each snapshot's 'content' field.
        time_series = []
        for s in sorted_snapshots:
            if metric in s['content']['content']['current_state']:
                time_series.append({
                    "timestamp": s['timestamp'],
                    "value": s['content']['content']['current_state'][metric]
                })
        
        if len(time_series) < 2: return None

        latest_point = time_series[-1]
        # self.logger.info(f"Calculating rate of change for {metric} using latest point: {latest_point}")
        past_timestamp_limit = datetime.fromisoformat(latest_point["timestamp"]).timestamp() - time_window_seconds


        
        reference_point = time_series[0]
        for point in time_series:
            if datetime.fromisoformat(point["timestamp"]).timestamp() >= past_timestamp_limit:
                reference_point = point
                break
        
        time_diff_seconds = datetime.fromisoformat(latest_point["timestamp"]).timestamp() - datetime.fromisoformat(reference_point["timestamp"]).timestamp()
        if time_diff_seconds < 1: return "+0.00/min"
        
        value_diff = latest_point["value"] - reference_point["value"]
        rate_per_minute = (value_diff / time_diff_seconds) * 60
        return f"{rate_per_minute:+.2f}/min"

    def build_context_from_kb_results(self, snapshots: List[Dict[str, Any]], last_action: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        The main method to transform raw KB results into the final context payload.
        """
        if not snapshots:
            return {"error": "No recent snapshots found in the Knowledge Base."}

        # Sort snapshots by timestamp just in case they are out of order
        sorted_snapshots = sorted(snapshots, key=lambda s: s['timestamp'])
        latest_snapshot = sorted_snapshots[-1]['content']['content']

        # 1. Get the Current State from the latest snapshot
        current_state = latest_snapshot.get('current_state', {})
        
        # 2. Calculate Historical Trends across all snapshots
        historical_trends = {}

        # Identify all unique metrics across the snapshots to calculate trends for
        
        
        all_metrics = set(
            k
            for s in sorted_snapshots
            for k in s.get("content", {}).get("content", {}).get("current_state", {}).keys()
        )

        
        for metric in all_metrics:
            # Create a time series of values for the current metric
            values = [
                        s["content"]["content"]["current_state"][metric]
                        for s in sorted_snapshots
                        if "content" in s and "content" in s["content"] and metric in s["content"]["content"].get("current_state", {})
                    ]

            
            ewma = self.calculate_ewma(values)
            roc = self.calculate_rate_of_change(sorted_snapshots, metric)
            
            historical_trends[metric] = {
                "5min_avg_ewma": round(ewma, 4) if ewma is not None else None,
                "5min_rate_of_change": roc
            }

        # 3. Process the Controller State from the last action
        controller_state = {}
        if last_action:
            action_content = last_action['content']
            action_time = datetime.fromisoformat(action_content["timestamp"])
            minutes_since = (datetime.now(timezone.utc) - action_time).total_seconds() / 60
            controller_state = {
                "last_action": {
                    "timestamp": action_content["timestamp"],
                    "action_type": action_content["action_type"]
                },
                "minutes_since_last_action": round(minutes_since, 2)
            }
        else:
            controller_state = {"last_action": None, "minutes_since_last_action": 9999}
            

        # NEW: Feedback loop logic
        feedback_on_last_action = {"evaluation": "No recent action to evaluate."}
        if last_action and len(sorted_snapshots) > 1:
            action_time_ts = datetime.fromisoformat(last_action['content']['timestamp']).timestamp()
            pre_action_snapshot = next((s for s in reversed(sorted_snapshots) if datetime.fromisoformat(s['timestamp']).timestamp() < action_time_ts), None)

            if pre_action_snapshot:
                pre_state = pre_action_snapshot['content']['content']['current_state']
                
                rt_before = pre_state.get('response_time_ms_weighted', 0)
                rt_after = current_state.get('response_time_ms_weighted', 0)
                rt_change = rt_after - rt_before

                util_before = pre_state.get('utilization', 0)
                util_after = current_state.get('utilization', 0)
                util_change = util_after - util_before

                # Define simple evaluation criteria
                evaluation = "NEUTRAL"
                rt_improvement_threshold = -10.0 # Must decrease by at least 10ms
                if rt_change < rt_improvement_threshold:
                    evaluation = "SUCCESSFUL"
                elif rt_change > abs(rt_improvement_threshold):
                    evaluation = "FAILED"

                feedback_on_last_action = {
                    "action_taken": last_action['content']['action_type'],
                    "params": last_action['content'].get('params', {}),
                    "evaluation": evaluation,
                    "outcome_details": f"Response time changed by {rt_change:+.2f}ms (from {rt_before:.2f} to {rt_after:.2f}). Utilization changed by {util_change*100:+.1f}%."
                }
                self.logger.info(f"Generated feedback for last action: {feedback_on_last_action}")

        # MODIFIED: Goals are now more specific and quantitative
        system_goals_and_constraints = {
            "goals": {
                "target_response_time_ms_weighted": 100.0,
                "target_utilization_min_for_scale_down": 0.40,
            },
            "constraints": {
                "max_servers": 3, 
                "min_servers": 1,
                "cooldown_period_minutes": 5,
            }
        }
        
        # 4. Define Goals and Constraints (can be loaded from config)
        system_goals_and_constraints = {
            "goals": {"target_utilization": 0.85},
            "constraints": {"max_servers": 3, "cooldown_period_minutes": 5}
        }
            
        return {
            "current_state": current_state,
            "historical_trends": historical_trends,
            "controller_state": controller_state,
            "system_goals_and_constraints": system_goals_and_constraints
        }