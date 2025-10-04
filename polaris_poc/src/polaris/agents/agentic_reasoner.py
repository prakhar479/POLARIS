"""
Agentic LLM Reasoner Implementation

An autonomous agentic reasoner that can dynamically decide which tools to use
(Knowledge Base queries, Digital Twin interactions) and make adaptation decisions
based on its analysis. The LLM acts as the agent's reasoning engine and decides
when and how to use available tools.
"""

import json
import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional, Union, Callable
import logging
from datetime import datetime, timezone
import httpx
import yaml

from .reasoner_core import (
    ReasoningInterface,
    ReasoningContext,
    ReasoningResult,
    ReasoningType,
)

from .reasoner_agent import (
    KnowledgeQueryInterface,
    DigitalTwinInterface,
    DTQuery,
    DTSimulation,
    DTDiagnosis,
    DTResponse,
    GRPCDigitalTwinClient,
)


class AgenticTool:
    """Base class for tools that the agentic reasoner can use."""
    
    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters."""
        raise NotImplementedError


class KnowledgeBaseTool(AgenticTool):
    """Tool for querying the knowledge base."""
    
    def __init__(self, kb_query: KnowledgeQueryInterface, logger: logging.Logger):
        super().__init__(
            name="query_knowledge_base",
            description="Query the knowledge base for historical data, observations, and telemetry",
            parameters={
                "query_type": {
                    "type": "string",
                    "enum": ["structured", "natural_language", "recent_observations", "raw_telemetry"],
                    "description": "Type of query to perform"
                },
                "data_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Types of data to query (for structured queries)"
                },
                "filters": {
                    "type": "object",
                    "description": "Filters to apply to the query"
                },
                "query_text": {
                    "type": "string",
                    "description": "Natural language query text"
                },
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum number of results to return"
                }
            }
        )
        self.kb_query = kb_query
        self.logger = logger
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute knowledge base query."""
        try:
            query_type = kwargs.get("query_type", "structured")
            limit = kwargs.get("limit", 10)
            
            if query_type == "structured":
                data_types = kwargs.get("data_types", ["observation"])
                filters = kwargs.get("filters", {})
                results = await self.kb_query.query_structured(
                    data_types=data_types,
                    filters=filters,
                    limit=limit
                )
            elif query_type == "natural_language":
                query_text = kwargs.get("query_text", "")
                results = await self.kb_query.query_natural_language(
                    query_text=query_text,
                    limit=limit
                )
            elif query_type == "recent_observations":
                results = await self.kb_query.query_recent_observations(limit=limit)
            elif query_type == "raw_telemetry":
                results = await self.kb_query.query_raw_telemetry(limit=limit)
            else:
                return {"success": False, "error": f"Unknown query type: {query_type}"}
            
            return {
                "success": True,
                "results": results or [],
                "count": len(results) if results else 0
            }
        except Exception as e:
            self.logger.error(f"Knowledge base query failed: {e}")
            return {"success": False, "error": str(e)}


class DigitalTwinTool(AgenticTool):
    """Tool for interacting with the digital twin."""
    
    def __init__(self, dt_interface: DigitalTwinInterface, logger: logging.Logger):
        super().__init__(
            name="query_digital_twin",
            description="Query the digital twin for system state, simulations, and diagnostics",
            parameters={
                "operation": {
                    "type": "string",
                    "enum": ["query", "simulate", "diagnose"],
                    "description": "Type of digital twin operation"
                },
                "query_type": {
                    "type": "string",
                    "description": "Type of query (for query operations)"
                },
                "query_content": {
                    "type": "string",
                    "description": "Content of the query"
                },
                "simulation_type": {
                    "type": "string",
                    "description": "Type of simulation (for simulate operations)"
                },
                "actions": {
                    "type": "array",
                    "description": "Actions to simulate"
                },
                "horizon_minutes": {
                    "type": "integer",
                    "default": 60,
                    "description": "Simulation horizon in minutes"
                },
                "anomaly_description": {
                    "type": "string",
                    "description": "Description of anomaly (for diagnose operations)"
                }
            }
        )
        self.dt_interface = dt_interface
        self.logger = logger
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute digital twin operation."""
        try:
            # Check if digital twin interface is available
            if not self.dt_interface:
                return {"success": False, "error": "Digital twin interface not available"}
            
            # Ensure connection is established
            if hasattr(self.dt_interface, 'stub') and not self.dt_interface.stub:
                self.logger.info("Digital twin not connected, attempting to connect...")
                await self.dt_interface.connect()
                if not self.dt_interface.stub:
                    return {"success": False, "error": "Failed to connect to digital twin"}
            
            operation = kwargs.get("operation", "query")
            self.logger.debug(f"Executing digital twin operation: {operation}")
            
            if operation == "query":
                query = DTQuery(
                    query_type=kwargs.get("query_type", "current_state"),
                    query_content=kwargs.get("query_content", "Get system overview"),
                    parameters=kwargs.get("parameters", {})
                )
                response = await self.dt_interface.query(query)
            elif operation == "simulate":
                simulation = DTSimulation(
                    simulation_type=kwargs.get("simulation_type", "forecast"),
                    actions=kwargs.get("actions", []),
                    horizon_minutes=kwargs.get("horizon_minutes", 60),
                    parameters=kwargs.get("parameters", {})
                )
                response = await self.dt_interface.simulate(simulation)
            elif operation == "diagnose":
                diagnosis = DTDiagnosis(
                    anomaly_description=kwargs.get("anomaly_description", ""),
                    context=kwargs.get("context", {})
                )
                response = await self.dt_interface.diagnose(diagnosis)
            else:
                return {"success": False, "error": f"Unknown operation: {operation}"}
            
            if response:
                result = {
                    "success": response.success,
                    "result": response.result,
                    "confidence": response.confidence,
                    "explanation": response.explanation,
                    "metadata": response.metadata or {}
                }
                
                # Add optional fields if they exist
                if hasattr(response, 'future_states') and response.future_states:
                    result["future_states"] = response.future_states
                if hasattr(response, 'hypotheses') and response.hypotheses:
                    result["hypotheses"] = response.hypotheses
                
                return result
            else:
                return {"success": False, "error": "No response from digital twin"}
                
        except Exception as e:
            self.logger.error(f"Digital twin operation failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}


class AgenticLLMReasoner(ReasoningInterface):
    """
    Agentic LLM reasoner that can dynamically decide which tools to use
    and make autonomous adaptation decisions.
    """
    
    def __init__(
        self,
        api_key: str,
        reasoning_type: ReasoningType,
        kb_query_interface: Optional[KnowledgeQueryInterface] = None,
        dt_interface: Optional[DigitalTwinInterface] = None,
        model: str = "gpt-oss:20b",
        max_tokens: int = 1024,
        temperature: float = 0.3,
        timeout: float = 600.0,
        base_url: str = "http://10.10.16.46:11435",
        max_tool_calls: int = 5,
        logger: Optional[logging.Logger] = None,
    ):
        self.api_key = api_key
        self.reasoning_type = reasoning_type
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.base_url = base_url
        self.max_tool_calls = max_tool_calls
        self.max_retries = 3
        self.logger = logger or logging.getLogger(f"AgenticReasoner.{reasoning_type.value}")
        
        # Initialize tools
        self.tools = {}
        if kb_query_interface:
            self.tools["query_knowledge_base"] = KnowledgeBaseTool(kb_query_interface, self.logger)
            self.logger.info("Knowledge Base tool initialized")
        if dt_interface:
            self.tools["query_digital_twin"] = DigitalTwinTool(dt_interface, self.logger)
            self.logger.info("Digital Twin tool initialized")
        
        if not self.tools:
            self.logger.warning("No tools available - agentic reasoner will have limited capabilities")
        else:
            self.logger.info(f"Agentic reasoner initialized with {len(self.tools)} tools: {list(self.tools.keys())}")
        
        # System prompt for the agentic reasoner
        self.system_prompt = self._build_system_prompt()
        
        self.logger.info(f"Initialized AgenticLLMReasoner with {len(self.tools)} tools available")
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the agentic reasoner."""
        tools_description = ""
        if self.tools:
            tools_description = "\n\nAvailable Tools:\n"
            for tool_name, tool in self.tools.items():
                tools_description += f"- {tool_name}: {tool.description}\n"
                tools_description += f"  Parameters: {json.dumps(tool.parameters, indent=2)}\n"
        else:
            tools_description = "\n\nNote: No external tools are currently available. You must make decisions based solely on the provided input data.\n"
        
        return f"""You are an autonomous adaptive system controller{"" if self.tools else " operating in standalone mode"} for making adaptation decisions.

Your role is to:
1. Analyze the current system situation
2. Use available tools to gather additional information as needed
3. Make informed adaptation decisions based on your analysis
4. Generate appropriate control actions

{tools_description}

Decision Making Process:
1. First, analyze the input data to understand the current situation
2. Determine what additional information you need
3. Use tools to gather that information
4. Synthesize all information to make an adaptation decision
5. Generate the appropriate control action

Control Actions Available:
- ADD_SERVER: Add a server to handle increased load
- REMOVE_SERVER: Remove a server to reduce costs
- SET_DIMMER: Adjust the dimmer value (0.0-1.0) to control optional content
- NO_ACTION: Take no action if system is operating optimally

System Constraints:
- Response time should stay below 1000ms
- Utilization should be between 50-85%
- Server count should be between 1-10
- Dimmer value should be between 0.0-1.0

Output Format:
Your response should be structured as follows:

1. **Analysis**: Brief analysis of the current situation
2. **Tool Calls**: If you need more information, specify tool calls in this format:
   ```
   TOOL_CALL: tool_name
   PARAMETERS: {{"param1": "value1", "param2": "value2"}}
   ```
3. **Decision**: Your final decision and reasoning
4. **Action**: The control action in JSON format:
   ```json
   {{
     "action_type": "ACTION_TYPE",
     "source": "agentic_reasoner",
     "action_id": "generated-uuid",
     "params": {{}},
     "priority": "low|medium|high",
     "reasoning": "Brief explanation of the decision"
   }}
   ```

Remember: You are autonomous and should make decisions based on the available data and tools. Be decisive but cautious.
"""
    
    async def reason(
        self, context: ReasoningContext, knowledge: Optional[List[Dict[str, Any]]] = None
    ) -> ReasoningResult:
        """Execute agentic reasoning with dynamic tool usage."""
        start_time = time.time()
        reasoning_steps = ["Starting agentic LLM reasoning"]
        tool_calls_made = 0
        
        try:
            # Initial analysis
            user_prompt = self._build_initial_prompt(context)
            reasoning_steps.append("Built initial analysis prompt")
            
            # Start the reasoning loop
            conversation_history = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            final_action = None
            
            for iteration in range(self.max_tool_calls + 1):  # +1 for final decision
                reasoning_steps.append(f"Reasoning iteration {iteration + 1}")
                
                # Get LLM response
                llm_response = await self._call_llm_with_history(conversation_history)
                reasoning_steps.append(f"Received LLM response (iteration {iteration + 1})")
                
                # Parse response for tool calls or final decision
                tool_calls, action = self._parse_llm_response(llm_response)
                
                if action:
                    # Final decision reached
                    final_action = action
                    reasoning_steps.append("Final decision reached")
                    break
                
                if tool_calls and tool_calls_made < self.max_tool_calls:
                    # Execute tool calls
                    tool_results = []
                    for tool_call in tool_calls:
                        tool_name = tool_call.get("tool_name")
                        parameters = tool_call.get("parameters", {})
                        
                        if tool_name in self.tools:
                            reasoning_steps.append(f"Executing tool: {tool_name}")
                            result = await self.tools[tool_name].execute(**parameters)
                            tool_results.append({
                                "tool_name": tool_name,
                                "parameters": parameters,
                                "result": result
                            })
                            tool_calls_made += 1
                        else:
                            reasoning_steps.append(f"Unknown tool requested: {tool_name}")
                    
                    # Add tool results to conversation
                    if tool_results:
                        conversation_history.append({"role": "assistant", "content": llm_response})
                        tool_results_text = "Tool Results:\n" + json.dumps(tool_results, indent=2)
                        conversation_history.append({"role": "user", "content": tool_results_text})
                        reasoning_steps.append(f"Added {len(tool_results)} tool results to conversation")
                else:
                    # No more tool calls allowed or no tool calls requested
                    if not action:
                        # Force a decision
                        conversation_history.append({"role": "assistant", "content": llm_response})
                        conversation_history.append({
                            "role": "user", 
                            "content": "Please provide your final decision and action in the required JSON format."
                        })
                        final_response = await self._call_llm_with_history(conversation_history)
                        _, final_action = self._parse_llm_response(final_response)
                        reasoning_steps.append("Forced final decision")
                    break
            
            # Ensure we have a valid action
            if not final_action:
                final_action = {
                    "action_type": "NO_ACTION",
                    "source": "agentic_reasoner",
                    "action_id": str(uuid.uuid4()),
                    "params": {},
                    "priority": "low",
                    "reasoning": "Unable to determine appropriate action"
                }
                reasoning_steps.append("Fallback to NO_ACTION")
            
            # Ensure action has required fields
            final_action.setdefault("action_id", str(uuid.uuid4()))
            final_action.setdefault("source", "agentic_reasoner")
            final_action.setdefault("priority", "medium")
            
            execution_time = time.time() - start_time
            
            return ReasoningResult(
                result=final_action,
                confidence=0.8,  # Could be dynamic based on tool results
                reasoning_steps=reasoning_steps,
                context=context,
                execution_time=execution_time,
                kb_queries_made=tool_calls_made,  # Approximate
                dt_queries_made=0  # Could track separately
            )
            
        except Exception as e:
            self.logger.error(f"Agentic reasoning failed: {e}", exc_info=True)
            execution_time = time.time() - start_time
            
            return ReasoningResult(
                result={
                    "action_type": "ALERT",
                    "source": "agentic_reasoner",
                    "action_id": str(uuid.uuid4()),
                    "params": {"message": f"Agentic reasoning failed: {e}"},
                    "priority": "high",
                    "reasoning": "Fallback due to reasoning failure"
                },
                confidence=0.1,
                reasoning_steps=reasoning_steps + [f"Error: {str(e)}"],
                context=context,
                execution_time=execution_time
            )
    
    def _build_initial_prompt(self, context: ReasoningContext) -> str:
        """Build the initial prompt for the reasoning session."""
        return f"""Current System Context:
{json.dumps(context.input_data, indent=2)}

Session ID: {context.session_id}
Reasoning Type: {context.reasoning_type.value}
Timestamp: {datetime.fromtimestamp(context.timestamp).isoformat()}

Please analyze this situation and determine what actions, if any, should be taken to maintain optimal system performance.
"""
    
    async def _call_llm_with_history(self, conversation_history: List[Dict[str, str]]) -> str:
        """Call the LLM with conversation history."""
        # Convert conversation to a single prompt for simple LLM APIs
        prompt_parts = []
        for message in conversation_history:
            role = message["role"]
            content = message["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        full_prompt = "\n\n".join(prompt_parts)
        
        endpoint_url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        headers = {"Content-Type": "application/json"}
        
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(endpoint_url, json=payload, headers=headers)
                    response.raise_for_status()
                    
                    response_data = response.json()
                    response_text = response_data.get("response", "").strip()
                    
                    if not response_text:
                        raise ValueError("Empty response from LLM")
                    
                    return response_text
                    
            except Exception as e:
                self.logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt + 1 == self.max_retries:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception("LLM call failed after all retries")
    
    def _parse_llm_response(self, response: str) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Parse LLM response for tool calls and final actions."""
        tool_calls = []
        action = None
        
        lines = response.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for tool calls
            if line.startswith("TOOL_CALL:"):
                tool_name = line.replace("TOOL_CALL:", "").strip()
                i += 1
                
                # Look for parameters on the next line
                if i < len(lines) and lines[i].strip().startswith("PARAMETERS:"):
                    params_line = lines[i].strip().replace("PARAMETERS:", "").strip()
                    try:
                        parameters = json.loads(params_line)
                    except json.JSONDecodeError:
                        parameters = {}
                    
                    tool_calls.append({
                        "tool_name": tool_name,
                        "parameters": parameters
                    })
            
            # Look for JSON action
            elif line.startswith("{") and not action:
                # Try to parse JSON action
                json_start = i
                json_lines = []
                brace_count = 0
                
                while i < len(lines):
                    current_line = lines[i]
                    json_lines.append(current_line)
                    
                    # Count braces to find complete JSON
                    brace_count += current_line.count('{') - current_line.count('}')
                    
                    if brace_count == 0 and len(json_lines) > 0:
                        break
                    i += 1
                
                try:
                    json_str = '\n'.join(json_lines)
                    action = json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            
            i += 1
        
        return tool_calls, action
    
    async def validate_input(self, context: ReasoningContext) -> bool:
        """Validate input context."""
        return True  # Basic validation - could be enhanced
    
    def get_required_knowledge_types(self, context: ReasoningContext) -> List[str]:
        """Get required knowledge types (not used in agentic approach)."""
        return []
    
    def extract_search_terms(self, context: ReasoningContext) -> List[str]:
        """Extract search terms (not used in agentic approach)."""
        return []


def create_agentic_reasoner_agent(
    agent_id: str,
    config_path: str,
    llm_api_key: str,
    nats_url: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> "ReasonerAgent":
    """
    Create a reasoner agent with agentic LLM reasoning implementation.
    """
    from .reasoner_agent import ReasonerAgent, ReasoningType
    
    agent = ReasonerAgent(
        agent_id=agent_id,
        reasoning_implementations={},
        config_path=config_path,
        nats_url=nats_url,
        logger=logger,
    )
    
    # Log available interfaces
    if logger:
        logger.info(f"Creating agentic reasoner with KB interface: {agent.kb_query is not None}")
        logger.info(f"Creating agentic reasoner with DT interface: {agent.dt_query is not None}")
        if agent.dt_query:
            logger.info(f"Digital Twin address: {getattr(agent.dt_query, 'grpc_address', 'unknown')}")
    
    # Create agentic reasoner for each reasoning type
    for reasoning_type in ReasoningType:
        agentic_reasoner = AgenticLLMReasoner(
            api_key=llm_api_key,
            reasoning_type=reasoning_type,
            kb_query_interface=agent.kb_query,
            dt_interface=agent.dt_query,
            logger=logger,
        )
        
        agent.add_reasoning_implementation(reasoning_type, agentic_reasoner)
    
    return agent