"""
Base System-Agnostic Prompts for Agentic Reasoner.

Contains common prompt components that are used across all managed systems.
System-specific prompts should inherit and extend these base prompts.
"""


def get_base_system_prompt_template(tools_description: str, system_specific_content: str) -> str:
    """
    Get the base system prompt template that can be customized per system.
    
    Args:
        tools_description: Description of available tools
        system_specific_content: System-specific knowledge, actions, constraints
        
    Returns:
        Formatted base system prompt
    """
    return f"""You are an autonomous adaptive system controller for making adaptation decisions.

Your role is to:
1. Analyze the current system situation
2. Use available tools to gather additional information as needed
3. Make informed adaptation decisions based on your analysis
4. Generate appropriate control actions

{tools_description}

KNOWLEDGE BASE QUERY GUIDELINES:
When using query_knowledge_base tool, follow these patterns:

1. **For recent observations/aggregated telemetry**:
   - query_type: "structured"
   - data_types: ["observation"]
   - Optional filters for specific metrics

2. **For raw telemetry events**:
   - query_type: "structured" 
   - data_types: ["raw_telemetry_event"]
   - Use filters to specify metric_name if needed

3. **For adaptation decisions history**:
   - query_type: "structured"
   - data_types: ["adaptation_decision"]

4. **For natural language search**:
   - query_type: "natural_language"
   - query_text: "your search terms"

Valid data_types: ["raw_telemetry_event", "adaptation_decision", "system_goal", "learned_pattern", "observation", "system_info", "generic_fact"]

IMPORTANT INSTRUCTIONS FOR TOOL USAGE:
- If you need additional information to make a good decision, you MUST use the available tools
- Always try to gather more context before making decisions when tools are available
- Use tools to check historical patterns, system state, or run simulations
- Tool usage is STRONGLY ENCOURAGED when tools are available

{system_specific_content}

CRITICAL WORKFLOW RULES:
1. **NEVER provide both tool calls AND action in the same response**
2. **If you need tools, ONLY provide tool calls - NO action**
3. **Wait for tool results, then provide your final action**
4. **Tool calls are processed iteratively - you can make up to 5 tool calls total**
5. **Each tool call response will be provided back to you before you can make more calls**

MANDATORY OUTPUT FORMAT:

**PHASE 1 - INFORMATION GATHERING (if tools needed):**
Your response should contain:

1. **Analysis**: Brief analysis of the current situation

2. **Tool Usage Decision**: 
   - State what information you need and which tools you will use
   - Be specific about what you hope to learn

3. **Tool Calls**: Use this EXACT format for each tool call:
   ```
   TOOL_CALL: tool_name
   PARAMETERS: {{"param1": "value1", "param2": "value2"}}
   ```

**PHASE 2 - FINAL DECISION (after tool results received):**
Your response should contain:

1. **Analysis**: Analysis of tool results and current situation

2. **Final Decision**: Your decision and reasoning based on all available information

3. **Action**: The control action in JSON format:
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

CRITICAL REMINDERS:
1. **NEVER provide tool calls AND action in the same response**
2. **If you use tools, wait for results before providing action**
3. **Tool results will be provided back to you in subsequent messages**
4. **You can make multiple rounds of tool calls (up to 5 total) before final action**
"""


def get_tools_description(tools_available: bool = True) -> str:
    """
    Get the tools description section.
    
    Args:
        tools_available: Whether KB and DT tools are available
        
    Returns:
        Formatted tools description
    """
    if tools_available:
        return """
Available Tools:
- query_knowledge_base: Query the knowledge base for historical data, observations, and telemetry
  Parameters: query_type, data_types, filters, query_text, limit
  
- query_digital_twin: Query the digital twin for system state, simulations, and diagnostics
  Parameters: operation (query/simulate/diagnose), query_type, query_content, 
              simulation_type, actions, horizon_minutes, anomaly_description
"""
    else:
        return "\nNote: No external tools are currently available. You must make decisions based solely on the provided input data.\n"


def format_base_user_prompt(timestamp: str, telemetry_events: str, recent_actions: str) -> str:
    """
    Format a base user prompt with common telemetry information.
    
    Args:
        timestamp: Current timestamp
        telemetry_events: Formatted telemetry events
        recent_actions: Formatted recent actions
        
    Returns:
        Formatted user prompt
    """
    return f"""
# CURRENT SYSTEM STATE

**Timestamp**: {timestamp}

## Telemetry Events
{telemetry_events}

## Recent Actions
{recent_actions}

---

**Your Task**: Analyze this system state and determine the optimal adaptation action.

If you need additional information (historical patterns, simulations, diagnostics), use the available tools in PHASE 1.
Otherwise, proceed directly to PHASE 2 with your final decision.
"""
