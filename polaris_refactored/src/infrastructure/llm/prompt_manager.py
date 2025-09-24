"""
Prompt Management System

Manages prompt templates, conversation flows, and context building
for agentic LLM interactions within the POLARIS framework.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import yaml
import json

from .models import (
    PromptTemplate, ConversationFlow, Message, MessageRole,
    AgenticConversation
)
from .exceptions import LLMConfigurationError


class PromptManager:
    """Manages prompt templates and conversation flows for LLM interactions."""
    
    def __init__(self, template_directory: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.template_directory = template_directory
        self._templates: Dict[str, PromptTemplate] = {}
        self._flows: Dict[str, ConversationFlow] = {}
        
        # Load default templates
        self._load_default_templates()
        
        # Load templates from directory if provided
        if template_directory and template_directory.exists():
            self._load_templates_from_directory(template_directory)
    
    def _load_default_templates(self) -> None:
        """Load default prompt templates."""
        default_templates = {
            "system_reasoning": PromptTemplate(
                name="system_reasoning",
                template="""You are an intelligent system adaptation agent for the POLARIS framework.

Your role is to analyze system state, reason about potential adaptations, and recommend actions.

Current Context:
- System ID: {system_id}
- Current Metrics: {current_metrics}
- System Health: {health_status}
- Available Actions: {available_actions}

Guidelines:
1. Analyze the current system state and identify any issues or optimization opportunities
2. Consider historical patterns and learned behaviors
3. Recommend specific adaptation actions with clear rationale
4. Provide confidence scores for your recommendations
5. Consider potential risks and side effects of proposed actions

Please provide your analysis and recommendations in JSON format:
{{
    "analysis": "Your analysis of the current situation",
    "recommendations": [
        {{
            "action_type": "action_name",
            "parameters": {{}},
            "rationale": "Why this action is recommended",
            "confidence": 0.85,
            "risk_level": "low|medium|high"
        }}
    ],
    "reasoning_steps": ["step1", "step2", "step3"]
}}""",
                variables=["system_id", "current_metrics", "health_status", "available_actions"]
            ),
            
            "world_model_update": PromptTemplate(
                name="world_model_update",
                template="""You are maintaining a digital twin world model for system: {system_id}

Current system state update:
{system_state}

Previous world model state:
{previous_state}

Please update the world model with this new information and provide:
1. Updated system understanding
2. Identified patterns or trends
3. Potential future states
4. Confidence in predictions

Respond in JSON format:
{{
    "updated_model": {{
        "system_description": "Natural language description of current system state",
        "key_metrics": {{}},
        "identified_patterns": [],
        "trend_analysis": {{}}
    }},
    "predictions": {{
        "short_term": "Next 5-10 minutes",
        "medium_term": "Next 30-60 minutes", 
        "confidence_scores": {{}}
    }}
}}""",
                variables=["system_id", "system_state", "previous_state"]
            ),
            
            "adaptation_impact_simulation": PromptTemplate(
                name="adaptation_impact_simulation",
                template="""Simulate the impact of the following adaptation action on system: {system_id}

Proposed Action:
- Type: {action_type}
- Parameters: {action_parameters}

Current System State:
{current_state}

System Dependencies:
{dependencies}

Please simulate the potential impact and provide:
1. Expected immediate effects
2. Potential cascading effects
3. Risk assessment
4. Success probability

Respond in JSON format:
{{
    "simulation_result": {{
        "immediate_effects": {{}},
        "cascading_effects": [],
        "risk_assessment": {{
            "level": "low|medium|high",
            "factors": []
        }},
        "success_probability": 0.85,
        "alternative_outcomes": []
    }},
    "recommendations": {{
        "proceed": true,
        "modifications": [],
        "precautions": []
    }}
}}""",
                variables=["system_id", "action_type", "action_parameters", "current_state", "dependencies"]
            ),
            
            "agentic_reasoning": PromptTemplate(
                name="agentic_reasoning",
                template="""You are an agentic reasoning system for POLARIS adaptation framework.

Objective: {reasoning_objective}

Available Tools:
{available_tools}

Current Context:
{context}

You can use the available tools to gather information, analyze data, and make informed decisions.
Think step by step and use tools as needed to accomplish your objective.

When you need to use a tool, respond with:
{{
    "action": "use_tool",
    "tool_name": "tool_name",
    "parameters": {{}}
}}

When you have enough information to provide a final answer, respond with:
{{
    "action": "final_answer",
    "result": {{}},
    "reasoning": "Your reasoning process",
    "confidence": 0.85
}}""",
                variables=["reasoning_objective", "available_tools", "context"]
            )
        }
        
        for template in default_templates.values():
            self._templates[template.name] = template
    
    def _load_templates_from_directory(self, directory: Path) -> None:
        """Load templates from YAML files in directory."""
        try:
            for file_path in directory.glob("*.yaml"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    
                if "templates" in data:
                    for template_data in data["templates"]:
                        template = PromptTemplate(
                            name=template_data["name"],
                            template=template_data["template"],
                            variables=template_data.get("variables", []),
                            metadata=template_data.get("metadata", {})
                        )
                        self._templates[template.name] = template
                        
                if "flows" in data:
                    for flow_data in data["flows"]:
                        flow = self._create_flow_from_data(flow_data)
                        self._flows[flow.flow_id] = flow
                        
        except Exception as e:
            self.logger.error(f"Failed to load templates from {directory}: {e}")
            raise LLMConfigurationError(
                f"Failed to load prompt templates: {str(e)}",
                config_key="template_directory",
                config_value=str(directory),
                cause=e
            )
    
    def _create_flow_from_data(self, flow_data: Dict[str, Any]) -> ConversationFlow:
        """Create ConversationFlow from configuration data."""
        initial_prompt = PromptTemplate(
            name=f"{flow_data['flow_id']}_initial",
            template=flow_data["initial_prompt"]["template"],
            variables=flow_data["initial_prompt"].get("variables", [])
        )
        
        system_prompt = None
        if "system_prompt" in flow_data:
            system_prompt = PromptTemplate(
                name=f"{flow_data['flow_id']}_system",
                template=flow_data["system_prompt"]["template"],
                variables=flow_data["system_prompt"].get("variables", [])
            )
        
        return ConversationFlow(
            flow_id=flow_data["flow_id"],
            name=flow_data["name"],
            description=flow_data["description"],
            initial_prompt=initial_prompt,
            system_prompt=system_prompt,
            available_tools=flow_data.get("available_tools", []),
            max_iterations=flow_data.get("max_iterations", 10),
            completion_criteria=flow_data.get("completion_criteria", {}),
            metadata=flow_data.get("metadata", {})
        )
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a prompt template by name."""
        return self._templates.get(name)
    
    def register_template(self, template: PromptTemplate) -> None:
        """Register a new prompt template."""
        self._templates[template.name] = template
        self.logger.debug(f"Registered prompt template: {template.name}")
    
    def get_flow(self, flow_id: str) -> Optional[ConversationFlow]:
        """Get a conversation flow by ID."""
        return self._flows.get(flow_id)
    
    def register_flow(self, flow: ConversationFlow) -> None:
        """Register a new conversation flow."""
        self._flows[flow.flow_id] = flow
        self.logger.debug(f"Registered conversation flow: {flow.flow_id}")
    
    def render_template(self, template_name: str, **kwargs) -> str:
        """Render a template with provided variables."""
        template = self.get_template(template_name)
        if not template:
            raise LLMConfigurationError(
                f"Template not found: {template_name}",
                config_key="template_name",
                config_value=template_name
            )
        
        try:
            return template.render(**kwargs)
        except KeyError as e:
            missing_var = str(e).strip("'\"")
            raise LLMConfigurationError(
                f"Missing template variable: {missing_var}",
                config_key="template_variables",
                config_value=missing_var,
                cause=e
            )
    
    def create_system_message(self, template_name: str, **kwargs) -> Message:
        """Create a system message from a template."""
        content = self.render_template(template_name, **kwargs)
        return Message(
            role=MessageRole.SYSTEM,
            content=content,
            metadata={"template": template_name, "variables": kwargs}
        )
    
    def create_user_message(self, template_name: str, **kwargs) -> Message:
        """Create a user message from a template."""
        content = self.render_template(template_name, **kwargs)
        return Message(
            role=MessageRole.USER,
            content=content,
            metadata={"template": template_name, "variables": kwargs}
        )
    
    def build_conversation_context(
        self,
        system_context: Dict[str, Any],
        user_query: str,
        conversation_history: Optional[List[Message]] = None
    ) -> List[Message]:
        """Build conversation context with system and user messages."""
        messages = []
        
        # Add system message if context provided
        if system_context:
            system_content = self.render_template("system_reasoning", **system_context)
            messages.append(Message(
                role=MessageRole.SYSTEM,
                content=system_content,
                metadata={"context": system_context}
            ))
        
        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add user query
        messages.append(Message(
            role=MessageRole.USER,
            content=user_query
        ))
        
        return messages
    
    def start_agentic_conversation(
        self,
        flow_id: str,
        context: Dict[str, Any],
        **kwargs
    ) -> AgenticConversation:
        """Start a new agentic conversation using a flow."""
        flow = self.get_flow(flow_id)
        if not flow:
            raise LLMConfigurationError(
                f"Conversation flow not found: {flow_id}",
                config_key="flow_id",
                config_value=flow_id
            )
        
        conversation = AgenticConversation(
            context=context,
            max_iterations=flow.max_iterations
        )
        
        # Add system message if flow has one
        if flow.system_prompt:
            system_content = flow.system_prompt.render(**context, **kwargs)
            conversation.add_message(Message(
                role=MessageRole.SYSTEM,
                content=system_content,
                metadata={"flow_id": flow_id, "template": flow.system_prompt.name}
            ))
        
        # Add initial prompt
        initial_content = flow.initial_prompt.render(**context, **kwargs)
        conversation.add_message(Message(
            role=MessageRole.USER,
            content=initial_content,
            metadata={"flow_id": flow_id, "template": flow.initial_prompt.name}
        ))
        
        return conversation
    
    def list_templates(self) -> List[str]:
        """List all available template names."""
        return list(self._templates.keys())
    
    def list_flows(self) -> List[str]:
        """List all available flow IDs."""
        return list(self._flows.keys())
    
    def export_templates(self, file_path: Path) -> None:
        """Export all templates to a YAML file."""
        data = {
            "templates": [
                {
                    "name": template.name,
                    "template": template.template,
                    "variables": template.variables,
                    "metadata": template.metadata
                }
                for template in self._templates.values()
            ],
            "flows": [
                {
                    "flow_id": flow.flow_id,
                    "name": flow.name,
                    "description": flow.description,
                    "initial_prompt": {
                        "template": flow.initial_prompt.template,
                        "variables": flow.initial_prompt.variables
                    },
                    "system_prompt": {
                        "template": flow.system_prompt.template,
                        "variables": flow.system_prompt.variables
                    } if flow.system_prompt else None,
                    "available_tools": flow.available_tools,
                    "max_iterations": flow.max_iterations,
                    "completion_criteria": flow.completion_criteria,
                    "metadata": flow.metadata
                }
                for flow in self._flows.values()
            ]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)