"""
LLM Integration Data Models

Defines data structures for LLM integration including configurations,
requests, responses, function calls, and agentic conversation models.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, field_validator
import uuid


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    MOCK = "mock"


class MessageRole(Enum):
    """Message roles in conversations."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCallStatus(Enum):
    """Tool call execution status."""
    PENDING = "pending"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


class LLMConfiguration(BaseModel):
    """LLM client configuration with validation."""
    provider: LLMProvider = LLMProvider.OPENAI
    api_endpoint: str = Field(min_length=1)
    api_key: Optional[str] = None
    model_name: str = Field(min_length=1)
    max_tokens: int = Field(default=1000, ge=1, le=32000)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    timeout: float = Field(default=30.0, ge=1.0, le=300.0)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay: float = Field(default=1.0, ge=0.1, le=60.0)
    cache_ttl: int = Field(default=300, ge=0, le=86400)  # seconds
    enable_function_calling: bool = True
    
    @field_validator('api_endpoint')
    @classmethod
    def validate_api_endpoint(cls, v):
        """Validate API endpoint URL."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError("API endpoint must be a valid HTTP/HTTPS URL")
        return v


@dataclass(frozen=True)
class Message:
    """Represents a message in a conversation."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_calls: List['ToolCall'] = field(default_factory=list)
    tool_call_id: Optional[str] = None


@dataclass(frozen=True)
class FunctionCall:
    """Represents a function call in LLM response."""
    name: str
    arguments: Dict[str, Any]
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass(frozen=True)
class ToolCall:
    """Represents a tool call with execution tracking."""
    tool_name: str
    parameters: Dict[str, Any]
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: ToolCallStatus = ToolCallStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None


@dataclass(frozen=True)
class LLMRequest:
    """Represents a request to an LLM."""
    messages: List[Message]
    model_name: str
    max_tokens: int = 1000
    temperature: float = 0.1
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[str] = None
    stream: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass(frozen=True)
class LLMResponse:
    """Represents a response from an LLM."""
    content: str
    model_name: str
    usage: Dict[str, int]
    function_calls: List[FunctionCall] = field(default_factory=list)
    finish_reason: str = "stop"
    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgenticConversation:
    """Represents an agentic conversation with tool usage tracking."""
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Message] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    max_iterations: int = 10
    current_iteration: int = 0
    is_complete: bool = False
    final_result: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def add_message(self, message: Message) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
        self.updated_at = datetime.now(timezone.utc)
    
    def add_tool_call(self, tool_call: ToolCall) -> None:
        """Add a tool call to the conversation."""
        self.tool_calls.append(tool_call)
        self.updated_at = datetime.now(timezone.utc)
    
    def add_reasoning_step(self, step: str) -> None:
        """Add a reasoning step to the trace."""
        self.reasoning_trace.append(step)
        self.updated_at = datetime.now(timezone.utc)
    
    def increment_iteration(self) -> None:
        """Increment the iteration counter."""
        self.current_iteration += 1
        self.updated_at = datetime.now(timezone.utc)
        
        if self.current_iteration >= self.max_iterations:
            self.is_complete = True
    
    def complete(self, result: Dict[str, Any]) -> None:
        """Mark the conversation as complete with a result."""
        self.is_complete = True
        self.final_result = result
        self.updated_at = datetime.now(timezone.utc)


@dataclass(frozen=True)
class ToolSchema:
    """Schema definition for a tool."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    required: List[str] = field(default_factory=list)
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required
                }
            }
        }
    
    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required
            }
        }


@dataclass(frozen=True)
class ToolResult:
    """Result of a tool execution."""
    call_id: str
    tool_name: str
    success: bool
    result: Dict[str, Any]
    error_message: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class PromptTemplate:
    """Template for generating prompts."""
    name: str
    template: str
    variables: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def render(self, **kwargs) -> str:
        """Render the template with provided variables."""
        return self.template.format(**kwargs)


@dataclass
class ConversationFlow:
    """Defines a conversation flow for agentic interactions."""
    flow_id: str
    name: str
    description: str
    initial_prompt: PromptTemplate
    system_prompt: Optional[PromptTemplate] = None
    available_tools: List[str] = field(default_factory=list)
    max_iterations: int = 10
    completion_criteria: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)