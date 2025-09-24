"""
Conversation Management System

Manages multi-turn agentic conversations with tool usage tracking,
context management, and conversation flow control.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable
from datetime import datetime, timezone
import uuid

from .models import (
    AgenticConversation, Message, MessageRole, ToolCall, ToolCallStatus,
    LLMRequest, LLMResponse
)
from .tool_registry import ToolRegistry, ToolResult
from .client import LLMClient
from .response_parser import ResponseParser
from .exceptions import LLMToolError, LLMIntegrationError


class Conversation:
    """Represents a single conversation with metadata and state tracking."""
    
    def __init__(
        self,
        conversation_id: Optional[str] = None,
        max_messages: int = 100,
        context: Optional[Dict[str, Any]] = None
    ):
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.messages: List[Message] = []
        self.max_messages = max_messages
        self.context = context or {}
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        self.metadata: Dict[str, Any] = {}
    
    def add_message(self, message: Message) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
        self.updated_at = datetime.now(timezone.utc)
        
        # Trim messages if exceeding max
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_messages(self, role: Optional[MessageRole] = None) -> List[Message]:
        """Get messages, optionally filtered by role."""
        if role:
            return [msg for msg in self.messages if msg.role == role]
        return self.messages.copy()
    
    def get_last_message(self, role: Optional[MessageRole] = None) -> Optional[Message]:
        """Get the last message, optionally filtered by role."""
        messages = self.get_messages(role)
        return messages[-1] if messages else None
    
    def clear_messages(self) -> None:
        """Clear all messages from the conversation."""
        self.messages.clear()
        self.updated_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation to dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "messages": [
                {
                    "role": msg.role.value,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "metadata": msg.metadata
                }
                for msg in self.messages
            ],
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }


class ConversationManager:
    """Manages multiple conversations and coordinates LLM interactions with tool usage."""
    
    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        response_parser: ResponseParser
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.response_parser = response_parser
        self.logger = logging.getLogger(__name__)
        
        self._conversations: Dict[str, Conversation] = {}
        self._agentic_conversations: Dict[str, AgenticConversation] = {}
        self._max_conversations = 1000
    
    def create_conversation(
        self,
        conversation_id: Optional[str] = None,
        max_messages: int = 100,
        context: Optional[Dict[str, Any]] = None
    ) -> Conversation:
        """Create a new conversation."""
        conversation = Conversation(
            conversation_id=conversation_id,
            max_messages=max_messages,
            context=context
        )
        
        self._conversations[conversation.conversation_id] = conversation
        
        # Trim conversations if exceeding max
        if len(self._conversations) > self._max_conversations:
            oldest_id = min(self._conversations.keys(), 
                          key=lambda k: self._conversations[k].created_at)
            del self._conversations[oldest_id]
        
        self.logger.debug(f"Created conversation: {conversation.conversation_id}")
        return conversation
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        return self._conversations.get(conversation_id)
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            self.logger.debug(f"Deleted conversation: {conversation_id}")
            return True
        return False
    
    async def send_message(
        self,
        conversation_id: str,
        content: str,
        role: MessageRole = MessageRole.USER,
        generate_response: bool = True
    ) -> Optional[Message]:
        """Send a message to a conversation and optionally generate LLM response."""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            raise LLMIntegrationError(
                f"Conversation not found: {conversation_id}",
                context={"conversation_id": conversation_id}
            )
        
        # Add user message
        user_message = Message(role=role, content=content)
        conversation.add_message(user_message)
        
        if generate_response:
            return await self._generate_response(conversation)
        
        return None
    
    async def _generate_response(self, conversation: Conversation) -> Message:
        """Generate LLM response for a conversation."""
        # Create LLM request
        request = LLMRequest(
            messages=conversation.messages,
            model_name=self.llm_client.config.model_name,
            max_tokens=self.llm_client.config.max_tokens,
            temperature=self.llm_client.config.temperature
        )
        
        try:
            # Generate response
            response = await self.llm_client.generate_response(request)
            
            # Create assistant message
            assistant_message = Message(
                role=MessageRole.ASSISTANT,
                content=response.content,
                metadata={"response_id": response.response_id, "usage": response.usage}
            )
            
            conversation.add_message(assistant_message)
            return assistant_message
            
        except Exception as e:
            error_msg = f"Failed to generate response: {str(e)}"
            self.logger.error(error_msg)
            
            error_message = Message(
                role=MessageRole.ASSISTANT,
                content=f"I apologize, but I encountered an error: {error_msg}",
                metadata={"error": True, "error_message": error_msg}
            )
            
            conversation.add_message(error_message)
            return error_message
    
    def create_agentic_conversation(
        self,
        objective: str,
        available_tools: Optional[List[str]] = None,
        max_iterations: int = 10,
        context: Optional[Dict[str, Any]] = None
    ) -> AgenticConversation:
        """Create a new agentic conversation."""
        conversation = AgenticConversation(
            max_iterations=max_iterations,
            context=context or {}
        )
        
        # Set objective in context
        conversation.context["objective"] = objective
        
        # Set available tools
        if available_tools:
            conversation.context["available_tools"] = available_tools
        else:
            conversation.context["available_tools"] = self.tool_registry.list_tools()
        
        self._agentic_conversations[conversation.conversation_id] = conversation
        
        self.logger.debug(f"Created agentic conversation: {conversation.conversation_id}")
        return conversation
    
    def get_agentic_conversation(self, conversation_id: str) -> Optional[AgenticConversation]:
        """Get an agentic conversation by ID."""
        return self._agentic_conversations.get(conversation_id)
    
    async def run_agentic_conversation(
        self,
        conversation_id: str,
        initial_message: Optional[str] = None
    ) -> AgenticConversation:
        """Run an agentic conversation to completion."""
        conversation = self.get_agentic_conversation(conversation_id)
        if not conversation:
            raise LLMIntegrationError(
                f"Agentic conversation not found: {conversation_id}",
                context={"conversation_id": conversation_id}
            )
        
        # Add initial message if provided
        if initial_message:
            conversation.add_message(Message(
                role=MessageRole.USER,
                content=initial_message
            ))
        
        while not conversation.is_complete and conversation.current_iteration < conversation.max_iterations:
            try:
                await self._run_agentic_iteration(conversation)
            except Exception as e:
                self.logger.error(f"Error in agentic iteration: {str(e)}")
                conversation.add_reasoning_step(f"Error occurred: {str(e)}")
                break
        
        if not conversation.is_complete:
            conversation.complete({
                "status": "max_iterations_reached",
                "iterations": conversation.current_iteration,
                "reason": "Maximum iterations reached without completion"
            })
        
        return conversation
    
    async def _run_agentic_iteration(self, conversation: AgenticConversation) -> None:
        """Run a single iteration of agentic conversation."""
        conversation.increment_iteration()
        conversation.add_reasoning_step(f"Starting iteration {conversation.current_iteration}")
        
        # Prepare tools for LLM
        available_tools = conversation.context.get("available_tools", [])
        tools = []
        if available_tools:
            for tool_name in available_tools:
                tool = self.tool_registry.get_tool(tool_name)
                if tool:
                    tools.append(tool.get_openai_format())
        
        # Create LLM request
        request = LLMRequest(
            messages=conversation.messages,
            model_name=self.llm_client.config.model_name,
            max_tokens=self.llm_client.config.max_tokens,
            temperature=self.llm_client.config.temperature,
            tools=tools if tools else None
        )
        
        # Generate response
        response = await self.llm_client.generate_response(request)
        
        # Parse response
        parsed = self.response_parser.parse_agentic_response(response)
        
        # Add assistant message
        assistant_message = Message(
            role=MessageRole.ASSISTANT,
            content=response.content,
            metadata={"parsed": parsed}
        )
        conversation.add_message(assistant_message)
        
        # Handle function calls
        if response.function_calls:
            await self._handle_function_calls(conversation, response.function_calls)
        
        # Handle tool usage from text
        elif parsed.get("tool_usage"):
            await self._handle_text_tool_usage(conversation, parsed["tool_usage"])
        
        # Check for completion
        elif parsed.get("tool_usage", {}).get("action") == "final_answer":
            result = parsed["tool_usage"].get("result", {})
            conversation.complete(result)
    
    async def _handle_function_calls(
        self,
        conversation: AgenticConversation,
        function_calls: List
    ) -> None:
        """Handle function calls from LLM response."""
        for func_call in function_calls:
            tool_call = ToolCall(
                tool_name=func_call.name,
                parameters=func_call.arguments,
                call_id=func_call.call_id,
                status=ToolCallStatus.EXECUTING
            )
            
            conversation.add_tool_call(tool_call)
            conversation.add_reasoning_step(f"Executing tool: {func_call.name}")
            
            # Execute tool
            result = await self.tool_registry.execute_tool_call(tool_call)
            
            # Update tool call status
            tool_call.status = ToolCallStatus.SUCCESS if result.success else ToolCallStatus.FAILED
            tool_call.result = result.result
            tool_call.error_message = result.error_message
            
            # Add tool result message
            tool_message = Message(
                role=MessageRole.TOOL,
                content=str(result.result) if result.success else f"Error: {result.error_message}",
                tool_call_id=func_call.call_id,
                metadata={"tool_result": result.result, "success": result.success}
            )
            conversation.add_message(tool_message)
    
    async def _handle_text_tool_usage(
        self,
        conversation: AgenticConversation,
        tool_usage: Dict[str, Any]
    ) -> None:
        """Handle tool usage instructions from text content."""
        if tool_usage.get("action") == "use_tool":
            tool_name = tool_usage.get("tool_name")
            parameters = tool_usage.get("parameters", {})
            
            if tool_name:
                conversation.add_reasoning_step(f"Using tool: {tool_name}")
                
                # Execute tool
                result = await self.tool_registry.execute_tool(tool_name, parameters)
                
                # Add tool result as user message for next iteration
                result_message = Message(
                    role=MessageRole.USER,
                    content=f"Tool '{tool_name}' result: {result.result if result.success else f'Error: {result.error_message}'}",
                    metadata={"tool_result": True, "tool_name": tool_name}
                )
                conversation.add_message(result_message)
    
    def list_conversations(self) -> List[str]:
        """List all conversation IDs."""
        return list(self._conversations.keys())
    
    def list_agentic_conversations(self) -> List[str]:
        """List all agentic conversation IDs."""
        return list(self._agentic_conversations.keys())
    
    def get_conversation_statistics(self) -> Dict[str, Any]:
        """Get statistics about conversations."""
        return {
            "total_conversations": len(self._conversations),
            "total_agentic_conversations": len(self._agentic_conversations),
            "active_conversations": len([
                c for c in self._agentic_conversations.values() 
                if not c.is_complete
            ]),
            "completed_conversations": len([
                c for c in self._agentic_conversations.values() 
                if c.is_complete
            ])
        }
    
    def clear_conversations(self, older_than_hours: Optional[int] = None) -> int:
        """Clear conversations, optionally only those older than specified hours."""
        if older_than_hours is None:
            count = len(self._conversations) + len(self._agentic_conversations)
            self._conversations.clear()
            self._agentic_conversations.clear()
            return count
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=older_than_hours)
        
        # Clear regular conversations
        old_conv_ids = [
            conv_id for conv_id, conv in self._conversations.items()
            if conv.updated_at < cutoff_time
        ]
        for conv_id in old_conv_ids:
            del self._conversations[conv_id]
        
        # Clear agentic conversations
        old_agentic_ids = [
            conv_id for conv_id, conv in self._agentic_conversations.items()
            if conv.updated_at < cutoff_time
        ]
        for conv_id in old_agentic_ids:
            del self._agentic_conversations[conv_id]
        
        total_cleared = len(old_conv_ids) + len(old_agentic_ids)
        self.logger.info(f"Cleared {total_cleared} conversations older than {older_than_hours} hours")
        
        return total_cleared