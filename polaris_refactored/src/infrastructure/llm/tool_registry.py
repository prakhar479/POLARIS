"""
Tool Registry System

Manages available tools for LLM agents including tool schemas,
execution, validation, and result handling.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Awaitable
from datetime import datetime, timezone

from .models import ToolSchema, ToolResult, ToolCall, ToolCallStatus
from .exceptions import LLMToolError


class BaseTool(ABC):
    """Abstract base class for LLM agent tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def get_schema(self) -> ToolSchema:
        """Get the tool's schema definition."""
        pass
    
    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given parameters."""
        pass
    
    async def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate tool parameters (default implementation)."""
        schema = self.get_schema()
        
        # Check required parameters
        for required_param in schema.required:
            if required_param not in parameters:
                raise LLMToolError(
                    f"Missing required parameter: {required_param}",
                    tool_name=self.name,
                    tool_parameters=parameters
                )
        
        return True
    
    def get_openai_format(self) -> Dict[str, Any]:
        """Get tool schema in OpenAI function calling format."""
        return self.get_schema().to_openai_format()
    
    def get_anthropic_format(self) -> Dict[str, Any]:
        """Get tool schema in Anthropic tool format."""
        return self.get_schema().to_anthropic_format()


class ToolRegistry:
    """Registry for managing available tools and their execution."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._tools: Dict[str, BaseTool] = {}
        self._execution_history: List[ToolResult] = []
        self._max_history_size = 1000
    
    def register_tool(self, tool: BaseTool) -> None:
        """Register a new tool."""
        self._tools[tool.name] = tool
        self.logger.info(f"Registered tool: {tool.name}")
    
    def unregister_tool(self, tool_name: str) -> None:
        """Unregister a tool."""
        if tool_name in self._tools:
            del self._tools[tool_name]
            self.logger.info(f"Unregistered tool: {tool_name}")
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(tool_name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def get_tool_schemas(self, format_type: str = "openai") -> List[Dict[str, Any]]:
        """Get all tool schemas in specified format."""
        schemas = []
        for tool in self._tools.values():
            if format_type.lower() == "openai":
                schemas.append(tool.get_openai_format())
            elif format_type.lower() == "anthropic":
                schemas.append(tool.get_anthropic_format())
            else:
                schemas.append(tool.get_schema().to_openai_format())  # Default to OpenAI
        return schemas
    
    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        call_id: Optional[str] = None
    ) -> ToolResult:
        """Execute a tool and return the result."""
        tool = self.get_tool(tool_name)
        if not tool:
            error_msg = f"Tool not found: {tool_name}"
            self.logger.error(error_msg)
            return ToolResult(
                call_id=call_id or "unknown",
                tool_name=tool_name,
                success=False,
                result={},
                error_message=error_msg
            )
        
        start_time = datetime.now(timezone.utc)
        
        try:
            # Validate parameters
            await tool.validate_parameters(parameters)
            
            # Execute tool
            result_data = await tool.execute(parameters)
            
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            result = ToolResult(
                call_id=call_id or "unknown",
                tool_name=tool_name,
                success=True,
                result=result_data,
                execution_time=execution_time
            )
            
            self._add_to_history(result)
            self.logger.debug(f"Tool {tool_name} executed successfully in {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            error_msg = str(e)
            
            result = ToolResult(
                call_id=call_id or "unknown",
                tool_name=tool_name,
                success=False,
                result={},
                error_message=error_msg,
                execution_time=execution_time
            )
            
            self._add_to_history(result)
            self.logger.error(f"Tool {tool_name} execution failed: {error_msg}")
            
            return result
    
    async def execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """Execute a ToolCall object and return the result."""
        return await self.execute_tool(
            tool_name=tool_call.tool_name,
            parameters=tool_call.parameters,
            call_id=tool_call.call_id
        )
    
    async def execute_multiple_tools(
        self,
        tool_calls: List[ToolCall],
        parallel: bool = True
    ) -> List[ToolResult]:
        """Execute multiple tool calls."""
        if parallel:
            # Execute tools in parallel
            tasks = [self.execute_tool_call(tool_call) for tool_call in tool_calls]
            return await asyncio.gather(*tasks, return_exceptions=False)
        else:
            # Execute tools sequentially
            results = []
            for tool_call in tool_calls:
                result = await self.execute_tool_call(tool_call)
                results.append(result)
            return results
    
    def _add_to_history(self, result: ToolResult) -> None:
        """Add result to execution history."""
        self._execution_history.append(result)
        
        # Trim history if it exceeds max size
        if len(self._execution_history) > self._max_history_size:
            self._execution_history = self._execution_history[-self._max_history_size:]
    
    def get_execution_history(
        self,
        tool_name: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[ToolResult]:
        """Get execution history, optionally filtered by tool name."""
        history = self._execution_history
        
        if tool_name:
            history = [r for r in history if r.tool_name == tool_name]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def get_tool_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics about tool usage."""
        stats = {}
        
        for tool_name in self._tools.keys():
            tool_history = [r for r in self._execution_history if r.tool_name == tool_name]
            
            if tool_history:
                successful = [r for r in tool_history if r.success]
                failed = [r for r in tool_history if not r.success]
                
                avg_execution_time = sum(r.execution_time for r in tool_history) / len(tool_history)
                
                stats[tool_name] = {
                    "total_calls": len(tool_history),
                    "successful_calls": len(successful),
                    "failed_calls": len(failed),
                    "success_rate": len(successful) / len(tool_history) if tool_history else 0,
                    "average_execution_time": avg_execution_time,
                    "last_used": max(r.completed_at for r in tool_history) if tool_history else None
                }
            else:
                stats[tool_name] = {
                    "total_calls": 0,
                    "successful_calls": 0,
                    "failed_calls": 0,
                    "success_rate": 0,
                    "average_execution_time": 0,
                    "last_used": None
                }
        
        return stats
    
    def clear_history(self) -> None:
        """Clear execution history."""
        self._execution_history.clear()
        self.logger.info("Tool execution history cleared")


# Example tool implementations for common use cases

class EchoTool(BaseTool):
    """Simple echo tool for testing."""
    
    def __init__(self):
        super().__init__("echo", "Echo back the provided message")
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "message": {
                    "type": "string",
                    "description": "Message to echo back"
                }
            },
            required=["message"]
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        message = parameters["message"]
        return {"echoed_message": message, "timestamp": datetime.now(timezone.utc).isoformat()}


class CalculatorTool(BaseTool):
    """Simple calculator tool for basic arithmetic."""
    
    def __init__(self):
        super().__init__("calculator", "Perform basic arithmetic operations")
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters={
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "Arithmetic operation to perform"
                },
                "a": {
                    "type": "number",
                    "description": "First operand"
                },
                "b": {
                    "type": "number",
                    "description": "Second operand"
                }
            },
            required=["operation", "a", "b"]
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        operation = parameters["operation"]
        a = parameters["a"]
        b = parameters["b"]
        
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                raise LLMToolError(
                    "Division by zero",
                    tool_name=self.name,
                    tool_parameters=parameters
                )
            result = a / b
        else:
            raise LLMToolError(
                f"Unknown operation: {operation}",
                tool_name=self.name,
                tool_parameters=parameters
            )
        
        return {
            "operation": operation,
            "operands": [a, b],
            "result": result
        }


def create_default_registry() -> ToolRegistry:
    """Create a tool registry with default tools."""
    registry = ToolRegistry()
    
    # Register default tools
    registry.register_tool(EchoTool())
    registry.register_tool(CalculatorTool())
    
    return registry