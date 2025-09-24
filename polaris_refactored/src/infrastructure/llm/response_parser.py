"""
Response Parser System

Handles parsing and validation of LLM responses including JSON schema validation,
function call parsing, and structured output extraction.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Union
from jsonschema import validate, ValidationError as JSONValidationError

from .models import LLMResponse, FunctionCall, ToolCall, ToolCallStatus
from .exceptions import LLMResponseParsingError


class JSONSchemaValidator:
    """Validates JSON responses against schemas."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._schemas: Dict[str, Dict[str, Any]] = {}
        self._load_default_schemas()
    
    def _load_default_schemas(self) -> None:
        """Load default JSON schemas for common response types."""
        self._schemas = {
            "reasoning_result": {
                "type": "object",
                "properties": {
                    "analysis": {"type": "string"},
                    "recommendations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "action_type": {"type": "string"},
                                "parameters": {"type": "object"},
                                "rationale": {"type": "string"},
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                "risk_level": {"type": "string", "enum": ["low", "medium", "high"]}
                            },
                            "required": ["action_type", "parameters", "rationale", "confidence"]
                        }
                    },
                    "reasoning_steps": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["analysis", "recommendations"]
            },
            
            "world_model_update": {
                "type": "object",
                "properties": {
                    "updated_model": {
                        "type": "object",
                        "properties": {
                            "system_description": {"type": "string"},
                            "key_metrics": {"type": "object"},
                            "identified_patterns": {"type": "array"},
                            "trend_analysis": {"type": "object"}
                        },
                        "required": ["system_description"]
                    },
                    "predictions": {
                        "type": "object",
                        "properties": {
                            "short_term": {"type": "string"},
                            "medium_term": {"type": "string"},
                            "confidence_scores": {"type": "object"}
                        }
                    }
                },
                "required": ["updated_model"]
            },
            
            "simulation_result": {
                "type": "object",
                "properties": {
                    "simulation_result": {
                        "type": "object",
                        "properties": {
                            "immediate_effects": {"type": "object"},
                            "cascading_effects": {"type": "array"},
                            "risk_assessment": {
                                "type": "object",
                                "properties": {
                                    "level": {"type": "string", "enum": ["low", "medium", "high"]},
                                    "factors": {"type": "array"}
                                },
                                "required": ["level"]
                            },
                            "success_probability": {"type": "number", "minimum": 0, "maximum": 1},
                            "alternative_outcomes": {"type": "array"}
                        },
                        "required": ["immediate_effects", "risk_assessment", "success_probability"]
                    },
                    "recommendations": {
                        "type": "object",
                        "properties": {
                            "proceed": {"type": "boolean"},
                            "modifications": {"type": "array"},
                            "precautions": {"type": "array"}
                        },
                        "required": ["proceed"]
                    }
                },
                "required": ["simulation_result", "recommendations"]
            },
            
            "agentic_action": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["use_tool", "final_answer"]},
                    "tool_name": {"type": "string"},
                    "parameters": {"type": "object"},
                    "result": {"type": "object"},
                    "reasoning": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["action"]
            }
        }
    
    def register_schema(self, name: str, schema: Dict[str, Any]) -> None:
        """Register a new JSON schema."""
        self._schemas[name] = schema
        self.logger.debug(f"Registered JSON schema: {name}")
    
    def validate(self, data: Dict[str, Any], schema_name: str) -> bool:
        """Validate data against a registered schema."""
        if schema_name not in self._schemas:
            raise LLMResponseParsingError(
                f"Schema not found: {schema_name}",
                expected_format=schema_name
            )
        
        try:
            validate(instance=data, schema=self._schemas[schema_name])
            return True
        except JSONValidationError as e:
            raise LLMResponseParsingError(
                f"JSON validation failed: {e.message}",
                response_content=str(data),
                expected_format=schema_name,
                validation_errors=[e.message],
                cause=e
            )
    
    def get_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a registered schema by name."""
        return self._schemas.get(name)
    
    def list_schemas(self) -> List[str]:
        """List all registered schema names."""
        return list(self._schemas.keys())


class FunctionCallParser:
    """Parses function calls from LLM responses."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def parse_function_calls(self, response: LLMResponse) -> List[ToolCall]:
        """Parse function calls from LLM response into ToolCall objects."""
        tool_calls = []
        
        for func_call in response.function_calls:
            tool_call = ToolCall(
                tool_name=func_call.name,
                parameters=func_call.arguments,
                call_id=func_call.call_id,
                status=ToolCallStatus.PENDING
            )
            tool_calls.append(tool_call)
        
        return tool_calls
    
    def extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON object from text content."""
        # Try to find JSON in code blocks first
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        matches = re.findall(json_pattern, text, re.DOTALL | re.IGNORECASE)

        if matches:
            # Prefer the largest match (most likely the full JSON object) and try parsing
            matches_sorted = sorted(matches, key=lambda s: len(s), reverse=True)
            for match in matches_sorted:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        # If the entire text is valid JSON, prefer that (covers the common case
        # where the LLM returns a single JSON document without extra noise).
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try to find JSON object(s) in the text. There may be nested/braced objects;
        # collect all candidate brace-balanced-ish matches and prefer the largest so
        # inner-nested objects aren't chosen over the full outer object.
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)

        if matches:
            # Try longer matches first (more likely to be the full document)
            matches_sorted = sorted(matches, key=lambda s: len(s), reverse=True)
            for match in matches_sorted:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

        return None
    
    def parse_tool_usage_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse tool usage instructions from text content."""
        json_data = self.extract_json_from_text(text)
        
        if json_data and "action" in json_data:
            if json_data["action"] == "use_tool":
                return {
                    "action": "use_tool",
                    "tool_name": json_data.get("tool_name"),
                    "parameters": json_data.get("parameters", {})
                }
            elif json_data["action"] == "final_answer":
                return {
                    "action": "final_answer",
                    "result": json_data.get("result", {}),
                    "reasoning": json_data.get("reasoning", ""),
                    "confidence": json_data.get("confidence", 0.5)
                }
        
        return None


class ResponseParser:
    """Main response parser that coordinates JSON validation and function call parsing."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.json_validator = JSONSchemaValidator()
        self.function_parser = FunctionCallParser()
    
    def parse_response(
        self,
        response: LLMResponse,
        expected_schema: Optional[str] = None,
        extract_json: bool = True
    ) -> Dict[str, Any]:
        """Parse LLM response with optional schema validation."""
        result = {
            "content": response.content,
            "function_calls": [],
            "tool_calls": [],
            "parsed_json": None,
            "validation_passed": False
        }
        
        # Parse function calls
        if response.function_calls:
            result["function_calls"] = response.function_calls
            result["tool_calls"] = self.function_parser.parse_function_calls(response)
        
        # Extract and validate JSON if requested
        if extract_json:
            json_data = self.function_parser.extract_json_from_text(response.content)
            if json_data:
                result["parsed_json"] = json_data
                
                # Validate against schema if provided
                if expected_schema:
                    try:
                        self.json_validator.validate(json_data, expected_schema)
                        result["validation_passed"] = True
                    except LLMResponseParsingError as e:
                        self.logger.warning(f"JSON validation failed: {e.message}")
                        result["validation_error"] = str(e)
                else:
                    result["validation_passed"] = True
        
        return result
    
    def parse_reasoning_response(self, response: LLMResponse) -> Dict[str, Any]:
        """Parse reasoning response with specific schema validation."""
        return self.parse_response(response, expected_schema="reasoning_result")
    
    def parse_world_model_response(self, response: LLMResponse) -> Dict[str, Any]:
        """Parse world model update response with specific schema validation."""
        return self.parse_response(response, expected_schema="world_model_update")
    
    def parse_simulation_response(self, response: LLMResponse) -> Dict[str, Any]:
        """Parse simulation result response with specific schema validation."""
        return self.parse_response(response, expected_schema="simulation_result")
    
    def parse_agentic_response(self, response: LLMResponse) -> Dict[str, Any]:
        """Parse agentic conversation response."""
        result = self.parse_response(response, extract_json=True)
        
        # Check for tool usage instructions in text
        tool_usage = self.function_parser.parse_tool_usage_from_text(response.content)
        if tool_usage:
            result["tool_usage"] = tool_usage
            
            # Validate agentic action format
            try:
                self.json_validator.validate(tool_usage, "agentic_action")
                result["agentic_validation_passed"] = True
            except LLMResponseParsingError:
                result["agentic_validation_passed"] = False
        
        return result
    
    def extract_structured_data(
        self,
        response: LLMResponse,
        schema_name: str,
        required_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Extract and validate structured data from response."""
        json_data = self.function_parser.extract_json_from_text(response.content)
        
        if not json_data:
            raise LLMResponseParsingError(
                "No JSON data found in response",
                response_content=response.content,
                expected_format=schema_name
            )
        
        # Validate against schema
        self.json_validator.validate(json_data, schema_name)
        
        # Check required fields if specified
        if required_fields:
            missing_fields = [field for field in required_fields if field not in json_data]
            if missing_fields:
                raise LLMResponseParsingError(
                    f"Missing required fields: {missing_fields}",
                    response_content=response.content,
                    expected_format=schema_name,
                    validation_errors=[f"Missing field: {field}" for field in missing_fields]
                )
        
        return json_data
    
    def register_custom_schema(self, name: str, schema: Dict[str, Any]) -> None:
        """Register a custom JSON schema."""
        self.json_validator.register_schema(name, schema)
    
    def get_available_schemas(self) -> List[str]:
        """Get list of available schemas."""
        return self.json_validator.list_schemas()