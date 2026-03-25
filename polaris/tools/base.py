"""Base abstractions for POLARIS tools.

This module defines the core interfaces for tools used by agentic strategies. Tools are
stateless, reusable components that provide specific capabilities to adaptation
strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from polaris.abstractions.connector import Connector
    from polaris.abstractions.knowledge_store import KnowledgeStore
    from polaris.abstractions.observability import Logger, MetricsCollector
    from polaris.abstractions.strategy import AdaptationContext
    from polaris.abstractions.world_model import WorldModel
    from polaris.core.models import SystemState


@dataclass
class ToolDependencies:
    """Dependencies injected into tools at execution time.

    Tools receive their dependencies through this object rather than being initialized
    with them, keeping tools stateless and reusable.

    Attributes:
        knowledge_store: Store for querying historical system data
        world_model: Model for predicting action outcomes
        connector: Optional connector to the managed system
        logger: Optional structured logger
        metrics: Metrics collector for observability
    """

    knowledge_store: "KnowledgeStore"
    world_model: "WorldModel"
    metrics: Optional["MetricsCollector"] = None
    connector: Optional["Connector"] = None
    logger: Optional["Logger"] = None


class ToolError(Exception):
    """Structured error information for tool execution failures.

    Attributes:
        code: Error code for programmatic handling
        message: Human-readable error description
        recoverable: Whether the error can be recovered from
        details: Additional error context
    """

    def __init__(  # noqa: B042
        self,
        code: str,
        message: str,
        recoverable: bool = True,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the ToolError.

        Args:
            code: Error code
            message: Error description
            recoverable: Whether it is a recoverable error
            details: Extra error context
        """
        super().__init__(message)
        self.code = code
        self.message = message
        self.recoverable = recoverable
        self.details = details

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format for LLM consumption."""
        result: Dict[str, Any] = {
            "error": self.message,
            "error_code": self.code,
            "recoverable": self.recoverable,
        }
        if self.details:
            result["details"] = self.details
        return result


class Tool(ABC):
    """Abstract base class for all POLARIS tools.

    Tools are stateless components that provide specific capabilities to adaptation
    strategies. They receive all necessary context and dependencies at execution time.

    Implementations should: 1. Define `name` and `description` properties 2. Implement
    `execute()` with proper error handling 3. Return structured dictionaries (not raw
    objects) 4. Use `ToolError` for failures

    Examples:
        ```
        class GetRecentStatesTool(Tool):
            @property
            def name(self) -> str:
                return "get_recent_states"
        ```

        ```
        @property
        def description(self) -> str:
            return "Query recent system states from the knowledge store"
        ```

        ```
        async def execute(self, args, state, context, deps):
            # Implementation here
            return {"states": [...]}
        ```
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier used in LLM prompts and configuration."""
        pass

    @property
    def description(self) -> str:
        """Human-readable description for LLM prompts.

        Override this to provide a more specific description.
        """
        return f"Tool: {self.name}"

    @abstractmethod
    async def execute(
        self,
        args: Dict[str, Any],
        state: "SystemState",
        context: "AdaptationContext",
        deps: ToolDependencies,
    ) -> Dict[str, Any]:
        """Execute the tool with the provided arguments and context.

        Args:
            args: Tool arguments from the LLM
            state: Current system state
            context: Adaptation context
            deps: Injected dependencies

        Returns:
            Dictionary with tool results or error information
        """
        pass
