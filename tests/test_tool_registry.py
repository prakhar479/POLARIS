"""Tests for tool registry."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from polaris.abstractions.observability import MetricsCollector
from polaris.abstractions.strategy import AdaptationContext
from polaris.core.models import SystemState
from polaris.tools.base import Tool, ToolDependencies, ToolError
from polaris.tools.registry import ToolRegistry


class DummyTool(Tool):
    name = "dummy_tool"
    description = "A dummy tool"

    async def execute(self, args, state, context, deps):
        return {"result": "success"}


class ErrorTool(Tool):
    name = "error_tool"
    description = "Throws an error"

    async def execute(self, args, state, context, deps):
        raise ValueError("Intentional error")


class TimeoutTool(Tool):
    name = "timeout_tool"
    description = "Times out"

    async def execute(self, args, state, context, deps):
        await asyncio.sleep(2)
        return {"result": "done"}


class ToolErrorTool(Tool):
    name = "tool_error_tool"
    description = "Throws ToolError"

    async def execute(self, args, state, context, deps):
        raise ToolError(code="test_code", message="test message", recoverable=True)


@pytest.fixture
def mock_metrics():
    return Mock(spec=MetricsCollector)


@pytest.fixture
def registry(mock_metrics):
    return ToolRegistry(metrics=mock_metrics)


def test_register_and_get(registry):
    tool = DummyTool()
    registry.register(tool)

    assert registry.has_tool("dummy_tool")
    assert registry.get("dummy_tool") == tool
    assert "dummy_tool" in registry.list_tools()
    assert tool in registry.all_tools()

    descriptions = registry.get_tool_descriptions()
    assert descriptions["dummy_tool"] == "A dummy tool"

    # Registering duplicate should raise ValueError
    with pytest.raises(ValueError, match="already registered"):
        registry.register(tool)


def test_register_all(registry):
    t1 = DummyTool()
    t2 = ErrorTool()
    registry.register_all([t1, t2])

    assert registry.has_tool("dummy_tool")
    assert registry.has_tool("error_tool")


def test_filter_by_allowed(registry):
    t1 = DummyTool()
    t2 = ErrorTool()
    registry.register_all([t1, t2])

    filtered = registry.filter_by_allowed(["dummy_tool"])
    assert len(filtered) == 1
    assert filtered[0] == t1

    all_allowed = registry.filter_by_allowed(None)
    assert len(all_allowed) == 2


@pytest.mark.asyncio
async def test_execute_success(registry):
    tool = DummyTool()
    registry.register(tool)

    state = Mock(spec=SystemState, system_id="sys1")
    context = Mock(spec=AdaptationContext)
    deps = Mock(spec=ToolDependencies)

    result = await registry.execute("dummy_tool", {}, state, context, deps)
    assert result == {"result": "success"}


@pytest.mark.asyncio
async def test_execute_unknown_tool(registry):
    state = Mock(spec=SystemState, system_id="sys1")
    context = Mock(spec=AdaptationContext)
    deps = Mock(spec=ToolDependencies)

    result = await registry.execute("non_existent", {}, state, context, deps)
    assert "error" in result or result.get("error_code") == "unknown_tool"


@pytest.mark.asyncio
async def test_execute_timeout(registry):
    tool = TimeoutTool()
    registry.register(tool)

    state = Mock(spec=SystemState, system_id="sys1")
    context = Mock(spec=AdaptationContext)
    deps = Mock(spec=ToolDependencies, logger=Mock())

    result = await registry.execute("timeout_tool", {}, state, context, deps, timeout=0.1)
    assert result.get("error_code") == "timeout"


@pytest.mark.asyncio
async def test_execute_exception(registry):
    tool = ErrorTool()
    registry.register(tool)

    state = Mock(spec=SystemState, system_id="sys1")
    context = Mock(spec=AdaptationContext)
    deps = Mock(spec=ToolDependencies, logger=Mock())

    result = await registry.execute("error_tool", {}, state, context, deps)
    assert result.get("error_code") == "execution_error"


@pytest.mark.asyncio
async def test_execute_tool_error(registry):
    tool = ToolErrorTool()
    registry.register(tool)

    state = Mock(spec=SystemState, system_id="sys1")
    context = Mock(spec=AdaptationContext)
    deps = Mock(spec=ToolDependencies, logger=Mock())

    result = await registry.execute("tool_error_tool", {}, state, context, deps)
    assert result.get("error_code") == "test_code"
