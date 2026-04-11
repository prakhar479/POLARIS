"""Tests for global tool factory registry."""

import polaris.tools.factories as tool_factories
from polaris.tools import build_registered_tools, register_tool_factory, registered_tool_types
from polaris.tools.base import Tool


class _CustomTool(Tool):
    @property
    def name(self) -> str:
        return "custom_test_tool_registry"

    @property
    def description(self) -> str:
        return "custom test tool"

    async def execute(self, args, state, context, deps):
        _ = (args, state, context, deps)
        return {"ok": True}


class _OverrideRecentStatesTool(Tool):
    @property
    def name(self) -> str:
        return "get_recent_states"

    @property
    def description(self) -> str:
        return "override get_recent_states"

    async def execute(self, args, state, context, deps):
        _ = (args, state, context, deps)
        return {"override": True}


def test_registered_tool_types_include_builtins():
    names = registered_tool_types()
    assert "get_recent_states" in names
    assert "get_action_history" in names


def test_build_registered_tools_by_allowed_subset():
    tools = build_registered_tools(["get_recent_states", "get_action_history"])
    names = sorted(tool.name for tool in tools)
    assert names == ["get_action_history", "get_recent_states"]


def test_register_tool_factory_enables_custom_tool_instantiation():
    register_tool_factory("custom_test_tool_registry", _CustomTool)

    tools = build_registered_tools(["custom_test_tool_registry"])

    assert len(tools) == 1
    assert tools[0].name == "custom_test_tool_registry"


def test_build_registered_tools_rejects_unknown_tool_name():
    try:
        build_registered_tools(["unknown_tool_for_registry_test"])
        assert False, "Expected ValueError for unknown tool"
    except ValueError as exc:
        assert "Unknown tool name" in str(exc)


def test_register_tool_factory_rejects_non_callable_factory():
    try:
        register_tool_factory("non_callable_factory_test", "not-callable")
        assert False, "Expected ValueError for non-callable factory"
    except ValueError as exc:
        assert "callable" in str(exc)


def test_build_registered_tools_normalizes_and_deduplicates_allowed_names():
    tools = build_registered_tools(
        [" get_recent_states ", "get_recent_states", "", "get_action_history"]
    )
    names = [tool.name for tool in tools]
    assert names == ["get_recent_states", "get_action_history"]


def test_pre_registered_override_survives_lazy_default_registration():
    saved_factories = dict(tool_factories._TOOL_FACTORIES)
    saved_registered = tool_factories._factories_registered
    try:
        tool_factories._TOOL_FACTORIES.clear()
        tool_factories._factories_registered = False

        register_tool_factory("get_recent_states", _OverrideRecentStatesTool)
        tools = build_registered_tools(["get_recent_states"])

        assert len(tools) == 1
        assert isinstance(tools[0], _OverrideRecentStatesTool)
    finally:
        tool_factories._TOOL_FACTORIES.clear()
        tool_factories._TOOL_FACTORIES.update(saved_factories)
        tool_factories._factories_registered = saved_registered
