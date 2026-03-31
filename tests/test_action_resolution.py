"""Tests for strict connector-aware action resolution."""

from polaris.strategies.action_resolution import ConnectorActionResolver


def test_resolver_requires_supported_actions():
    resolver = ConnectorActionResolver()

    assert resolver.resolve_action_type("scale_up", None) is None
    assert resolver.resolve_action_type("scale_up", []) is None


def test_resolver_matches_supported_action_exactly():
    resolver = ConnectorActionResolver()
    supported = ["scale_up", "scale_down"]

    assert resolver.resolve_action_type("scale_up", supported) == "scale_up"
    assert resolver.resolve_action_type("SCALE-UP", supported) == "scale_up"


def test_resolver_honors_explicit_contract_aliases_only():
    resolver = ConnectorActionResolver()
    supported = ["expand_capacity", "shrink_capacity"]
    aliases = {"scale_up": "expand_capacity", "scale_down": "shrink_capacity"}

    assert resolver.resolve_action_type("scale_up", supported, aliases) == "expand_capacity"
    assert resolver.resolve_action_type("scale_down", supported, aliases) == "shrink_capacity"


def test_resolver_rejects_unknown_and_heuristic_tokens():
    resolver = ConnectorActionResolver()
    supported = ["restart_deployment", "scale_deployment"]

    assert resolver.resolve_action_type("restart deployment", supported) == "restart_deployment"
    assert resolver.resolve_action_type("please restart deployment now", supported) is None
    assert resolver.resolve_action_type("add_server", supported) is None
