"""Tests for `python -m polaris.cli` entrypoint behavior."""

import runpy
import sys
import types

import pytest


def test_cli_module_entrypoint_exits_with_main_return_code(monkeypatch):
    fake_main_module = types.ModuleType("polaris.cli.main")
    fake_main_module.main = lambda: 23  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "polaris.cli.main", fake_main_module)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("polaris.cli.__main__", run_name="__main__")

    assert exc_info.value.code == 23
