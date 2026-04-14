#!/usr/bin/env python3
"""Dependency policy guard for Polaris.

This check prevents dependency metadata drift across pyproject, setup shim,
and shared constraints used by local/CI/Docker installs.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path
from typing import Dict, Iterable, List, Set, cast

try:
    from packaging.requirements import Requirement
except Exception:  # pragma: no cover
    Requirement = None


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
SETUP = ROOT / "setup.py"
CONSTRAINTS = ROOT / "requirements" / "constraints.txt"

RUNTIME_GROUPS = ("dependencies", "llm", "dashboard", "connectors", "suave")


def _parse_req_name(requirement: str) -> str:
    if Requirement is not None:
        return cast(str, Requirement(requirement).name.lower())

    token = re.split(r"[<>=!~;\s\[]", requirement, maxsplit=1)[0]
    return token.strip().lower()


def _has_upper_bound(requirement: str) -> bool:
    if Requirement is None:
        return "<" in requirement

    parsed = Requirement(requirement)
    return any(spec.operator in {"<", "<="} for spec in parsed.specifier)


def _duplicates(requirements: Iterable[str]) -> Set[str]:
    seen: Set[str] = set()
    dupes: Set[str] = set()
    for req in requirements:
        name = _parse_req_name(req)
        if name in seen:
            dupes.add(name)
        seen.add(name)
    return dupes


def _load_pyproject() -> Dict[str, object]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data.get("project")
    if not isinstance(project, dict):
        raise ValueError("Missing [project] table in pyproject.toml")
    return project


def _read_constraint_names() -> Set[str]:
    if not CONSTRAINTS.exists():
        raise ValueError(f"Missing constraints file: {CONSTRAINTS}")

    names: Set[str] = set()
    for raw in CONSTRAINTS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-", "--")):
            continue
        names.add(_parse_req_name(line))
    return names


def main() -> int:
    errors: List[str] = []

    try:
        project = _load_pyproject()
    except Exception as exc:
        print(f"[FAIL] pyproject parse failed: {exc}")
        return 1

    scripts = project.get("scripts")
    if not isinstance(scripts, dict) or scripts.get("polaris") != "polaris.cli.main:main":
        errors.append("[project.scripts] must define polaris = 'polaris.cli.main:main'")

    deps = project.get("dependencies")
    if not isinstance(deps, list):
        errors.append("[project].dependencies must be a list")
        deps = []

    optional = project.get("optional-dependencies")
    if not isinstance(optional, dict):
        errors.append("[project.optional-dependencies] must be a table")
        optional = {}

    # Enforce upper bounds for runtime/feature dependencies to reduce breakage risk.
    if isinstance(deps, list):
        missing_caps = [req for req in deps if not _has_upper_bound(req)]
        if missing_caps:
            errors.append(
                "runtime dependencies missing upper bounds: " + ", ".join(sorted(missing_caps))
            )

    for group in ("llm", "dashboard", "connectors", "suave"):
        items = optional.get(group)
        if not isinstance(items, list):
            errors.append(f"optional dependency group '{group}' must exist and be a list")
            continue
        missing_caps = [req for req in items if not _has_upper_bound(req)]
        if missing_caps:
            errors.append(
                f"optional group '{group}' has entries without upper bounds: "
                + ", ".join(sorted(missing_caps))
            )
        dupes = _duplicates(items)
        if dupes:
            errors.append(f"optional group '{group}' has duplicate packages: {sorted(dupes)}")

    if isinstance(deps, list):
        dupes = _duplicates(deps)
        if dupes:
            errors.append(f"runtime dependencies have duplicate packages: {sorted(dupes)}")

    setup_content = SETUP.read_text(encoding="utf-8")
    forbidden_setup_tokens = (
        "install_requires",
        "extras_require",
        "entry_points",
        "python_requires",
        "find_packages(",
    )
    violations = [token for token in forbidden_setup_tokens if token in setup_content]
    if violations:
        errors.append(
            "setup.py must remain a metadata shim without duplicated dependency fields. "
            f"Found tokens: {', '.join(violations)}"
        )

    try:
        constraint_names = _read_constraint_names()
    except Exception as exc:
        errors.append(str(exc))
        constraint_names = set()

    expected_names: Set[str] = set()
    if isinstance(deps, list):
        expected_names.update(_parse_req_name(req) for req in deps)

    for group in ("llm", "dashboard", "connectors", "suave", "dev"):
        items = optional.get(group)
        if isinstance(items, list):
            expected_names.update(_parse_req_name(req) for req in items)

    missing_from_constraints = sorted(
        name for name in expected_names if name not in constraint_names
    )
    if missing_from_constraints:
        errors.append(
            "constraints.txt missing packages from pyproject: "
            + ", ".join(missing_from_constraints)
        )

    if errors:
        print("Dependency standardization check failed:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("Dependency standardization check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
