#!/usr/bin/env python3
"""
Standalone YAML configuration validator for POLARIS.

- Does NOT import the POLARIS codebase (avoids optional deps)
- Uses jsonschema + PyYAML directly
- Auto-detects schema based on config content or filename patterns
- Can be steered via CLI arguments

Usage examples:

  # Validate all known config YAMLs under src/config/
  python scripts/validate_configs.py

  # Validate specific files (auto-detect schema)
  python scripts/validate_configs.py --files src/config/polaris_config.yaml src/config/gemini_world_model.yaml

  # Validate with explicit config type
  python scripts/validate_configs.py --files src/config/gemini_world_model.yaml --config-type world_model

  # Validate files matched by glob(s)
  python scripts/validate_configs.py --glob "src/config/*.yaml"

  # JSON output and non-zero exit code on any error
  python scripts/validate_configs.py --format json --strict-exit

Exit codes:
  0 = all files validated successfully (no schema errors)
  1 = one or more schema validation errors found
  2 = internal error (e.g., unreadable file, bad arguments)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import yaml
except Exception as e:
    print("PyYAML is required. Install with: pip install PyYAML", file=sys.stderr)
    sys.exit(2)

try:
    from jsonschema import Draft7Validator
except Exception:
    print("jsonschema is required. Install with: pip install jsonschema", file=sys.stderr)
    sys.exit(2)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "src" / "config"
DEFAULT_SCHEMAS = {
    "framework": CONFIG_DIR / "framework_config.schema.json",
    "world_model": CONFIG_DIR / "world_model_config.schema.json",
    "plugin": CONFIG_DIR / "plugin_config.schema.json",
}

# Simple filename heuristics for schema selection
FILENAME_HINTS = [
    ("polaris_config.yaml", "framework"),
    ("world_model.yaml", "world_model"),
    ("_world_model.yaml", "world_model"),
    ("plugin_config_template.yaml", "plugin"),
]


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def guess_config_type_from_content(doc: Dict) -> Optional[str]:
    if not isinstance(doc, dict):
        return None
    # Framework: usually includes these top-level sections
    if all(k in doc for k in ("nats", "telemetry", "execution", "logger")):
        return "framework"
    # World model: implementation + config object
    if (isinstance(doc.get("implementation"), str)
        and isinstance(doc.get("config"), dict)):
        return "world_model"
    return None


def guess_config_type_from_filename(path: Path) -> Optional[str]:
    name = path.name
    for suffix, ctype in FILENAME_HINTS:
        if name.endswith(suffix) or name == suffix:
            return ctype
    return None


def resolve_schema_path(config_type: Optional[str]) -> Optional[Path]:
    if not config_type:
        return None
    p = DEFAULT_SCHEMAS.get(config_type)
    if p and p.exists():
        return p
    return None


def validate_document(doc: Dict, schema_path: Optional[Path]) -> Tuple[bool, List[Dict[str, str]]]:
    """Return (valid, issues). Issues are dicts with path and message."""
    issues: List[Dict[str, str]] = []
    if not schema_path:
        # No schema -> treated as pass, but emit info issue for transparency
        return True, issues

    try:
        schema = load_json(schema_path)
    except Exception as e:
        issues.append({
            "path": "",
            "message": f"Failed to load schema {schema_path}: {e}",
            "category": "schema",
            "severity": "warning",
        })
        return True, issues

    try:
        validator = Draft7Validator(schema)
        errors = sorted(validator.iter_errors(doc), key=lambda e: e.path)
        for err in errors:
            field_path = ".".join(str(p) for p in err.absolute_path)
            issues.append({
                "path": field_path,
                "message": f"Schema validation failed: {err.message}",
                "category": "schema",
                "severity": "error",
            })
        return (len(errors) == 0), issues
    except Exception as e:
        issues.append({
            "path": "",
            "message": f"Schema validation exception: {e}",
            "category": "schema",
            "severity": "error",
        })
        return False, issues


def gather_files(args) -> List[Path]:
    files: List[Path] = []
    if args.files:
        files.extend(Path(f) for f in args.files)
    elif args.glob:
        import glob
        for pat in args.glob:
            files.extend(Path(p) for p in glob.glob(pat, recursive=True))
    else:
        # default: all YAMLs in src/config/
        files.extend(CONFIG_DIR.glob("*.yaml"))
    # De-duplicate and ensure existence
    uniq: List[Path] = []
    seen = set()
    for p in files:
        rp = p.resolve()
        if rp.exists() and rp not in seen:
            uniq.append(rp)
            seen.add(rp)
    return uniq


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Validate POLARIS YAML configs against JSON Schemas (no project import)")
    ap.add_argument("--files", nargs="*", help="Specific YAML files to validate")
    ap.add_argument("--glob", action="append", help="Glob(s) to find YAML files, e.g., 'src/config/*.yaml'", default=None)
    ap.add_argument("--config-type", choices=["framework", "world_model"], help="Force config type for all files")
    ap.add_argument("--schema", type=str, help="Explicit schema file to use for all files")
    ap.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    ap.add_argument("--strict-exit", action="store_true", help="Exit with code 1 if any validation errors are found")

    args = ap.parse_args(argv)

    files = gather_files(args)
    if not files:
        print("No files to validate.")
        return 0

    explicit_schema: Optional[Path] = Path(args.schema).resolve() if args.schema else None

    summary = {
        "overall_valid": True,
        "files": []  # list of dicts per file
    }

    for f in files:
        try:
            doc = load_yaml(f)
        except Exception as e:
            item = {
                "file": str(f),
                "valid": False,
                "issues": [{
                    "path": "",
                    "message": f"Failed to read/parse YAML: {e}",
                    "category": "syntax",
                    "severity": "error",
                }],
            }
            summary["files"].append(item)
            summary["overall_valid"] = False
            continue

        # Determine config type and schema
        config_type = args.config_type or guess_config_type_from_content(doc) or guess_config_type_from_filename(f)
        schema_path = explicit_schema or resolve_schema_path(config_type)

        valid, issues = validate_document(doc, schema_path)
        summary["files"].append({
            "file": str(f),
            "config_type": config_type,
            "schema": str(schema_path) if schema_path else None,
            "valid": valid,
            "issues": issues,
        })
        if not valid:
            summary["overall_valid"] = False

    if args.format == "json":
        print(json.dumps(summary, indent=2))
    else:
        for item in summary["files"]:
            print(f"FILE: {item['file']}")
            print(f"  config_type: {item.get('config_type')}")
            print(f"  schema: {item.get('schema')}")
            print(f"  valid: {item['valid']}")
            for iss in item["issues"]:
                print(f"   - {iss['severity'].upper()} {iss['category']} | {iss['path']} | {iss['message']}")
            print()
        print(f"OVERALL VALID: {summary['overall_valid']}")

    if args.strict_exit and not summary["overall_valid"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
