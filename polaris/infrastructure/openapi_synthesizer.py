"""OpenAPI / Swagger Action Synthesizer for POLARIS.

Provides automated zero-touch ingestion of OpenAPI v3.x and Swagger 2.0 specifications
to dynamically synthesize ActionSchema definitions, REST endpoint bindings, and SystemContracts.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml

from polaris.abstractions.system_contract import ActionSchema, SystemContract


@dataclass(frozen=True)
class HttpActionEndpoint:
    """HTTP binding metadata for executing a synthesized adaptation action."""

    action_type: str
    path: str
    method: str = "POST"
    path_parameters: Tuple[str, ...] = ()
    query_parameters: Tuple[str, ...] = ()
    body_parameters: Tuple[str, ...] = ()
    content_type: str = "application/json"


@dataclass
class SynthesizedApi:
    """Synthesized contract, action schemas, and endpoint bindings for an HTTP system."""

    system_id: str
    title: str = ""
    version: str = ""
    base_url: Optional[str] = None
    action_schemas: Dict[str, ActionSchema] = field(default_factory=dict)
    action_endpoints: Dict[str, HttpActionEndpoint] = field(default_factory=dict)

    def to_system_contract(self) -> SystemContract:
        """Convert synthesized action schemas into a formal SystemContract."""
        return SystemContract(
            system_id=self.system_id,
            supported_action_types=tuple(self.action_schemas.keys()),
            actions=self.action_schemas,
            metadata={
                "title": self.title,
                "version": self.version,
                "base_url": self.base_url,
                "synthesized_from": "openapi",
            },
        )


class OpenApiSynthesizer:
    """Synthesizer transforming OpenAPI v3.x and Swagger 2.0 specifications into POLARIS contracts."""

    MUTATION_METHODS: Set[str] = {"POST", "PUT", "PATCH", "DELETE"}
    ALL_METHODS: Set[str] = {"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"}

    @classmethod
    def _normalize_identifier(cls, name: str) -> str:
        """Convert camelCase, PascalCase, or kebab-case into snake_case identifier."""
        # Insert underscore before capital letters preceded by lowercase
        s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
        # Replace non-alphanumeric characters with underscore
        s2 = re.sub(r"[^a-zA-Z0-9]+", "_", s1)
        cleaned = s2.strip("_").lower()
        return cleaned or "action"

    @classmethod
    def _generate_action_name(cls, method: str, path: str, operation_id: Optional[str]) -> str:
        """Derive a canonical action type identifier."""
        if operation_id:
            return cls._normalize_identifier(operation_id)

        clean_path = path.strip("/")
        # Replace {param} with by_param
        clean_path = re.sub(r"\{([^}]+)\}", r"by_\1", clean_path)
        clean_path = re.sub(r"[^a-zA-Z0-9]+", "_", clean_path).strip("_").lower()

        if not clean_path:
            return method.lower()
        return f"{method.lower()}_{clean_path}"

    @classmethod
    def _resolve_ref(
        cls, spec: Dict[str, Any], ref: str, visited: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """Resolve internal $ref references (e.g. #/components/schemas/Foo)."""
        if not ref.startswith("#/"):
            return {}

        visited = visited or set()
        if ref in visited:
            return {}
        visited.add(ref)

        tokens = ref.lstrip("#/").split("/")
        curr: Any = spec
        for token in tokens:
            if isinstance(curr, dict) and token in curr:
                curr = curr[token]
            else:
                return {}

        if isinstance(curr, dict) and "$ref" in curr:
            return cls._resolve_ref(spec, curr["$ref"], visited)

        return curr if isinstance(curr, dict) else {}

    @classmethod
    def _extract_property_schema(
        cls, prop_spec: Dict[str, Any], root_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract a simplified JSON Schema dictionary suitable for ActionSchema."""
        if "$ref" in prop_spec:
            prop_spec = cls._resolve_ref(root_spec, prop_spec["$ref"])

        param_type = prop_spec.get("type", "string")
        # OpenAPI integer types
        if param_type in ("integer", "int32", "int64"):
            canonical_type = "integer"
        elif param_type in ("number", "float", "double"):
            canonical_type = "number"
        elif param_type == "boolean":
            canonical_type = "boolean"
        elif param_type == "array":
            canonical_type = "array"
        elif param_type == "object":
            canonical_type = "object"
        else:
            canonical_type = "string"

        schema_dict: Dict[str, Any] = {"type": canonical_type}

        if "description" in prop_spec:
            schema_dict["description"] = prop_spec["description"]
        if "minimum" in prop_spec:
            schema_dict["minimum"] = prop_spec["minimum"]
        if "maximum" in prop_spec:
            schema_dict["maximum"] = prop_spec["maximum"]
        if "enum" in prop_spec and isinstance(prop_spec["enum"], (list, tuple)):
            schema_dict["enum"] = list(prop_spec["enum"])
        if "default" in prop_spec:
            schema_dict["default"] = prop_spec["default"]

        return schema_dict

    @classmethod
    def _infer_impacts(cls, action_name: str, op_data: Dict[str, Any]) -> Tuple[str, str, str]:
        """Infer performance, cost, and QoS impacts from metadata or naming heuristics."""
        # Explicit OpenAPI extensions take precedence
        perf = op_data.get("x-polaris-performance-impact")
        cost = op_data.get("x-polaris-cost-impact")
        qos = op_data.get("x-polaris-qos-impact")

        if perf and cost and qos:
            return str(perf), str(cost), str(qos)

        lower_name = action_name.lower()
        if any(w in lower_name for w in ("scale_up", "scale_out", "increase", "add_worker")):
            perf = perf or "positive"
            cost = cost or "negative"
            qos = qos or "positive"
        elif any(w in lower_name for w in ("scale_down", "scale_in", "decrease", "remove_worker")):
            perf = perf or "negative"
            cost = cost or "positive"
            qos = qos or "neutral"
        elif "throttle" in lower_name and "unthrottle" not in lower_name:
            perf = perf or "negative"
            cost = cost or "positive"
            qos = qos or "negative"
        elif "unthrottle" in lower_name:
            perf = perf or "positive"
            cost = cost or "neutral"
            qos = qos or "positive"
        else:
            perf = perf or "neutral"
            cost = cost or "neutral"
            qos = qos or "neutral"

        return str(perf), str(cost), str(qos)

    @classmethod
    def synthesize_from_dict(
        cls,
        spec: Dict[str, Any],
        system_id: str = "http_service",
        base_url: Optional[str] = None,
        mutation_only: bool = True,
        operation_filter: Optional[List[str]] = None,
    ) -> SynthesizedApi:
        """Synthesize POLARIS action schemas and endpoints from an OpenAPI dictionary.

        Args:
            spec: Parsed OpenAPI / Swagger specification dictionary
            system_id: Target system identifier
            base_url: Optional base URL override
            mutation_only: If True, only include mutation methods (POST, PUT, PATCH, DELETE)
            operation_filter: Optional list of specific operationIds or action types to include
        """
        info = spec.get("info", {})
        title = info.get("title", "")
        version = info.get("version", "")

        # Extract base_url if not provided
        derived_base_url = base_url
        if not derived_base_url:
            # OpenAPI 3 servers
            servers = spec.get("servers", [])
            if servers and isinstance(servers, list) and isinstance(servers[0], dict):
                derived_base_url = servers[0].get("url")
            # Swagger 2 host + basePath
            elif "host" in spec:
                schemes = spec.get("schemes", ["http"])
                scheme = schemes[0] if schemes else "http"
                base_path = spec.get("basePath", "").rstrip("/")
                derived_base_url = f"{scheme}://{spec['host']}{base_path}"

        allowed_methods = cls.MUTATION_METHODS if mutation_only else cls.ALL_METHODS
        paths = spec.get("paths", {})

        action_schemas: Dict[str, ActionSchema] = {}
        action_endpoints: Dict[str, HttpActionEndpoint] = {}

        for path, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue

            path_params = path_item.get("parameters", [])

            for method_key, op_data in path_item.items():
                method_upper = method_key.upper()
                if method_upper not in allowed_methods:
                    continue
                if not isinstance(op_data, dict):
                    continue

                operation_id = op_data.get("operationId")
                action_name = cls._generate_action_name(method_upper, path, operation_id)

                if operation_filter and action_name not in operation_filter:
                    continue

                summary = op_data.get("summary") or op_data.get("description") or ""
                description = (
                    summary or f"Auto-synthesized adaptation action for {method_upper} {path}"
                )

                param_schemas: Dict[str, Any] = {}
                required_params: List[str] = []
                path_param_names: List[str] = []
                query_param_names: List[str] = []
                body_param_names: List[str] = []

                # Combine path-level and operation-level parameters
                all_params = list(path_params) + list(op_data.get("parameters", []))
                for param in all_params:
                    if "$ref" in param:
                        param = cls._resolve_ref(spec, param["$ref"])
                    if not isinstance(param, dict):
                        continue

                    p_name = param.get("name")
                    p_in = param.get("in")
                    if not p_name or not p_in:
                        continue

                    # Swagger 2 body parameter
                    if p_in == "body":
                        b_schema = param.get("schema", {})
                        if "$ref" in b_schema:
                            b_schema = cls._resolve_ref(spec, b_schema["$ref"])
                        props = b_schema.get("properties", {})
                        b_reqs = set(b_schema.get("required", []))
                        for prop_name, prop_spec in props.items():
                            param_schemas[prop_name] = cls._extract_property_schema(prop_spec, spec)
                            body_param_names.append(prop_name)
                            if prop_name in b_reqs:
                                required_params.append(prop_name)
                        continue

                    p_schema = param.get("schema", param)
                    param_schemas[p_name] = cls._extract_property_schema(p_schema, spec)

                    if p_in == "path":
                        path_param_names.append(p_name)
                        required_params.append(p_name)
                    elif p_in == "query":
                        query_param_names.append(p_name)
                        if param.get("required"):
                            required_params.append(p_name)

                # OpenAPI 3 requestBody
                req_body = op_data.get("requestBody")
                if isinstance(req_body, dict):
                    if "$ref" in req_body:
                        req_body = cls._resolve_ref(spec, req_body["$ref"])
                    content = req_body.get("content", {})
                    # Prefer application/json, otherwise pick first available
                    media_schema = None
                    if "application/json" in content:
                        media_schema = content["application/json"].get("schema", {})
                    elif content:
                        first_content = next(iter(content.values()))
                        if isinstance(first_content, dict):
                            media_schema = first_content.get("schema", {})

                    if isinstance(media_schema, dict):
                        if "$ref" in media_schema:
                            media_schema = cls._resolve_ref(spec, media_schema["$ref"])

                        props = media_schema.get("properties", {})
                        b_reqs = set(media_schema.get("required", []))
                        for prop_name, prop_spec in props.items():
                            param_schemas[prop_name] = cls._extract_property_schema(prop_spec, spec)
                            body_param_names.append(prop_name)
                            if prop_name in b_reqs:
                                required_params.append(prop_name)

                perf_impact, cost_impact, qos_impact = cls._infer_impacts(action_name, op_data)

                # Check for extensions
                rollback = op_data.get("x-polaris-rollback-action")
                exp_duration = float(op_data.get("x-polaris-expected-duration-seconds", 0.0))
                verif_window = float(op_data.get("x-polaris-verification-window-seconds", 0.0))

                action_schema = ActionSchema(
                    action_type=action_name,
                    description=description,
                    parameters_schema=param_schemas,
                    required_parameters=tuple(dict.fromkeys(required_params)),
                    rollback_action=str(rollback) if rollback else None,
                    expected_duration_seconds=exp_duration,
                    default_verification_window_seconds=verif_window,
                    performance_impact=perf_impact,
                    cost_impact=cost_impact,
                    qos_impact=qos_impact,
                    metadata={
                        "path": path,
                        "method": method_upper,
                        "tags": op_data.get("tags", []),
                        "operation_id": operation_id,
                    },
                )

                endpoint = HttpActionEndpoint(
                    action_type=action_name,
                    path=path,
                    method=method_upper,
                    path_parameters=tuple(path_param_names),
                    query_parameters=tuple(query_param_names),
                    body_parameters=tuple(body_param_names),
                )

                action_schemas[action_name] = action_schema
                action_endpoints[action_name] = endpoint

        return SynthesizedApi(
            system_id=system_id,
            title=title,
            version=version,
            base_url=derived_base_url,
            action_schemas=action_schemas,
            action_endpoints=action_endpoints,
        )

    @classmethod
    def synthesize_from_json(
        cls,
        json_str: str,
        system_id: str = "http_service",
        base_url: Optional[str] = None,
        mutation_only: bool = True,
    ) -> SynthesizedApi:
        """Synthesize from a raw JSON OpenAPI string."""
        spec = json.loads(json_str)
        return cls.synthesize_from_dict(
            spec, system_id=system_id, base_url=base_url, mutation_only=mutation_only
        )

    @classmethod
    def synthesize_from_yaml(
        cls,
        yaml_str: str,
        system_id: str = "http_service",
        base_url: Optional[str] = None,
        mutation_only: bool = True,
    ) -> SynthesizedApi:
        """Synthesize from a YAML OpenAPI string."""
        spec = yaml.safe_load(yaml_str) or {}
        return cls.synthesize_from_dict(
            spec, system_id=system_id, base_url=base_url, mutation_only=mutation_only
        )

    @classmethod
    def synthesize_from_file(
        cls,
        file_path: Union[str, Path],
        system_id: str = "http_service",
        base_url: Optional[str] = None,
        mutation_only: bool = True,
    ) -> SynthesizedApi:
        """Synthesize from an OpenAPI JSON or YAML file on disk."""
        path_obj = Path(file_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"OpenAPI file not found: {file_path}")

        content = path_obj.read_text(encoding="utf-8")
        if path_obj.suffix.lower() in (".yaml", ".yml"):
            return cls.synthesize_from_yaml(
                content, system_id=system_id, base_url=base_url, mutation_only=mutation_only
            )
        return cls.synthesize_from_json(
            content, system_id=system_id, base_url=base_url, mutation_only=mutation_only
        )

    @classmethod
    async def synthesize_from_url(
        cls,
        url: str,
        system_id: str = "http_service",
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 10.0,
        mutation_only: bool = True,
    ) -> SynthesizedApi:
        """Fetch and synthesize directly from an OpenAPI live URL endpoint."""
        import httpx

        async with httpx.AsyncClient(headers=headers or {}, timeout=timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            text = resp.text

        try:
            return cls.synthesize_from_json(text, system_id=system_id, mutation_only=mutation_only)
        except json.JSONDecodeError:
            return cls.synthesize_from_yaml(text, system_id=system_id, mutation_only=mutation_only)
