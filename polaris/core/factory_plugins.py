"""Connector plugin loading helpers for factory registration."""

import importlib
import importlib.metadata
import inspect
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from polaris.abstractions import Connector, Logger, MetricsCollector

ConnectorFactory = Callable[[Any, "Logger", Optional["MetricsCollector"]], "Connector"]
ConnectorConfigValidator = Callable[[Dict[str, Any]], None]
RegisterConnectorFactory = Callable[[str, ConnectorFactory], None]
RegisterConnectorConfigValidator = Callable[[str, ConnectorConfigValidator], None]
RegisterToolFactory = Callable[..., Any]


class ConnectorPluginLoader:
    """Load connector plugins from entry points and explicit module imports."""

    def __init__(
        self,
        *,
        register_connector_factory: RegisterConnectorFactory,
        register_connector_config_validator: RegisterConnectorConfigValidator,
        register_tool_factory: Optional[RegisterToolFactory],
        entry_point_group: str,
        logger: Any,
    ) -> None:
        """Initialize loader with registration callbacks and plugin discovery settings."""
        self._register_connector_factory = register_connector_factory
        self._register_connector_config_validator = register_connector_config_validator
        self._register_tool_factory = register_tool_factory
        self._entry_point_group = entry_point_group
        self._logger = logger

        self._entry_points_discovered = False
        self._loaded_plugin_modules: set[str] = set()
        self._loaded_entry_points: set[str] = set()

    def _invoke_plugin_registration(self, registration_hook: Callable[..., Any]) -> None:
        """Invoke plugin registration hook with strict canonical signature."""
        signature = inspect.signature(registration_hook)
        names = set(signature.parameters.keys())
        required = {"register_connector_factory", "register_connector_config_validator"}
        if not required.issubset(names):
            raise TypeError(
                "register_polaris_plugins must accept keyword parameters "
                "'register_connector_factory' and 'register_connector_config_validator'"
            )

        kwargs: Dict[str, Any] = {
            "register_connector_factory": self._register_connector_factory,
            "register_connector_config_validator": self._register_connector_config_validator,
        }
        if self._register_tool_factory and "register_tool_factory" in names:
            kwargs["register_tool_factory"] = self._register_tool_factory

        registration_hook(**kwargs)

    def _activate_plugin(self, plugin: Any) -> None:
        """Activate a loaded plugin object if it exposes registration hooks."""
        registration_hook = getattr(plugin, "register_polaris_plugins", None)
        if callable(registration_hook):
            self._invoke_plugin_registration(registration_hook)
            return

        if isinstance(plugin, ModuleType):
            # Module may have already self-registered via import side effects.
            return

        if callable(plugin):
            self._invoke_plugin_registration(plugin)

    def discover_entry_points(self) -> None:
        """Load connector plugins exposed via entry points once per process."""
        if self._entry_points_discovered:
            return

        self._entry_points_discovered = True

        try:
            entry_points = importlib.metadata.entry_points(group=self._entry_point_group)
        except Exception as exc:
            self._logger.warning(
                "Failed to enumerate connector entry points",
                extra={"error": str(exc), "group": self._entry_point_group},
            )
            return

        for entry_point in entry_points:
            entry_point_id = f"{entry_point.group}:{entry_point.name}"
            if entry_point_id in self._loaded_entry_points:
                continue

            try:
                plugin = entry_point.load()
                self._activate_plugin(plugin)
                self._loaded_entry_points.add(entry_point_id)
            except Exception as exc:
                # Keep startup resilient when optional third-party plugins fail to load.
                self._logger.warning(
                    "Skipping connector plugin entry point after load failure",
                    extra={
                        "entry_point": entry_point_id,
                        "module": getattr(entry_point, "module", None),
                        "error": str(exc),
                    },
                )
                continue

    def discover_explicit_plugins(self, plugin_imports: Optional[List[str]] = None) -> List[str]:
        """Load connector plugins from explicit module import paths."""
        loaded_now: List[str] = []
        for module_path in plugin_imports or []:
            normalized_path = module_path.strip()
            if not normalized_path or normalized_path in self._loaded_plugin_modules:
                continue

            plugin_module = importlib.import_module(normalized_path)
            self._activate_plugin(plugin_module)
            self._loaded_plugin_modules.add(normalized_path)
            loaded_now.append(normalized_path)

        return loaded_now
