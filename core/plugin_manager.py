from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

from .legacy_discovery import discover_legacy_plugins, legacy_script_statuses
from .models import AnalysisResult, UploadedData

LOGGER = logging.getLogger("ast_sensor_analytics")


class PluginProtocol(Protocol):
    id: str
    name: str
    description: str

    def confidence(self, files: list[UploadedData]) -> float: ...
    def run(self, files: list[UploadedData], options: dict[str, Any]) -> AnalysisResult: ...


@dataclass
class PluginInfo:
    plugin: PluginProtocol
    confidence: float


@lru_cache(maxsize=1)
def _discover_builtin_plugins_cached() -> tuple[PluginProtocol, ...]:
    """Import hand-written plugins only once per Python process."""
    import plugins

    found: list[PluginProtocol] = []
    for module_info in pkgutil.iter_modules(plugins.__path__):
        if module_info.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"plugins.{module_info.name}")
            plugin = getattr(module, "PLUGIN", None)
            if plugin is not None:
                found.append(plugin)
        except Exception as exc:
            LOGGER.exception("Could not load plugin %s: %s", module_info.name, exc)
    return tuple(sorted(found, key=lambda p: p.name.lower()))


def discover_plugins() -> list[PluginProtocol]:
    """Return built-in plugins plus automatically discovered legacy analyses.

    The legacy folder is signature-scanned on each Streamlit rerun. Parsing and
    introspection are cached by filename/mtime/size, so adding or replacing a
    legacy/*.py file becomes visible after the next browser rerun/refresh without
    changing app.py or creating a new hand-written plugin.
    """
    plugins: list[PluginProtocol] = list(_discover_builtin_plugins_cached())
    try:
        plugins.extend(discover_legacy_plugins())
    except Exception as exc:
        LOGGER.exception("Legacy analysis discovery failed: %s", exc)
    return sorted(plugins, key=lambda p: p.name.lower())


def ranked_plugins(files: list[UploadedData]) -> list[PluginInfo]:
    ranked: list[PluginInfo] = []
    for plugin in discover_plugins():
        try:
            score = float(plugin.confidence(files))
        except Exception as exc:
            LOGGER.exception("Plugin confidence check failed for %s: %s", plugin.id, exc)
            score = 0.0
        ranked.append(PluginInfo(plugin=plugin, confidence=max(0.0, min(1.0, score))))
    return sorted(ranked, key=lambda p: p.confidence, reverse=True)


__all__ = [
    "PluginInfo",
    "PluginProtocol",
    "discover_plugins",
    "ranked_plugins",
    "legacy_script_statuses",
]
