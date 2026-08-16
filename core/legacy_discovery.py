from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import inspect
import io
import json
import logging
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from .models import AnalysisResult, UploadedData
from .tempfiles import materialized_uploads

LOGGER = logging.getLogger("ast_sensor_analytics")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = PROJECT_ROOT / "legacy"

# These scripts already have hand-written adapters in plugins/. Discovering them a
# second time would create duplicate Analysis entries. They remain in legacy/ and
# are still called by their dedicated plugins.
DEDICATED_LEGACY_SCRIPTS = {
    "analisar_relampagos.py",
    "analise_radiacao.py",
    "plot_medicoes_uv_todos_sensores.py",
}

ENTRYPOINT_NAMES = (
    "run_analysis",
    "analyze_file",
    "analyse_file",
    "analisar_ficheiro",
    "analisar_arquivo",
    "analyze",
    "analyse",
    "analisar",
    "process_file",
    "processar_ficheiro",
    "processar_arquivo",
    "processar",
    "process",
    "run",
)

# Filename/source matching is intentionally deterministic. No LLM or external
# service is involved in deciding which legacy analysis belongs to which sensor.
SENSOR_ALIASES: dict[str, tuple[str, ...]] = {
    "ch4": ("ch4", "methane", "metano"),
    "alcohol": ("alcohol", "ethanol", "etanol"),
    "gas": ("gas", "gases", "mq2", "mq_2", "mq4", "mq_4", "mq135", "mq_135"),
    "uv": ("uv", "uva", "uvb", "uvc", "ultraviolet", "indice_uv", "index_uv", "uvi"),
    "radiation": ("radiation", "radiacao", "geiger", "cpm", "dose"),
    "lightning": ("lightning", "relampago", "trovoada", "as3935"),
    "magnetometer": ("magnetometer", "magnetometro", "magnetic", "geomagnetic", "mag_x", "mag_y", "mag_z"),
    "accelerometer": ("accelerometer", "acelerometro", "acceleracao", "accel_x", "accel_y", "accel_z"),
    "gyroscope": ("gyroscope", "giroscopio", "gyro_x", "gyro_y", "gyro_z"),
    "imu": ("imu", "inertial", "sensor_fusion"),
    "weather": ("weather", "meteorologia", "meteorological", "meteo", "temperature", "temperatura", "humidity", "humidade", "pressure", "pressao"),
    "particles": ("particles", "particle", "particulas", "sps", "pm25", "pm2_5", "pm10"),
    "volatiles": ("volatiles", "volatile", "voc", "vocs"),
    "gnss": ("gnss", "gps", "rtk", "ecef", "enu", "satellite", "satelite"),
}

FAMILY_COMPATIBILITY: dict[str, set[str]] = {
    "gas": {"gas", "ch4", "alcohol"},
    "ch4": {"ch4", "gas"},
    "alcohol": {"alcohol", "gas"},
    "imu": {"imu", "accelerometer", "gyroscope", "magnetometer"},
    "accelerometer": {"accelerometer", "imu"},
    "gyroscope": {"gyroscope", "imu"},
    "magnetometer": {"magnetometer", "imu"},
    "weather": {"weather"},
    "gnss": {"gnss"},
    "uv": {"uv"},
    "radiation": {"radiation"},
    "lightning": {"lightning"},
    "particles": {"particles"},
    "volatiles": {"volatiles"},
}

PATH_PARAM_NAMES = {
    "file", "filename", "filepath", "file_path", "path", "ficheiro", "arquivo", "caminho", "input_file", "input_path",
}
FILES_PARAM_NAMES = {
    "files", "filenames", "filepaths", "paths", "ficheiros", "arquivos", "input_files", "input_paths",
}
DIR_PARAM_NAMES = {"folder", "directory", "dir", "root", "pasta", "diretorio", "input_dir", "input_folder"}
DF_PARAM_NAMES = {"df", "dataframe", "frame", "data", "dados", "table", "tabela"}
DFS_PARAM_NAMES = {"dfs", "dataframes", "frames", "tables", "tabelas"}
OPTIONS_PARAM_NAMES = {"options", "option", "config", "configuration", "settings", "params", "parameters", "opcoes", "configuracao"}


def _ascii(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace(".", "_")
    return re.sub(r"[^a-z0-9_]+", "_", text).strip("_")


def _contains_alias(text: str, alias: str) -> bool:
    alias = _ascii(alias)
    if not alias:
        return False
    # Short aliases such as UV are bounded so they do not match arbitrary words.
    if len(alias) <= 3:
        return bool(re.search(rf"(?:^|_){re.escape(alias)}(?:_|$)", text))
    return alias in text


def _sensor_from_text(*values: Any) -> str | None:
    text = "_".join(_ascii(value) for value in values if value is not None)
    matches: list[tuple[int, str]] = []
    for sensor, aliases in SENSOR_ALIASES.items():
        for alias in aliases:
            if _contains_alias(text, alias):
                # Prefer more specific/longer aliases when several families match.
                matches.append((len(_ascii(alias)), sensor))
    if not matches:
        return None
    return max(matches, key=lambda item: item[0])[1]


def _friendly_name(stem: str, sensor: str | None) -> str:
    cleaned = re.sub(r"[_\-]+", " ", stem).strip()
    cleaned = re.sub(r"\b(analise|analysis|analisar|analyze|analyse|processar|process)\b", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    sensor_label = {
        "ch4": "CH4", "uv": "UV", "gnss": "GNSS", "imu": "IMU",
        "radiation": "Radiation", "lightning": "Lightning", "weather": "Weather",
        "particles": "Particles", "volatiles": "Volatiles", "alcohol": "Alcohol",
        "gas": "Gas", "magnetometer": "Magnetometer", "accelerometer": "Accelerometer",
        "gyroscope": "Gyroscope",
    }.get(sensor or "", (sensor or "").title())

    def title_words(text: str) -> str:
        words = []
        acronyms = {"ch4": "CH4", "uv": "UV", "gnss": "GNSS", "imu": "IMU", "gps": "GPS", "rtk": "RTK"}
        for word in text.split():
            words.append(acronyms.get(_ascii(word), word.title()))
        return " ".join(words)

    if sensor:
        if cleaned and _ascii(sensor_label) not in _ascii(cleaned):
            return f"{sensor_label} · {title_words(cleaned)}"
        return title_words(cleaned) if cleaned else f"{sensor_label} · Legacy analysis"
    return title_words(cleaned) or stem


def _manifest_for(script: Path) -> dict[str, Any]:
    sidecar = script.with_suffix(".json")
    if not sidecar.exists():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        LOGGER.warning("Could not read legacy manifest %s: %s", sidecar.name, exc)
        return {"_manifest_error": str(exc)}


def _function_specs(source: str) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [{"_syntax_error": f"line {exc.lineno}: {exc.msg}"}]

    specs: list[dict[str, Any]] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        positional = list(node.args.posonlyargs) + list(node.args.args)
        defaults_start = len(positional) - len(node.args.defaults)
        params: list[dict[str, Any]] = []
        for idx, arg in enumerate(positional):
            params.append({
                "name": arg.arg,
                "required": idx < defaults_start,
            })
        for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
            params.append({"name": arg.arg, "required": default is None})
        specs.append({
            "name": node.name,
            "params": params,
            "vararg": node.args.vararg is not None,
            "kwarg": node.args.kwarg is not None,
            "async": isinstance(node, ast.AsyncFunctionDef),
        })
    return specs


def _param_input_kind(name: str) -> str | None:
    key = _ascii(name)
    if key in PATH_PARAM_NAMES:
        return "file"
    if key in FILES_PARAM_NAMES:
        return "files"
    if key in DIR_PARAM_NAMES:
        return "directory"
    if key in DFS_PARAM_NAMES:
        return "dataframes"
    if key in DF_PARAM_NAMES:
        return "dataframe"
    if key in OPTIONS_PARAM_NAMES:
        return "options"
    return None


def _select_entrypoint(specs: list[dict[str, Any]], manifest: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    if specs and "_syntax_error" in specs[0]:
        return None, None, specs[0]["_syntax_error"]

    by_name = {spec["name"]: spec for spec in specs}
    requested = str(manifest.get("function") or manifest.get("entrypoint") or "").strip()
    if requested:
        spec = by_name.get(requested)
        if spec is None:
            return None, None, f"Manifest entry point '{requested}' was not found."
        candidates = [spec]
    else:
        candidates = []
        lower_lookup = {_ascii(name): spec for name, spec in by_name.items()}
        for preferred in ENTRYPOINT_NAMES:
            spec = lower_lookup.get(_ascii(preferred))
            if spec is not None and spec not in candidates:
                candidates.append(spec)

    if not candidates:
        return None, None, "No supported analysis function was found. Add a function such as analyze_file(path), analisar_ficheiro(path), analyze(df), or use a JSON manifest."

    manifest_input = _ascii(manifest.get("input") or "") or None
    valid_inputs = {"file", "files", "directory", "dataframe", "dataframes", "none"}
    if manifest_input and manifest_input not in valid_inputs:
        return None, None, f"Unsupported manifest input '{manifest_input}'."

    for spec in candidates:
        params = spec.get("params", [])
        unknown_required = []
        inferred_first: str | None = None
        for idx, param in enumerate(params):
            kind = _param_input_kind(param["name"])
            if idx == 0 and manifest_input:
                kind = manifest_input
            if idx == 0 and kind is not None and kind != "options":
                inferred_first = kind
            if param["required"] and kind is None:
                unknown_required.append(param["name"])
        if not params:
            inferred_first = "none"
        if not unknown_required:
            return spec["name"], inferred_first or manifest_input or "none", None

    return None, None, "The candidate function has required parameters whose meaning cannot be inferred automatically. Add a JSON manifest or use conventional parameter names."


@dataclass(frozen=True)
class LegacyScriptStatus:
    filename: str
    name: str
    sensor: str | None
    entrypoint: str | None
    input_kind: str | None
    supported: bool
    reason: str = ""
    manifest_used: bool = False


@dataclass(frozen=True)
class LegacyScriptDefinition:
    path: Path
    name: str
    sensor: str | None
    entrypoint: str
    input_kind: str
    description: str
    manifest: dict[str, Any]


def _scan_script(script: Path) -> tuple[LegacyScriptDefinition | None, LegacyScriptStatus | None]:
    if script.name in DEDICATED_LEGACY_SCRIPTS or script.name == "__init__.py" or script.name.startswith("_"):
        return None, None
    try:
        source = script.read_text(encoding="utf-8-sig", errors="replace")
    except Exception as exc:
        status = LegacyScriptStatus(script.name, script.stem, None, None, None, False, f"Could not read script: {exc}")
        return None, status

    manifest = _manifest_for(script)
    manifest_error = manifest.get("_manifest_error")
    sensor = _ascii(manifest.get("sensor") or "") or _sensor_from_text(script.stem, source[:4000])
    if sensor in {"any", "all", "generic"}:
        sensor = "generic"
    elif sensor and sensor not in SENSOR_ALIASES:
        # Allow common aliases in the manifest, e.g. methane instead of ch4.
        sensor = _sensor_from_text(sensor) or sensor

    specs = _function_specs(source)
    entrypoint, input_kind, reason = _select_entrypoint(specs, manifest)
    name = str(manifest.get("name") or "").strip() or _friendly_name(script.stem, sensor)
    description = str(manifest.get("description") or "").strip()
    if not description and entrypoint:
        description = f"Legacy analysis from {script.name} · {entrypoint}({input_kind})."

    if manifest_error:
        reason = f"Invalid JSON manifest: {manifest_error}"
    if not sensor:
        reason = reason or "Sensor family could not be inferred from the filename/source. Add a JSON manifest with a 'sensor' field."
    if not entrypoint:
        status = LegacyScriptStatus(script.name, name, sensor, None, input_kind, False, reason or "No compatible entry point.", bool(manifest))
        return None, status

    definition = LegacyScriptDefinition(
        path=script,
        name=name,
        sensor=sensor,
        entrypoint=entrypoint,
        input_kind=input_kind or "none",
        description=description,
        manifest=manifest,
    )
    status = LegacyScriptStatus(script.name, name, sensor, entrypoint, input_kind, True, "", bool(manifest))
    return definition, status


def _legacy_signature(root: Path = LEGACY_ROOT) -> tuple[tuple[str, int, int], ...]:
    if not root.exists():
        return ()
    entries: list[tuple[str, int, int]] = []
    for path in root.glob("*.py"):
        if path.name.startswith("_"):
            continue
        try:
            stat = path.stat()
            entries.append((path.name, int(stat.st_mtime_ns), int(stat.st_size)))
            sidecar = path.with_suffix(".json")
            if sidecar.exists():
                side_stat = sidecar.stat()
                entries.append((sidecar.name, int(side_stat.st_mtime_ns), int(side_stat.st_size)))
        except OSError:
            continue
    return tuple(sorted(entries))


@lru_cache(maxsize=16)
def _scan_cached(root_text: str, signature: tuple[tuple[str, int, int], ...]) -> tuple[tuple[LegacyScriptDefinition, ...], tuple[LegacyScriptStatus, ...]]:
    del signature
    root = Path(root_text)
    definitions: list[LegacyScriptDefinition] = []
    statuses: list[LegacyScriptStatus] = []
    for script in sorted(root.glob("*.py"), key=lambda path: path.name.lower()):
        definition, status = _scan_script(script)
        if definition is not None:
            definitions.append(definition)
        if status is not None:
            statuses.append(status)
    return tuple(definitions), tuple(statuses)


def _dataset_families(files: list[UploadedData]) -> set[str]:
    families: set[str] = set()
    for file in files:
        parts: list[Any] = [file.name, file.kind]
        if file.dataframe is not None:
            parts.extend(map(str, file.dataframe.columns))
        try:
            parts.append(file.raw[:12000].decode("utf-8", errors="ignore"))
        except Exception:
            pass
        text = "_".join(_ascii(part) for part in parts if part is not None)
        for sensor, aliases in SENSOR_ALIASES.items():
            if any(_contains_alias(text, alias) for alias in aliases):
                families.add(sensor)

    # Promote related families so a generic gas/IMU script can match a specific
    # CH4 or accelerometer dataset without weakening exact matches.
    if families.intersection({"ch4", "alcohol"}):
        families.add("gas")
    if families.intersection({"accelerometer", "gyroscope", "magnetometer"}):
        families.add("imu")
    return families


def _module_name(definition: LegacyScriptDefinition) -> str:
    try:
        stat = definition.path.stat()
        stamp = f"{definition.path}:{stat.st_mtime_ns}:{stat.st_size}"
    except OSError:
        stamp = str(definition.path)
    digest = hashlib.sha1(stamp.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"ast_legacy_{_ascii(definition.path.stem)}_{digest}"


def _load_module(definition: LegacyScriptDefinition):
    module_name = _module_name(definition)
    # A changed script gets a changed module name because mtime/size are included.
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, definition.path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {definition.path.name}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    # Older scientific scripts often import helper modules stored beside the
    # analysis file. Make that folder importable only while loading the module.
    legacy_dir = str(definition.path.parent.resolve())
    inserted = legacy_dir not in sys.path
    if inserted:
        sys.path.insert(0, legacy_dir)
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted:
            try:
                sys.path.remove(legacy_dir)
            except ValueError:
                pass
    return module


def _native_dataframes(files: list[UploadedData]) -> list[pd.DataFrame]:
    frames = [file.dataframe.copy() for file in files if isinstance(file.dataframe, pd.DataFrame) and not file.dataframe.empty]
    return frames


def _value_for_param(param_name: str, forced_kind: str | None, *, root: Path, paths: list[Path], files: list[UploadedData], options: dict[str, Any]) -> tuple[bool, Any]:
    kind = forced_kind or _param_input_kind(param_name)
    if kind == "file":
        if not paths:
            return False, None
        return True, str(paths[0])
    if kind == "files":
        return True, [str(path) for path in paths]
    if kind == "directory":
        return True, str(root)
    if kind == "dataframe":
        frames = _native_dataframes(files)
        if not frames:
            raise ValueError("This legacy function expects a DataFrame, but the selected source is not available as a native table. Use a file/path input or a manifest suited to the script.")
        return True, frames[0]
    if kind == "dataframes":
        frames = _native_dataframes(files)
        if not frames:
            raise ValueError("This legacy function expects DataFrames, but the selected source is not available as native tables.")
        return True, frames
    if kind == "options":
        return True, dict(options)
    if kind == "none":
        return False, None
    return False, None


def _invoke_function(function: Any, definition: LegacyScriptDefinition, *, root: Path, paths: list[Path], files: list[UploadedData], options: dict[str, Any]) -> Any:
    signature = inspect.signature(function)
    positional: list[Any] = []
    keyword: dict[str, Any] = {}
    first_parameter = True

    for parameter in signature.parameters.values():
        if parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        forced = definition.input_kind if first_parameter and definition.input_kind != "none" else None
        first_parameter = False
        known, value = _value_for_param(parameter.name, forced, root=root, paths=paths, files=files, options=options)
        if not known:
            if parameter.default is not inspect._empty:
                continue
            raise ValueError(
                f"Cannot infer the required parameter '{parameter.name}' for {definition.entrypoint}(). "
                "Use a conventional parameter name or add a JSON manifest next to the script."
            )
        if parameter.kind == inspect.Parameter.KEYWORD_ONLY:
            keyword[parameter.name] = value
        else:
            positional.append(value)
    return function(*positional, **keyword)


def _figure_png_bytes(fig: Any) -> bytes | None:
    if not hasattr(fig, "savefig"):
        return None
    try:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", dpi=140)
        return buffer.getvalue()
    except Exception:
        return None


def _table_from_value(value: Any) -> pd.DataFrame | None:
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, pd.Series):
        return value.to_frame()
    if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        try:
            return pd.DataFrame(value)
        except Exception:
            return None
    return None


def _normalise_return(definition: LegacyScriptDefinition, returned: Any, stdout: str, stderr: str, captured_figures: list[Any]) -> AnalysisResult:
    if isinstance(returned, AnalysisResult):
        result = returned
    else:
        metrics: dict[str, Any] = {}
        tables: dict[str, pd.DataFrame] = {}
        figures: list[tuple[str, Any]] = []
        notes: list[str] = []
        warnings: list[str] = []
        summary = ""

        def consume(label: str, value: Any) -> None:
            nonlocal summary
            table = _table_from_value(value)
            if table is not None:
                tables[label] = table
                return
            if hasattr(value, "to_plotly_json"):
                figures.append((label, value))
                return
            png = _figure_png_bytes(value)
            if png is not None:
                figures.append((label, png))
                return
            if isinstance(value, dict):
                # Respect the common AnalysisResult-like dictionary shape first.
                if "summary" in value and isinstance(value["summary"], str):
                    summary = value["summary"]
                for key, item in value.items():
                    if key in {"title", "summary"}:
                        continue
                    if key == "metrics" and isinstance(item, dict):
                        metrics.update(item)
                        continue
                    if key == "tables" and isinstance(item, dict):
                        for table_name, table_value in item.items():
                            table = _table_from_value(table_value)
                            if table is not None:
                                tables[str(table_name)] = table
                        continue
                    if key == "warnings" and isinstance(item, (list, tuple)):
                        warnings.extend(str(entry) for entry in item)
                        continue
                    if key == "notes" and isinstance(item, (list, tuple)):
                        notes.extend(str(entry) for entry in item)
                        continue
                    nested_table = _table_from_value(item)
                    if nested_table is not None:
                        tables[str(key)] = nested_table
                    elif hasattr(item, "to_plotly_json"):
                        figures.append((str(key), item))
                    elif isinstance(item, (str, int, float, bool)) or item is None:
                        metrics[str(key)] = item
                return
            if isinstance(value, (tuple, list)):
                for idx, item in enumerate(value, 1):
                    consume(f"{label} {idx}", item)
                return
            if isinstance(value, (str, int, float, bool)) or value is None:
                if value is not None:
                    metrics[label] = value

        if returned is not None:
            consume("Result", returned)

        for idx, figure in enumerate(captured_figures, 1):
            png = _figure_png_bytes(figure)
            if png is not None:
                figures.append((f"Figure {idx}", png))

        if stdout.strip():
            notes.append("Script output:\n\n```text\n" + stdout.strip()[-5000:] + "\n```")
        if stderr.strip():
            warnings.append("The script wrote diagnostic output to stderr. See Method notes / application log if troubleshooting is required.")
            LOGGER.warning("Legacy script %s stderr: %s", definition.path.name, stderr.strip()[-5000:])

        if not metrics and not tables and not figures:
            metrics["Status"] = "Completed"
            notes.append("The legacy function completed but did not return structured data or leave a capturable figure.")

        result = AnalysisResult(
            title=definition.name,
            summary=summary or f"Analysis executed from legacy/{definition.path.name}.",
            metrics=metrics,
            tables=tables,
            figures=figures,
            warnings=warnings,
            notes=notes,
        )

    # Keep the discovered script identity visible in Method notes, not as noisy UI.
    source_note = f"Legacy source: {definition.path.name} · entry point: {definition.entrypoint}()."
    if source_note not in result.notes:
        result.notes.append(source_note)
    return result


class AutoLegacyPlugin:
    def __init__(self, definition: LegacyScriptDefinition):
        self.definition = definition
        digest = hashlib.sha1(str(definition.path.resolve()).encode("utf-8", errors="replace")).hexdigest()[:10]
        self.id = f"legacy_auto_{_ascii(definition.path.stem)}_{digest}"
        self.name = definition.name
        self.description = definition.description

    def confidence(self, files: list[UploadedData]) -> float:
        if not files:
            return 0.0
        if self.definition.sensor == "generic":
            return 0.35
        families = _dataset_families(files)
        if self.definition.sensor in families:
            return 0.92
        compatible = FAMILY_COMPATIBILITY.get(self.definition.sensor or "", {self.definition.sensor or ""})
        if families.intersection(compatible):
            return 0.78
        return 0.0

    def run(self, files: list[UploadedData], options: dict[str, Any]) -> AnalysisResult:
        if not files and self.definition.input_kind != "none":
            raise ValueError("Select data before running this legacy analysis.")

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        captured_figures: list[Any] = []

        with materialized_uploads(files) as (root, paths):
            previous_env = {key: os.environ.get(key) for key in ("AST_INPUT_FILE", "AST_INPUT_FILES", "AST_INPUT_DIR")}
            previous_argv = list(sys.argv)
            os.environ["AST_INPUT_DIR"] = str(root)
            os.environ["AST_INPUT_FILES"] = os.pathsep.join(str(path) for path in paths)
            if paths:
                os.environ["AST_INPUT_FILE"] = str(paths[0])
            # Also expose selected files as command-line arguments. This helps
            # legacy functions that read sys.argv while remaining harmless to
            # scripts that use the inferred function argument directly.
            sys.argv = [str(self.definition.path)] + [str(path) for path in paths]

            plt = None
            previous_show = None
            before_figures: set[int] = set()
            try:
                try:
                    import matplotlib.pyplot as plt  # type: ignore
                    before_figures = set(plt.get_fignums())
                    previous_show = plt.show
                    plt.show = lambda *args, **kwargs: None  # type: ignore[assignment]
                except Exception:
                    plt = None

                with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                    module = _load_module(self.definition)
                    function = getattr(module, self.definition.entrypoint, None)
                    if not callable(function):
                        raise RuntimeError(f"Entry point {self.definition.entrypoint}() is not callable.")
                    returned = _invoke_function(
                        function,
                        self.definition,
                        root=root,
                        paths=paths,
                        files=files,
                        options=options,
                    )

                if plt is not None:
                    for figure_number in sorted(set(plt.get_fignums()) - before_figures):
                        try:
                            captured_figures.append(plt.figure(figure_number))
                        except Exception:
                            continue
                return _normalise_return(
                    self.definition,
                    returned,
                    stdout_buffer.getvalue(),
                    stderr_buffer.getvalue(),
                    captured_figures,
                )
            finally:
                if plt is not None and previous_show is not None:
                    try:
                        plt.show = previous_show  # type: ignore[assignment]
                        for figure in captured_figures:
                            plt.close(figure)
                    except Exception:
                        pass
                sys.argv = previous_argv
                for key, old_value in previous_env.items():
                    if old_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old_value


def discover_legacy_plugins(root: Path = LEGACY_ROOT) -> list[AutoLegacyPlugin]:
    definitions, _statuses = _scan_cached(str(root.resolve()), _legacy_signature(root))
    return [AutoLegacyPlugin(definition) for definition in definitions]


def legacy_script_statuses(root: Path = LEGACY_ROOT) -> list[LegacyScriptStatus]:
    _definitions, statuses = _scan_cached(str(root.resolve()), _legacy_signature(root))
    return list(statuses)
