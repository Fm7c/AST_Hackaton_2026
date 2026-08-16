from __future__ import annotations

import hashlib
import logging
import re
from collections import defaultdict
from typing import Any

import pandas as pd

from .data_loader import normalize_columns
from .models import NormalizedDataset, UploadedData
from .plugin_manager import ranked_plugins

LOGGER = logging.getLogger("ast_sensor_analytics")


DEFAULT_PLUGIN_OPTIONS: dict[str, dict[str, Any]] = {
    "radiation_events": {"time_unit": "min", "chain_limit_s": 2.0, "window_min": 70.0, "hist_bin_min": 0.05},
    "uv_multisensor": {"gap_seconds": 90},
    "particles_sps": {"sample_interval_s": 1.0},
    "gas_alcohol": {"sample_interval_s": 1.0},
    "volatiles_multisensor": {"top_channels": 6},
    "gnss_precision": {"max_plot_points": 8000},
    "gnss_rtk": {"max_plot_points": 8000},
    "gnss_satellites": {"max_plot_points": 8000, "max_rows_per_sheet": 15000},
    "lightning_as3935": {},
    "rtk_mcos_catalog": {},
    "imu_magnetometer": {},
}

# These current plugins intentionally analyse one file at a time.
SINGLE_FILE_NORMALIZATION = {"radiation_events", "uv_multisensor"}


def _dataset_id(*parts: str) -> str:
    text = "|".join(parts)
    digest = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:12]
    base = re.sub(r"[^a-z0-9]+", "_", parts[0].lower()).strip("_")[:36] or "dataset"
    return f"{base}_{digest}"


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = normalize_columns(df.copy())
    # Remove completely empty columns but preserve sparse sensor fields.
    result = result.dropna(axis=1, how="all")
    return result.reset_index(drop=True)


def _from_result(plugin: Any, files: list[UploadedData], confidence: float, result: Any) -> list[NormalizedDataset]:
    source_names = [f.name for f in files]
    base_name = plugin.name
    description = result.summary or plugin.description
    notes = list(getattr(result, "notes", []) or []) + list(getattr(result, "warnings", []) or [])

    frames: list[tuple[str, pd.DataFrame]] = []
    raw = getattr(result, "raw", {}) or {}
    if isinstance(raw, dict) and isinstance(raw.get("dataframe"), pd.DataFrame):
        frames.append(("Normalised measurements", raw["dataframe"]))
    elif plugin.id == "rtk_mcos_catalog" and isinstance(raw.get("catalog"), pd.DataFrame):
        frames.append(("RTK catalogue", raw["catalog"]))
    elif plugin.id == "lightning_as3935":
        aliases = [
            ("Events", ("Events", "Eventos")),
            ("Configuration / sensitivity", ("Configuration / sensitivity", "Configuração / sensibilidade")),
            ("File summary", ("File summary", "Resumo por ficheiro")),
        ]
        for label, keys in aliases:
            table = next((result.tables.get(key) for key in keys if key in result.tables), None)
            if isinstance(table, pd.DataFrame) and not table.empty:
                frames.append((label, table))
    elif plugin.id == "radiation_events":
        aliases = [
            ("Events", ("Events", "Eventos")),
            ("Intervals", ("Intervals", "Intervalos")),
            ("Windowed rate", ("Windowed rate", "Taxa por janela")),
            ("Possible chains", ("Possible chains", "Possíveis cadeias")),
        ]
        for label, keys in aliases:
            table = next((result.tables.get(key) for key in keys if key in result.tables), None)
            if isinstance(table, pd.DataFrame) and not table.empty:
                frames.append((label, table))
    elif plugin.id == "gnss_satellites":
        aliases = [
            ("Observations (sample)", ("Observations (sample)", "Observações (amostra)")),
            ("Navigation (sample)", ("Navigation (sample)", "Navegação (amostra)")),
            ("Satellites", ("Satellites", "Satélites")),
            ("Files / sheets", ("Files / sheets", "Ficheiros / folhas")),
        ]
        for label, keys in aliases:
            table = next((result.tables.get(key) for key in keys if key in result.tables), None)
            if isinstance(table, pd.DataFrame) and not table.empty:
                frames.append((label, table))
    else:
        # Generic fallback for a plugin that returns useful tables but no raw dataframe.
        candidates = [(name, table) for name, table in result.tables.items() if isinstance(table, pd.DataFrame) and not table.empty]
        if candidates:
            name, table = max(candidates, key=lambda item: (len(item[1]), len(item[1].columns)))
            frames.append((name, table))

    datasets: list[NormalizedDataset] = []
    for index, (label, frame) in enumerate(frames):
        clean = _clean_frame(frame)
        if clean.empty:
            continue
        display = base_name if len(frames) == 1 else f"{base_name} · {label}"
        datasets.append(
            NormalizedDataset(
                id=_dataset_id(plugin.id, label, *source_names, str(index)),
                name=display,
                dataframe=clean,
                source_files=source_names,
                parser=f"Special parser · {plugin.name}",
                description=description,
                special_plugin_id=plugin.id,
                confidence=float(confidence),
                notes=notes,
            )
        )
    return datasets


def _generic_dataset(file: UploadedData, suffix: str = "") -> NormalizedDataset | None:
    if file.dataframe is None or file.dataframe.empty:
        return None
    frame = _clean_frame(file.dataframe)
    if frame.empty:
        return None
    name = f"{file.name}{suffix}"
    return NormalizedDataset(
        id=_dataset_id("generic", file.name, suffix),
        name=name,
        dataframe=frame,
        source_files=[file.name],
        parser="Generic table loader",
        description="Table read from the source file and normalised for the dashboard.",
        confidence=1.0,
    )


LIGHTWEIGHT_SPECIAL_IDS = {"gas_alcohol", "particles_sps", "volatiles_multisensor", "gnss_precision", "uv_multisensor"}


def _lightweight_special_datasets(plugin: Any, files: list[UploadedData], confidence: float) -> list[NormalizedDataset] | None:
    """Normalise common official datasets without constructing analysis figures.

    The dedicated ``plugin.run`` methods create Plotly figures and statistical
    summaries. That is appropriate only when the user opens Analysis, not while
    the Overview page is merely preparing data. On large Windows datasets those
    figures were the main source of multi-minute startup delays.
    """
    plugin_id = plugin.id
    source_names = [f.name for f in files]
    frames: list[pd.DataFrame] = []

    if plugin_id == "gas_alcohol":
        from plugins.gas_alcohol import parse_gas, _target_from_name
        for file in files:
            frame, _meta = parse_gas(file)
            if frame.empty:
                continue
            frame["elapsed_s"] = frame["sample"] * 1.0
            frame["target"] = _target_from_name(file.name)
            frames.append(frame)

    elif plugin_id == "particles_sps":
        from plugins.particles_sps import parse_particles
        for file in files:
            frame = parse_particles(file)
            if frame.empty:
                continue
            frame["elapsed_s"] = frame.groupby("file").cumcount().astype(float)
            frames.append(frame)

    elif plugin_id == "volatiles_multisensor":
        from plugins.volatiles_multisensor import parse_volatiles
        for file in files:
            frame = parse_volatiles(file)
            if not frame.empty:
                frames.append(frame)

    elif plugin_id == "uv_multisensor":
        from core.tempfiles import materialized_uploads
        from legacy import plot_medicoes_uv_todos_sensores as legacy_uv
        for file in files:
            with materialized_uploads([file]) as (_root, paths):
                frame = legacy_uv.carregar_dados(paths[0])
            if not frame.empty:
                frame.insert(0, "file", file.name)
                frames.append(frame)

    elif plugin_id == "gnss_precision":
        import numpy as np
        from plugins.gnss_precision import _compatible_df, ecef_to_geodetic, ecef_to_enu
        for file in files:
            frame = _compatible_df(file)
            if frame is None:
                continue
            for col in ("xpos", "ypos", "zpos"):
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
            frame = frame.dropna(subset=["xpos", "ypos", "zpos"]).copy()
            if frame.empty:
                continue
            x = frame["xpos"].to_numpy()
            y = frame["ypos"].to_numpy()
            z = frame["zpos"].to_numpy()
            lat, lon, alt = ecef_to_geodetic(x, y, z)
            east, north, up, _origin = ecef_to_enu(x, y, z)
            frame["latitude"] = lat
            frame["longitude"] = lon
            frame["altitude_wgs84_m"] = alt
            frame["east_m"] = east
            frame["north_m"] = north
            frame["up_m"] = up
            frame["horizontal_error_m"] = np.hypot(east, north)
            frame["error_3d_m"] = np.sqrt(east * east + north * north + up * up)
            if "itow" in frame.columns:
                itow = pd.to_numeric(frame["itow"], errors="coerce")
                frame["elapsed_s"] = (itow - itow.iloc[0]) / 1000.0
            else:
                frame["elapsed_s"] = np.arange(len(frame), dtype=float)
            frame.insert(0, "file", file.name)
            frames.append(frame)
    else:
        return None

    if not frames:
        return []
    data = pd.concat(frames, ignore_index=True, sort=False)
    clean = _clean_frame(data)
    if clean.empty:
        return []
    return [NormalizedDataset(
        id=_dataset_id(plugin_id, "normalised", *source_names),
        name=plugin.name,
        dataframe=clean,
        source_files=source_names,
        parser=f"Special parser · {plugin.name}",
        description=plugin.description,
        special_plugin_id=plugin_id,
        confidence=float(confidence),
        notes=[],
    )]


def build_universal_datasets(files: list[UploadedData], minimum_special_confidence: float = 0.50) -> list[NormalizedDataset]:
    """Convert active files into datasets consumable by the base visualisations.

    Normalisation is deliberately lightweight. A dedicated scientific analysis is
    only executed when the user opens Analysis; Overview/Visualisation should not
    pay the cost of building analysis figures.
    """
    if not files:
        return []

    grouped: dict[str, list[tuple[UploadedData, Any, float]]] = defaultdict(list)
    generic_files: list[UploadedData] = []
    fallback_special: list[tuple[UploadedData, Any, float]] = []

    for file in files:
        ranked = ranked_plugins([file])
        best = ranked[0] if ranked else None

        # Most already-tabular sources need no special parser for base views.
        # Keep dedicated lightweight handling only where it adds essential fields
        # (e.g. ECEF -> latitude/longitude) or where the raw text is not tabular.
        if file.dataframe is not None and not file.dataframe.empty:
            if best is not None and best.confidence >= minimum_special_confidence and best.plugin.id in {"gnss_precision", "volatiles_multisensor", "particles_sps", "gas_alcohol", "uv_multisensor"}:
                grouped[best.plugin.id].append((file, best.plugin, best.confidence))
            else:
                generic_files.append(file)
            continue

        if best is not None and best.confidence >= minimum_special_confidence:
            if best.plugin.id in LIGHTWEIGHT_SPECIAL_IDS:
                grouped[best.plugin.id].append((file, best.plugin, best.confidence))
            else:
                fallback_special.append((file, best.plugin, best.confidence))
        else:
            generic_files.append(file)

    datasets: list[NormalizedDataset] = []
    consumed_names: set[str] = set()

    # Fast parsers for the common official datasets.
    for plugin_id, items in grouped.items():
        plugin = items[0][1]
        groups = [[item] for item in items] if plugin_id in SINGLE_FILE_NORMALIZATION else [items]
        for subgroup in groups:
            subfiles = [item[0] for item in subgroup]
            confidence = max(item[2] for item in subgroup)
            try:
                derived = _lightweight_special_datasets(plugin, subfiles, confidence) or []
            except Exception as exc:
                LOGGER.exception("Lightweight dataset normalisation failed in plugin %s: %s", plugin_id, exc)
                derived = []
            if derived:
                datasets.extend(derived)
                consumed_names.update(f.name for f in subfiles)
            else:
                generic_files.extend(subfiles)

    # Less common raw formats (UV/radiation/lightning/MCOS) keep their existing
    # parser path. Their supplied examples are small, so this does not affect the
    # normal startup path for the official large datasets.
    by_plugin: dict[str, list[tuple[UploadedData, Any, float]]] = defaultdict(list)
    for item in fallback_special:
        by_plugin[item[1].id].append(item)
    for plugin_id, items in by_plugin.items():
        plugin = items[0][1]
        groups = [[item] for item in items] if plugin_id in SINGLE_FILE_NORMALIZATION else [items]
        for subgroup in groups:
            subfiles = [item[0] for item in subgroup]
            confidence = max(item[2] for item in subgroup)
            options = dict(DEFAULT_PLUGIN_OPTIONS.get(plugin_id, {}))
            try:
                result = plugin.run(subfiles, options)
                derived = _from_result(plugin, subfiles, confidence, result)
            except Exception as exc:
                LOGGER.exception("Dataset normalisation failed in plugin %s: %s", plugin_id, exc)
                derived = []
            if derived:
                datasets.extend(derived)
                consumed_names.update(f.name for f in subfiles)
            else:
                generic_files.extend(subfiles)

    # Generic fallback for ordinary tables and parser failures.
    seen: set[str] = set()
    for file in [*generic_files, *files]:
        if file.name in consumed_names or file.name in seen:
            continue
        seen.add(file.name)
        generic = _generic_dataset(file)
        if generic is not None:
            datasets.append(generic)

    datasets.sort(key=lambda d: (d.parser.startswith("Generic"), d.name.casefold()))
    return datasets


def file_signature(files: list[UploadedData]) -> tuple[Any, ...]:
    """Cheap process-local cache key for the active selection."""
    signature = []
    for file in files:
        raw = file.raw or b""
        head = raw[:4096]
        tail = raw[-4096:] if len(raw) > 4096 else b""
        digest = hashlib.sha1(head + tail).hexdigest()[:12]
        shape = None if file.dataframe is None else tuple(file.dataframe.shape)
        signature.append((file.name, file.kind, len(raw), digest, shape))
    return tuple(signature)
