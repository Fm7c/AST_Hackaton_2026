from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .data_loader import enrich_dataframe


# Keep browser rendering responsive even when scientific files contain hundreds
# of thousands (or millions) of observations. Calculations can still use the
# complete dataframe; this limit is only for visualization.
MAX_GENERIC_PLOT_POINTS = 7000
MAX_MAP_POINTS = 5000
MAX_VECTOR_MAP_ARROWS = 180


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    return enrich_dataframe(df)


def _plot_sample(df: pd.DataFrame, max_points: int = MAX_GENERIC_PLOT_POINTS) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    # Linspace preserves the first and final sample and distributes points
    # uniformly instead of accidentally dropping the tail.
    idx = np.linspace(0, len(df) - 1, max_points, dtype=int)
    return df.iloc[np.unique(idx)].copy()


def summary_metrics(df: pd.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "Rows": len(df),
        "Columns": len(df.columns),
        "Missing values": int(df.isna().sum().sum()),
        "Numeric fields": len(meta.get("numeric_columns", [])),
    }
    time_col = meta.get("time_column")
    if time_col and time_col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[time_col]):
            parsed = pd.to_datetime(df[time_col], errors="coerce")
            if parsed.notna().any():
                metrics["Start"] = parsed.min()
                metrics["End"] = parsed.max()
        else:
            numeric_time = pd.to_numeric(df[time_col], errors="coerce")
            if numeric_time.notna().any():
                metrics["Start time"] = float(numeric_time.min())
                metrics["End time"] = float(numeric_time.max())
    return metrics


def default_timeseries(df: pd.DataFrame, meta: dict[str, Any], columns: list[str] | None = None):
    numeric = list(meta.get("numeric_columns", []))
    if not numeric:
        return None
    columns = columns or numeric[:4]
    columns = [c for c in columns if c in numeric]
    if not columns:
        return None

    plot_df = _plot_sample(df).copy()
    time_col = meta.get("time_column")
    if time_col and time_col in plot_df.columns:
        x_axis = time_col
        x_title = "Time"
    else:
        plot_df = plot_df.rename_axis("sample_index").reset_index()
        x_axis = "sample_index"
        x_title = "Sample"

    # When a special parser combined several source files, keep their traces
    # separate instead of drawing a false line between independent experiments.
    if "file" in plot_df.columns and plot_df["file"].nunique(dropna=True) > 1:
        long = plot_df[[x_axis, "file", *columns]].melt(
            id_vars=[x_axis, "file"], var_name="variable", value_name="value"
        )
        fig = px.line(
            long,
            x=x_axis,
            y="value",
            color="variable",
            line_group="file",
            hover_data=["file"],
            title="Time series / sample series",
        )
    else:
        fig = px.line(
            plot_df,
            x=x_axis,
            y=columns,
            title="Time series" if time_col else "Sample series",
        )

    fig.update_layout(
        xaxis_title=x_title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=95, b=70),
        hovermode="x unified" if len(columns) > 1 else "closest",
    )
    return fig


def _segment_aggregate(df: pd.DataFrame, columns: list[str], aggregation: str, chunks: int | None = None) -> pd.DataFrame:
    clean = df[columns].apply(pd.to_numeric, errors="coerce")
    if clean.dropna(how="all").empty:
        return pd.DataFrame()
    chunks = chunks or min(24, max(5, int(np.sqrt(max(len(clean), 1)))))
    segment = pd.cut(np.arange(len(clean)), bins=chunks, labels=False, include_lowest=True)
    work = clean.copy()
    work.insert(0, "segment", segment)
    grouped = work.groupby("segment", observed=True)
    if aggregation == "median":
        result = grouped.median(numeric_only=True)
    elif aggregation == "max":
        result = grouped.max(numeric_only=True)
    elif aggregation == "min":
        result = grouped.min(numeric_only=True)
    else:
        result = grouped.mean(numeric_only=True)
    return result.rename_axis("segment").reset_index()


def bar_figure(df: pd.DataFrame, column: str, aggregation: str = "mean"):
    grouped = _segment_aggregate(df, [column], aggregation)
    if grouped.empty:
        return None
    return px.bar(grouped, x="segment", y=column, title=f"{column} · {aggregation} by segment")


def multi_bar_figure(
    df: pd.DataFrame,
    columns: list[str],
    aggregation: str = "mean",
    barmode: str = "group",
):
    columns = [c for c in columns if c in df.columns][:8]
    if not columns:
        return None
    grouped = _segment_aggregate(df, columns, aggregation)
    if grouped.empty:
        return None
    long = grouped.melt(id_vars="segment", var_name="variable", value_name="value")
    title_mode = "Stacked" if barmode == "stack" else "Clustered"
    fig = px.bar(
        long,
        x="segment",
        y="value",
        color="variable",
        barmode=barmode,
        title=f"{title_mode} bars · {aggregation} by segment",
    )
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


def gauge_figure(value: float, title: str, min_value: float, max_value: float):
    if not np.isfinite(min_value):
        min_value = 0.0
    if not np.isfinite(max_value) or max_value <= min_value:
        max_value = min_value + 1.0
    return go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(value),
            title={"text": title},
            gauge={"axis": {"range": [float(min_value), float(max_value)]}},
        )
    )


def uv_index_gauge_figure(value: float, title: str = "UV Index"):
    """UV Index gauge with a fixed 0–12 display range."""
    value = float(value)
    shown = max(0.0, min(12.0, value))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=shown,
            number={"suffix": " UVI"},
            title={"text": title},
            gauge={
                "axis": {"range": [0, 12], "tickvals": [0, 2, 5, 7, 10, 12]},
                "bar": {"thickness": 0.28},
                "steps": [
                    {"range": [0, 3], "color": "#2EAD62"},
                    {"range": [3, 6], "color": "#E6C84F"},
                    {"range": [6, 8], "color": "#F39C44"},
                    {"range": [8, 11], "color": "#D9534F"},
                    {"range": [11, 12], "color": "#8E5AB5"},
                ],
                "threshold": {"line": {"width": 4}, "thickness": 0.75, "value": shown},
            },
        )
    )
    fig.update_layout(height=360, margin=dict(t=70, b=25, l=35, r=35))
    return fig


def map_figure(df: pd.DataFrame, meta: dict[str, Any], value_col: str | None = None, title: str | None = None):
    lat = meta.get("latitude_column")
    lon = meta.get("longitude_column")
    if not lat or not lon:
        return None
    plot_df = _plot_sample(df, max_points=MAX_MAP_POINTS).copy()
    plot_df[lat] = pd.to_numeric(plot_df[lat], errors="coerce")
    plot_df[lon] = pd.to_numeric(plot_df[lon], errors="coerce")
    valid = plot_df[lat].between(-90, 90) & plot_df[lon].between(-180, 180)
    plot_df = plot_df.loc[valid]
    if plot_df.empty:
        return None
    kwargs: dict[str, Any] = {
        "data_frame": plot_df,
        "lat": lat,
        "lon": lon,
        "zoom": 8,
        "height": 560,
        "title": title or "Georeferenced measurements",
    }
    if value_col and value_col in plot_df.columns:
        kwargs["color"] = value_col
    kwargs["mapbox_style"] = "open-street-map"
    fig = px.scatter_mapbox(**kwargs)
    # Make plain geographic points easy to see against OpenStreetMap.
    # When a value column is used, its colour scale remains meaningful, so only
    # the marker size is increased. Without a value column, use a clear red marker.
    if value_col and value_col in plot_df.columns:
        fig.update_traces(marker={"size": 9})
    else:
        fig.update_traces(marker={"size": 11, "color": "#E53935"})
    fig.update_layout(margin=dict(l=0, r=0, t=55, b=0))
    return fig


def meteorological_columns(columns: list[str]) -> dict[str, str]:
    """Return recognized meteo fields without assuming units not present in data."""
    normalized = {str(c).lower(): c for c in columns}
    result: dict[str, str] = {}
    candidates = {
        "Temperature": ("temperature", "temperature_c", "temperatura", "temperatura_c", "temp", "temp_c"),
        "Humidity": ("humidity", "humidity_pct", "humidade", "humidade_pct", "rh", "relative_humidity"),
        "Pressure": ("pressure", "pressure_hpa", "pressao", "pressao_hpa", "barometric_pressure"),
    }
    for label, names in candidates.items():
        for name in names:
            if name in normalized:
                result[label] = normalized[name]
                break
    return result


def meteorological_overlay_figure(
    df: pd.DataFrame,
    meta: dict[str, Any],
    meteo_columns: dict[str, str],
    active_layers: list[str] | tuple[str, ...] | None = None,
):
    """Overlay temperature, humidity and pressure on one geographic map.

    The three measurements normally share the same GNSS position, so markers use
    different diameters and opacity. This makes simultaneous layers visible as
    concentric points rather than moving a measurement away from its real
    coordinate. Each layer also has its own colour scale and can be toggled from
    the Plotly legend.
    """
    lat_col = meta.get("latitude_column")
    lon_col = meta.get("longitude_column")
    if not lat_col or not lon_col:
        return None

    requested = list(active_layers or meteo_columns.keys())
    requested = [label for label in requested if label in meteo_columns]
    if not requested:
        return None

    needed = [lat_col, lon_col] + [meteo_columns[label] for label in meteo_columns]
    # Preserve order while removing duplicate column names.
    needed = list(dict.fromkeys(needed))
    work = _plot_sample(df[needed].copy(), max_points=MAX_MAP_POINTS)
    work[lat_col] = pd.to_numeric(work[lat_col], errors="coerce")
    work[lon_col] = pd.to_numeric(work[lon_col], errors="coerce")
    valid_geo = work[lat_col].between(-90, 90) & work[lon_col].between(-180, 180)
    work = work.loc[valid_geo].copy()
    if work.empty:
        return None

    for col in meteo_columns.values():
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    # Outer-to-inner marker order. If all three are enabled the result is a
    # readable concentric overlay at the exact same geographic coordinate.
    style = {
        "Pressure": {"size": 18, "opacity": 0.34, "colorscale": "Viridis", "colorbar_y": 0.18},
        "Humidity": {"size": 13, "opacity": 0.56, "colorscale": "Blues", "colorbar_y": 0.50},
        "Temperature": {"size": 8, "opacity": 0.90, "colorscale": "Turbo", "colorbar_y": 0.82},
    }
    ordered = [label for label in ("Pressure", "Humidity", "Temperature") if label in requested]
    ordered += [label for label in requested if label not in ordered]

    hover_lines: list[str] = []
    time_col = meta.get("time_column")
    for _, row in work.iterrows():
        parts = [
            f"Latitude: {float(row[lat_col]):.6f}",
            f"Longitude: {float(row[lon_col]):.6f}",
        ]
        if time_col and time_col in df.columns and row.name in df.index:
            try:
                parts.insert(0, f"Time: {df.loc[row.name, time_col]}")
            except Exception:
                pass
        for label, col in meteo_columns.items():
            value = row.get(col)
            if pd.notna(value):
                try:
                    parts.append(f"{label}: {float(value):.4g}")
                except Exception:
                    parts.append(f"{label}: {value}")
        hover_lines.append("<br>".join(parts))

    fig = go.Figure()
    for label in ordered:
        col = meteo_columns[label]
        values = pd.to_numeric(work[col], errors="coerce")
        mask = values.notna()
        if not mask.any():
            continue
        cfg = style.get(label, {"size": 10, "opacity": 0.65, "colorscale": "Viridis", "colorbar_y": 0.5})
        layer_hover = [hover_lines[i] for i, keep in enumerate(mask.to_numpy()) if keep]
        fig.add_trace(
            go.Scattermapbox(
                lat=work.loc[mask, lat_col],
                lon=work.loc[mask, lon_col],
                mode="markers",
                marker={
                    "size": cfg["size"],
                    "opacity": cfg["opacity"],
                    "color": values.loc[mask],
                    "colorscale": cfg["colorscale"],
                    "showscale": True,
                    "colorbar": {
                        "title": label,
                        "x": 1.01,
                        "y": cfg["colorbar_y"],
                        "len": 0.27,
                        "thickness": 12,
                    },
                },
                text=layer_hover,
                hoverinfo="text",
                name=label,
            )
        )

    if not fig.data:
        return None
    fig.update_layout(
        title="Meteorological overlay",
        mapbox={
            "style": "open-street-map",
            "zoom": 8,
            "center": {
                "lat": float(work[lat_col].mean()),
                "lon": float(work[lon_col].mean()),
            },
        },
        height=600,
        margin=dict(l=0, r=90, t=65, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
    )
    return fig


def classify_vector_group(label: str, columns: tuple[str, str, str]) -> str:
    text = " ".join([label, *columns]).lower()
    if any(token in text for token in ("mag", "magnetic", "field", "bx", "by", "bz")):
        return "magnetometer"
    if any(token in text for token in ("gyro", "gyroscope", "angular", "gx", "gy", "gz")):
        return "gyroscope"
    if any(token in text for token in ("accel", "acceleration", "ax", "ay", "az")):
        return "accelerometer"
    if "enu" in text or all(token in text for token in ("east", "north", "up")):
        return "enu"
    if "ecef" in text or all(token in text for token in ("xpos", "ypos", "zpos")):
        return "ecef"
    return "vector"


def magnetic_vector_map_figure(
    df: pd.DataFrame,
    meta: dict[str, Any],
    columns: tuple[str, str, str],
):
    """Show horizontal magnetic-field direction directly on a geographic map.

    Arrow lengths are normalized for visibility; magnitude is reported in hover
    text. This avoids pretending that an unknown sensor unit can be converted to
    degrees/metres while still satisfying the directional map requirement.
    """
    lat_col = meta.get("latitude_column")
    lon_col = meta.get("longitude_column")
    if not lat_col or not lon_col:
        return None
    x_col, y_col, z_col = columns
    needed = [lat_col, lon_col, x_col, y_col, z_col]
    work = df[needed].copy()
    for c in needed:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna()
    valid = work[lat_col].between(-90, 90) & work[lon_col].between(-180, 180)
    work = work.loc[valid]
    if work.empty:
        return None
    work = _plot_sample(work, max_points=MAX_VECTOR_MAP_ARROWS)

    horizontal = np.hypot(work[x_col].to_numpy(float), work[y_col].to_numpy(float))
    max_horizontal = float(np.nanmax(horizontal)) if len(horizontal) else 0.0
    if not np.isfinite(max_horizontal) or max_horizontal <= 0:
        return None

    # Constant visual arrow length based on local map spread. Direction comes from
    # the vector components. For longitude, compensate approximately for latitude.
    lat_span = max(float(work[lat_col].max() - work[lat_col].min()), 0.002)
    lon_span = max(float(work[lon_col].max() - work[lon_col].min()), 0.002)
    visual_deg = max(0.00008, min(lat_span, lon_span) * 0.045)

    lats: list[float | None] = []
    lons: list[float | None] = []
    hover: list[str | None] = []
    endpoint_lat: list[float] = []
    endpoint_lon: list[float] = []
    endpoint_hover: list[str] = []

    for _, row in work.iterrows():
        x = float(row[x_col])
        y = float(row[y_col])
        z = float(row[z_col])
        h = math.hypot(x, y)
        if h <= 0:
            continue
        lat0 = float(row[lat_col])
        lon0 = float(row[lon_col])
        north = y / h
        east = x / h
        lat1 = lat0 + north * visual_deg
        coslat = max(0.2, math.cos(math.radians(lat0)))
        lon1 = lon0 + east * visual_deg / coslat
        magnitude = math.sqrt(x * x + y * y + z * z)
        text = f"{x_col}={x:.4g}<br>{y_col}={y:.4g}<br>{z_col}={z:.4g}<br>|B|={magnitude:.4g}"
        lats.extend([lat0, lat1, None])
        lons.extend([lon0, lon1, None])
        hover.extend([text, text, None])
        endpoint_lat.append(lat1)
        endpoint_lon.append(lon1)
        endpoint_hover.append(text)

    fig = go.Figure()
    fig.add_trace(
        go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode="lines",
            line={"width": 2},
            text=hover,
            hoverinfo="text",
            name="Horizontal direction",
        )
    )
    fig.add_trace(
        go.Scattermapbox(
            lat=endpoint_lat,
            lon=endpoint_lon,
            mode="markers",
            marker={"size": 8, "color": "#E53935"},
            text=endpoint_hover,
            hoverinfo="text",
            name="Vector endpoint",
        )
    )
    fig.update_layout(
        title="Magnetic field on map · normalised horizontal vectors",
        mapbox={"style": "open-street-map", "zoom": 8, "center": {"lat": float(work[lat_col].mean()), "lon": float(work[lon_col].mean())}},
        height=600,
        margin=dict(l=0, r=0, t=65, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
    )
    return fig


def describe_numeric(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame()
    numeric_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return numeric_df.describe().T.rename_axis("variable").reset_index()


def detect_vector_groups(columns: list[str]) -> dict[str, tuple[str, str, str]]:
    """Find common three-axis vectors using several naming conventions."""
    original = {str(c).lower(): c for c in columns}
    groups: dict[str, tuple[str, str, str]] = {}

    enu_candidates = [("east_m", "north_m", "up_m"), ("east", "north", "up")]
    for e, n, u in enu_candidates:
        if all(k in original for k in (e, n, u)):
            groups["ENU"] = (original[e], original[n], original[u])
            break

    # Explicit common forms first so user-facing labels are meaningful.
    explicit = {
        "Accelerometer": [
            ("accel_x", "accel_y", "accel_z"), ("accelerometer_x", "accelerometer_y", "accelerometer_z"),
            ("ax", "ay", "az"),
        ],
        "Gyroscope": [
            ("gyro_x", "gyro_y", "gyro_z"), ("gyroscope_x", "gyroscope_y", "gyroscope_z"),
            ("gx", "gy", "gz"),
        ],
        "Magnetometer": [
            ("mag_x", "mag_y", "mag_z"), ("magnetic_x", "magnetic_y", "magnetic_z"),
            ("mx", "my", "mz"), ("bx", "by", "bz"),
        ],
        "ECEF": [("xpos", "ypos", "zpos"), ("x_pos", "y_pos", "z_pos")],
    }
    used_columns: set[str] = set()
    for label, candidate_sets in explicit.items():
        for candidate in candidate_sets:
            if all(c in original for c in candidate):
                value = tuple(original[c] for c in candidate)
                groups[label] = value  # type: ignore[assignment]
                used_columns.update(value)
                break

    by_remainder: dict[str, dict[str, str]] = {}
    for column in columns:
        if column in used_columns:
            continue
        raw = str(column)
        text = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")

        m = re.match(r"^(.*?)[_]?([xyz])$", text)
        if m and m.group(1):
            remainder = m.group(1).rstrip("_")
            by_remainder.setdefault(remainder, {})[m.group(2)] = column

        m = re.match(r"^([xyz])[_]?(.*)$", text)
        if m and m.group(2):
            remainder = m.group(2).lstrip("_")
            if remainder not in {"lat", "latitude", "lon", "long", "longitude"}:
                by_remainder.setdefault(remainder, {})[m.group(1)] = column

    for prefix, axes in by_remainder.items():
        if {"x", "y", "z"}.issubset(axes):
            label = prefix.replace("_", " ").strip().upper() if prefix in {"ecef", "enu"} else prefix.replace("_", " ").strip()
            groups.setdefault(label or "vector", (axes["x"], axes["y"], axes["z"]))
    return groups


def base_mode_capabilities(df: pd.DataFrame, meta: dict[str, Any]) -> dict[str, dict[str, Any]]:
    numeric = list(meta.get("numeric_columns", []))
    vectors = detect_vector_groups(numeric)

    has_geo = bool(meta.get("latitude_column") and meta.get("longitude_column"))
    if has_geo:
        lat = pd.to_numeric(df[meta["latitude_column"]], errors="coerce")
        lon = pd.to_numeric(df[meta["longitude_column"]], errors="coerce")
        valid_geo = lat.between(-90, 90) & lon.between(-180, 180)
        has_geo = bool(valid_geo.sum() >= 1)

    return {
        "Overview": {"available": len(df) > 0, "reason": "dataset summary and statistics"},
        "Time Series": {"available": bool(numeric), "reason": "requires at least one numeric variable"},
        "Bars & Gauges": {"available": bool(numeric), "reason": "requires at least one numeric variable"},
        "Map": {"available": has_geo, "reason": "requires valid latitude and longitude"},
        "Vectors": {"available": bool(vectors), "reason": "requires a recognised XYZ/ENU vector"},
        "Warnings": {"available": bool(numeric), "reason": "requires at least one numeric variable"},
        "Playback": {"available": len(df) >= 2 and bool(numeric), "reason": "requires at least two numeric samples"},
        "Raw Data": {"available": len(df) > 0, "reason": "normalised table"},
    }


def vector_timeseries_figure(
    df: pd.DataFrame,
    meta: dict[str, Any],
    columns: tuple[str, str, str],
    label: str,
):
    x_col, y_col, z_col = columns
    plot_df = _plot_sample(df).copy()
    for col in columns:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    magnitude_col = f"{label}_magnitude"
    plot_df[magnitude_col] = np.sqrt(plot_df[x_col] ** 2 + plot_df[y_col] ** 2 + plot_df[z_col] ** 2)
    time_col = meta.get("time_column")
    if time_col and time_col in plot_df.columns:
        x_axis = time_col
        x_title = "Time"
    else:
        plot_df = plot_df.rename_axis("sample").reset_index()
        x_axis = "sample"
        x_title = "Sample"
    fig = go.Figure()
    for col in (*columns, magnitude_col):
        fig.add_trace(go.Scatter(x=plot_df[x_axis], y=plot_df[col], mode="lines", name=col))
    fig.update_layout(
        title=f"Vector · {label}",
        xaxis_title=x_title,
        yaxis_title="Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=95, b=70),
        hovermode="x unified",
    )
    return fig


def vector_snapshot_figure(values: tuple[float, float, float], title: str):
    x, y, z = map(float, values)
    max_abs = max(abs(x), abs(y), abs(z), 1e-9)
    fig = go.Figure(
        data=go.Cone(
            x=[0.0], y=[0.0], z=[0.0],
            u=[x], v=[y], w=[z],
            sizemode="absolute",
            sizeref=max_abs,
            showscale=False,
            anchor="tail",
        )
    )
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="cube"),
        height=520,
    )
    return fig


def threshold_figure(
    df: pd.DataFrame,
    meta: dict[str, Any],
    column: str,
    low: float | None,
    high: float | None,
):
    plot_df = _plot_sample(df).copy()
    plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    time_col = meta.get("time_column")
    if time_col and time_col in plot_df.columns:
        x_axis = time_col
    else:
        plot_df = plot_df.rename_axis("sample").reset_index()
        x_axis = "sample"
    fig = px.line(plot_df, x=x_axis, y=column, title=f"Threshold monitor · {column}")
    if low is not None:
        fig.add_hline(y=float(low), line_dash="dash", annotation_text="Lower threshold")
    if high is not None:
        fig.add_hline(y=float(high), line_dash="dash", annotation_text="Upper threshold")
    fig.update_layout(margin=dict(t=75, b=65))
    return fig
