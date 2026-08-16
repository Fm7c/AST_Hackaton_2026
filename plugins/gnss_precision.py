from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core.models import AnalysisResult, UploadedData


WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


def ecef_to_geodetic(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    h = np.zeros_like(lat)
    for _ in range(8):
        sin_lat = np.sin(lat)
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        h = p / np.maximum(np.cos(lat), 1e-15) - n
        lat = np.arctan2(z, p * (1.0 - WGS84_E2 * n / np.maximum(n + h, 1e-9)))
    sin_lat = np.sin(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    h = p / np.maximum(np.cos(lat), 1e-15) - n
    return np.degrees(lat), np.degrees(lon), h


def ecef_to_enu(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float]]:
    x0, y0, z0 = (float(np.nanmedian(x)), float(np.nanmedian(y)), float(np.nanmedian(z)))
    lat0_deg, lon0_deg, _ = ecef_to_geodetic(np.array([x0]), np.array([y0]), np.array([z0]))
    lat0 = np.radians(lat0_deg[0])
    lon0 = np.radians(lon0_deg[0])
    dx, dy, dz = np.asarray(x) - x0, np.asarray(y) - y0, np.asarray(z) - z0
    east = -np.sin(lon0) * dx + np.cos(lon0) * dy
    north = -np.sin(lat0) * np.cos(lon0) * dx - np.sin(lat0) * np.sin(lon0) * dy + np.cos(lat0) * dz
    up = np.cos(lat0) * np.cos(lon0) * dx + np.cos(lat0) * np.sin(lon0) * dy + np.sin(lat0) * dz
    return east, north, up, (x0, y0, z0)


def _compatible_df(file: UploadedData) -> pd.DataFrame | None:
    if file.dataframe is None:
        return None
    cols = set(file.dataframe.columns)
    if {"xpos", "ypos", "zpos"}.issubset(cols):
        return file.dataframe.copy()
    return None


class GNSSPrecisionPlugin:
    id = "gnss_precision"
    name = "GNSS · Precision / ECEF"
    description = "Precision analysis for the official MAT datasets: ECEF → WGS84, ENU scatter, CEP50/CEP95 and stability over time."

    def confidence(self, files: list[UploadedData]) -> float:
        score = 0.0
        for file in files[:8]:
            df = _compatible_df(file)
            if df is not None:
                hits = sum(col in df.columns for col in ("xpos", "ypos", "zpos", "itow", "meanacc", "obs", "valid"))
                score = max(score, min(1.0, 0.55 + 0.07 * hits))
            elif "gnssprecision" in file.name.lower():
                score = max(score, 0.35)
        return score

    def run(self, files: list[UploadedData], options: dict[str, Any]) -> AnalysisResult:
        frames = []
        for file in files:
            df = _compatible_df(file)
            if df is None:
                continue
            for col in ("xpos", "ypos", "zpos"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["xpos", "ypos", "zpos"]).copy()
            if df.empty:
                continue
            lat, lon, alt = ecef_to_geodetic(df["xpos"].to_numpy(), df["ypos"].to_numpy(), df["zpos"].to_numpy())
            east, north, up, _origin = ecef_to_enu(df["xpos"].to_numpy(), df["ypos"].to_numpy(), df["zpos"].to_numpy())
            df["latitude"] = lat
            df["longitude"] = lon
            df["altitude_wgs84_m"] = alt
            df["east_m"] = east
            df["north_m"] = north
            df["up_m"] = up
            df["horizontal_error_m"] = np.hypot(east, north)
            df["error_3d_m"] = np.sqrt(east * east + north * north + up * up)
            if "itow" in df.columns:
                itow = pd.to_numeric(df["itow"], errors="coerce")
                df["elapsed_s"] = (itow - itow.iloc[0]) / 1000.0
            else:
                df["elapsed_s"] = np.arange(len(df), dtype=float)
            df.insert(0, "file", file.name)
            frames.append(df)

        if not frames:
            raise ValueError("Select a GNSS precision MAT file containing Xpos/Ypos/Zpos.")

        data = pd.concat(frames, ignore_index=True, sort=False)
        max_plot = int(options.get("max_plot_points", 8000))
        step = max(1, len(data) // max_plot)
        plot_df = data.iloc[::step].copy()

        figures: list[tuple[str, Any]] = []
        figures.append((
            "WGS84 map",
            px.scatter_mapbox(
                plot_df,
                lat="latitude",
                lon="longitude",
                color="horizontal_error_m",
                hover_name="file",
                zoom=14,
                height=560,
                mapbox_style="open-street-map",
                title="GNSS precision · ECEF positions converted to latitude/longitude",
            ),
        ))

        scatter = px.scatter(
            plot_df,
            x="east_m",
            y="north_m",
            color="file",
            title="Horizontal ENU dispersion around the median position",
            labels={"east_m": "East (m)", "north_m": "North (m)"},
        )
        scatter.update_yaxes(scaleanchor="x", scaleratio=1)
        figures.append(("Horizontal precision", scatter))

        figures.append((
            "Horizontal error",
            px.line(
                plot_df,
                x="elapsed_s",
                y="horizontal_error_m",
                color="file",
                title="Horizontal error relative to median position",
                labels={"elapsed_s": "Elapsed time (s)", "horizontal_error_m": "Horizontal error (m)"},
            ),
        ))

        if "meanacc" in data.columns:
            figures.append((
                "MeanAcc",
                px.line(
                    plot_df,
                    x="elapsed_s",
                    y="meanacc",
                    color="file",
                    title="MeanAcc reported in MAT file (unit not documented in supplied data)",
                ),
            ))

        h = pd.to_numeric(data["horizontal_error_m"], errors="coerce").dropna()
        e3 = pd.to_numeric(data["error_3d_m"], errors="coerce").dropna()
        metrics = {
            "Files": int(data["file"].nunique()),
            "Samples": len(data),
            "CEP50 (m)": round(float(h.quantile(0.50)), 4),
            "CEP95 (m)": round(float(h.quantile(0.95)), 4),
            "Maximum horizontal error (m)": round(float(h.max()), 4),
            "3D error P95 (m)": round(float(e3.quantile(0.95)), 4),
            "Mean latitude": round(float(data["latitude"].mean()), 7),
            "Mean longitude": round(float(data["longitude"].mean()), 7),
        }

        summary = (
            data.groupby("file", as_index=False)
            .agg(
                samples=("horizontal_error_m", "size"),
                cep50_m=("horizontal_error_m", lambda s: s.quantile(0.50)),
                cep95_m=("horizontal_error_m", lambda s: s.quantile(0.95)),
                max_horizontal_m=("horizontal_error_m", "max"),
                mean_altitude_m=("altitude_wgs84_m", "mean"),
            )
        )

        preview = data if len(data) <= 15000 else data.iloc[:: max(1, len(data) // 15000)].copy()
        return AnalysisResult(
            title=self.name,
            summary="Converts ECEF X/Y/Z positions to geographic coordinates and measures dispersion in local ENU coordinates.",
            metrics=metrics,
            tables={"File summary": summary, "Points": preview},
            figures=figures,
            notes=[
                "Xpos/Ypos/Zpos are treated as ECEF coordinates in metres; the resulting positions are consistent with the Taveiro area in the tested files.",
                "CEP50/CEP95 are percentiles of horizontal distance from the session median position, useful for comparing precision and repeatability.",
                "MeanAcc is shown as a raw value because its unit is not documented in the supplied data.",
            ],
            raw={"dataframe": data},
        )


PLUGIN = GNSSPrecisionPlugin()
