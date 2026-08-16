from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core.models import AnalysisResult, UploadedData


EARTH_RADIUS_M = 6371008.8


def _compatible_df(file: UploadedData) -> pd.DataFrame | None:
    if file.dataframe is None:
        return None
    cols = set(file.dataframe.columns)
    if {"lat", "lon"}.issubset(cols):
        return file.dataframe.copy()
    if {"latitude", "longitude"}.issubset(cols):
        df = file.dataframe.copy().rename(columns={"latitude": "lat", "longitude": "lon"})
        return df
    return None


def local_offsets(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat0 = np.radians(float(np.nanmedian(lat)))
    lon0 = np.radians(float(np.nanmedian(lon)))
    latr = np.radians(lat)
    lonr = np.radians(lon)
    east = (lonr - lon0) * np.cos(lat0) * EARTH_RADIUS_M
    north = (latr - lat0) * EARTH_RADIUS_M
    return east, north


class GNSSRTKPlugin:
    id = "gnss_rtk"
    name = "GNSS · RTK / Rover"
    description = "Map, horizontal spread, altitude, fix quality and satellite count for the official RTK_BaseRover MAT files."

    def confidence(self, files: list[UploadedData]) -> float:
        score = 0.0
        for file in files[:8]:
            df = _compatible_df(file)
            if df is not None:
                hits = sum(c in df.columns for c in ("lat", "lon", "altitude", "fixquality", "numsats", "time"))
                low_name = file.name.lower()
                has_rtk_signature = (
                    "fixquality" in df.columns
                    or "numsats" in df.columns
                    or "rover" in low_name
                    or "rtk" in low_name
                    or "gnssresrtk" in low_name
                )
                # Latitude/longitude alone describe many datasets (meteorology,
                # particles, routes...). Do not steal them from the universal
                # generic loader unless there is an actual RTK signature.
                if has_rtk_signature:
                    score = max(score, min(1.0, 0.55 + hits * 0.07))
                else:
                    score = max(score, 0.35)
            elif "gnssresrtk" in file.name.lower() or "rover" in file.name.lower():
                # Many files in GNSSresRTK are MATLAB MCOS tables. Recognize them
                # but keep lower confidence when SciPy cannot expose the table.
                score = max(score, 0.38)
        return score

    def run(self, files: list[UploadedData], options: dict[str, Any]) -> AnalysisResult:
        frames = []
        unreadable = []
        for file in files:
            df = _compatible_df(file)
            if df is None:
                if file.metadata.get("matlab_mcos"):
                    unreadable.append((file.name, file.metadata.get("matlab_tables", [])))
                continue
            df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
            df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
            df = df.dropna(subset=["lat", "lon"]).copy()
            if df.empty:
                continue
            east, north = local_offsets(df["lat"].to_numpy(), df["lon"].to_numpy())
            df["east_m"] = east
            df["north_m"] = north
            df["horizontal_error_m"] = np.hypot(east, north)
            if "time" in df.columns:
                t = pd.to_numeric(df["time"], errors="coerce")
                # The official files use second-of-day and can wrap at midnight.
                delta = t.diff().fillna(0)
                wraps = (delta < -43200).cumsum() * 86400
                continuous = t + wraps
                df["elapsed_s"] = continuous - continuous.iloc[0]
            else:
                df["elapsed_s"] = np.arange(len(df), dtype=float)
            df.insert(0, "file", file.name)
            frames.append(df)

        if not frames:
            detail = ""
            if unreadable:
                detail = " These MAT files contain MATLAB table/MCOS objects that SciPy cannot materialise directly."
            raise ValueError("No RTK MAT file with usable latitude/longitude fields was found." + detail)

        data = pd.concat(frames, ignore_index=True, sort=False)
        max_plot = int(options.get("max_plot_points", 8000))
        plot_df = data.iloc[:: max(1, len(data) // max_plot)].copy()

        figures: list[tuple[str, Any]] = []
        kwargs: dict[str, Any] = dict(
            data_frame=plot_df,
            lat="lat",
            lon="lon",
            hover_name="file",
            zoom=15,
            height=560,
            mapbox_style="open-street-map",
            title="RTK rover positions",
        )
        if "fixquality" in plot_df.columns:
            kwargs["color"] = "fixquality"
        figures.append(("Map", px.scatter_mapbox(**kwargs)))

        scatter = px.scatter(
            plot_df,
            x="east_m",
            y="north_m",
            color="file",
            title="Horizontal dispersion relative to median position",
            labels={"east_m": "East (m)", "north_m": "North (m)"},
        )
        scatter.update_yaxes(scaleanchor="x", scaleratio=1)
        figures.append(("Horizontal precision", scatter))

        if "altitude" in plot_df.columns:
            figures.append((
                "Altitude",
                px.line(plot_df, x="elapsed_s", y="altitude", color="file", title="Altitude over session", labels={"elapsed_s": "Elapsed time (s)", "altitude": "Altitude (m)"}),
            ))

        if "numsats" in plot_df.columns or "fixquality" in plot_df.columns:
            fig = go.Figure()
            for file_name, group in plot_df.groupby("file", sort=False):
                if "numsats" in group.columns:
                    fig.add_trace(go.Scatter(x=group["elapsed_s"], y=group["numsats"], name=f"Satellites · {file_name}"))
                if "fixquality" in group.columns:
                    fig.add_trace(go.Scatter(x=group["elapsed_s"], y=group["fixquality"], name=f"Fix quality · {file_name}", yaxis="y2"))
            fig.update_layout(
                title="Satellites and fix quality",
                xaxis_title="Elapsed time (s)",
                yaxis_title="Number of satellites",
                yaxis2=dict(title="Fix quality", overlaying="y", side="right"),
            )
            figures.append(("Fix / satellites", fig))

        h = data["horizontal_error_m"].dropna()
        metrics = {
            "Files": int(data["file"].nunique()),
            "Samples": len(data),
            "CEP50 (m)": round(float(h.quantile(0.50)), 4),
            "CEP95 (m)": round(float(h.quantile(0.95)), 4),
            "Maximum horizontal error (m)": round(float(h.max()), 4),
            "Mean latitude": round(float(data["lat"].mean()), 7),
            "Mean longitude": round(float(data["lon"].mean()), 7),
        }
        if "numsats" in data.columns:
            metrics["Mean satellites"] = round(float(pd.to_numeric(data["numsats"], errors="coerce").mean()), 2)
        if "fixquality" in data.columns:
            metrics["Fix quality modal"] = int(pd.to_numeric(data["fixquality"], errors="coerce").mode().iloc[0])

        summary = data.groupby("file", as_index=False).agg(
            samples=("horizontal_error_m", "size"),
            cep50_m=("horizontal_error_m", lambda s: s.quantile(0.50)),
            cep95_m=("horizontal_error_m", lambda s: s.quantile(0.95)),
        )

        warnings_out = []
        if unreadable:
            names = ", ".join(name for name, _tables in unreadable[:4])
            warnings_out.append(
                "Some selected files use MATLAB table/MCOS and were skipped in this run: " + names + ". "
                "To analyse them point by point they must first be converted to arrays/CSV or materialised with MATLAB during ingestion."
            )

        preview = data if len(data) <= 15000 else data.iloc[:: max(1, len(data) // 15000)].copy()
        return AnalysisResult(
            title=self.name,
            summary="RTK analysis for BaseRover MAT files containing latitude, longitude, altitude, fix quality and satellite count.",
            metrics=metrics,
            tables={"File summary": summary, "Points": preview},
            figures=figures,
            warnings=warnings_out,
            notes=[
                "BaseRover time is treated as seconds of day and midnight rollover is handled automatically.",
                "Horizontal dispersion is calculated in metres in a local frame around the median session position.",
            ],
            raw={"dataframe": data},
        )


PLUGIN = GNSSRTKPlugin()
