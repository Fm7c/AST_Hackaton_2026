from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core.models import AnalysisResult, UploadedData


FLOAT_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?")


def _number(text: str) -> float | None:
    m = FLOAT_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", "."))
    except ValueError:
        return None


def parse_particles(file: UploadedData) -> pd.DataFrame:
    text = file.raw.decode("utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines()]
    rows: list[dict[str, Any]] = []
    i = 0
    sample = 0

    while i < len(lines):
        if not lines[i].lower().startswith("pm"):
            i += 1
            continue

        row: dict[str, Any] = {"file": file.name, "sample": sample}
        sample += 1
        i += 1

        # PM block until NC
        while i < len(lines) and not lines[i].upper().startswith("NC"):
            parts = lines[i].split()
            if len(parts) >= 2:
                try:
                    diameter = float(parts[0].replace(",", "."))
                    value = float(parts[1].replace(",", "."))
                    key = f"pm_{str(diameter).replace('.', '_')}"
                    row[key] = value
                except ValueError:
                    pass
            i += 1

        if i < len(lines) and lines[i].upper().startswith("NC"):
            i += 1
            while i < len(lines):
                low = lines[i].lower()
                if low.startswith("typical particle size") or low.startswith("coordinates") or low.startswith("pm") or not lines[i]:
                    break
                parts = lines[i].split()
                if len(parts) >= 2:
                    try:
                        diameter = float(parts[0].replace(",", "."))
                        value = float(parts[1].replace(",", "."))
                        key = f"nc_{str(diameter).replace('.', '_')}"
                        row[key] = value
                    except ValueError:
                        pass
                i += 1

        if i < len(lines) and lines[i].lower().startswith("typical particle size"):
            row["typical_particle_size"] = _number(lines[i].split(":", 1)[-1])
            i += 1

        if i < len(lines) and lines[i].lower().startswith("coordinates"):
            nums = FLOAT_RE.findall(lines[i].split(":", 1)[-1])
            if len(nums) >= 2:
                try:
                    row["latitude"] = float(nums[0].replace(",", "."))
                    row["longitude"] = float(nums[1].replace(",", "."))
                except ValueError:
                    pass
            i += 1

        if any(key.startswith("pm_") for key in row):
            rows.append(row)

    return pd.DataFrame(rows)


class ParticlesPlugin:
    id = "particles_sps"
    name = "Particles · SPS"
    description = "Particulate matter and number concentration (PM/NC), particle size and optional GNSS route mapping."

    def confidence(self, files: list[UploadedData]) -> float:
        score = 0.0
        for file in files[:8]:
            text = file.raw[:20000].decode("utf-8", errors="ignore").lower()
            hits = sum(token in text for token in ("pm (ug/cm^3)", "typical particle size", "sps sensor", "nc:"))
            score = max(score, min(1.0, hits / 3.0))
        return score

    def run(self, files: list[UploadedData], options: dict[str, Any]) -> AnalysisResult:
        frames = []
        for file in files:
            if self.confidence([file]) >= 0.5:
                df = parse_particles(file)
                if not df.empty:
                    frames.append(df)
        if not frames:
            raise ValueError("No valid PM/NC blocks were found in the selected files.")

        df = pd.concat(frames, ignore_index=True)
        interval_s = float(options.get("sample_interval_s", 1.0))
        df["elapsed_s"] = df.groupby("file").cumcount() * interval_s

        pm_cols = [c for c in ("pm_1_0", "pm_2_5", "pm_4_0", "pm_10_0") if c in df.columns]
        nc_cols = [c for c in ("nc_0_5", "nc_1_0", "nc_2_5", "nc_4_0", "nc_10_0") if c in df.columns]

        figures: list[tuple[str, Any]] = []
        if pm_cols:
            figures.append((
                "PM ao longo da medição",
                px.line(
                    df,
                    x="elapsed_s",
                    y=pm_cols,
                    color_discrete_sequence=px.colors.qualitative.Safe,
                    labels={"elapsed_s": "Elapsed time (s)", "value": "PM", "variable": "Channel"},
                    title="Particulate matter · evolução temporal",
                ),
            ))

        if nc_cols:
            mean_nc = df[nc_cols].apply(pd.to_numeric, errors="coerce").mean().rename_axis("channel").reset_index(name="mean")
            figures.append((
                "Number concentration",
                px.bar(mean_nc, x="channel", y="mean", title="Mean number concentration by threshold"),
            ))

        if "typical_particle_size" in df.columns:
            figures.append((
                "Tamanho típico",
                px.line(df, x="elapsed_s", y="typical_particle_size", title="Typical particle size", labels={"typical_particle_size": "Tamanho"}),
            ))

        geo = df.dropna(subset=[c for c in ("latitude", "longitude") if c in df.columns]) if {"latitude", "longitude"}.issubset(df.columns) else pd.DataFrame()
        if not geo.empty:
            plot_df = geo.iloc[:: max(1, len(geo) // 5000)].copy()
            kwargs: dict[str, Any] = dict(
                data_frame=plot_df,
                lat="latitude",
                lon="longitude",
                zoom=12,
                height=560,
                mapbox_style="open-street-map",
                title="Percurso e concentração de partículas",
                hover_name="file",
            )
            if "pm_2_5" in plot_df.columns:
                kwargs["color"] = "pm_2_5"
            figures.append(("Map", px.scatter_mapbox(**kwargs)))

        metrics: dict[str, Any] = {
            "Files": int(df["file"].nunique()),
            "Samples": len(df),
            "Com coordenadas": int(len(geo)),
        }
        for col, label in (("pm_2_5", "PM2.5 médio"), ("pm_2_5", "PM2.5 máximo"), ("pm_10_0", "PM10 máximo")):
            if col in df.columns:
                values = pd.to_numeric(df[col], errors="coerce")
                if label.endswith("médio"):
                    metrics[label] = round(float(values.mean()), 3)
                else:
                    metrics[label] = round(float(values.max()), 3)
        if "typical_particle_size" in df.columns:
            metrics["Mean typical particle size"] = round(float(pd.to_numeric(df["typical_particle_size"], errors="coerce").mean()), 3)

        warnings_out = []
        if geo.empty:
            warnings_out.append("This file does not include coordinates, so map view is not available for this measurement.")

        preview = df if len(df) <= 10000 else df.iloc[:: max(1, len(df) // 10000)].copy()
        return AnalysisResult(
            title=self.name,
            summary="Particulate-matter analysis of the supplied SPS files, including PM, number concentration, typical particle size and coordinates when available.",
            metrics=metrics,
            tables={"Measurements": preview},
            figures=figures,
            warnings=warnings_out,
            notes=[
                "The files do not provide per-sample timestamps; elapsed time uses a configurable interval while preserving sample order.",
                "Quando existem coordenadas, o percurso é apresentado diretamente no mapa e pode ser colorido por PM2.5.",
            ],
            raw={"dataframe": df},
        )


PLUGIN = ParticlesPlugin()
