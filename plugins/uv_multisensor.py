from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from core.models import AnalysisResult, UploadedData
from core.generic_views import uv_index_gauge_figure
from core.tempfiles import materialized_uploads
from legacy import plot_medicoes_uv_todos_sensores as legacy


class UVMultiSensorPlugin:
    id = "uv_multisensor"
    name = "UV · Multi-sensor"
    description = "AS7331, UVI reference, OPT3001 and VEML3328 measurements with temporal comparison and sample-gap diagnostics."

    def confidence(self, files: list[UploadedData]) -> float:
        if not files:
            return 0.0
        score = 0.0
        for file in files[:3]:
            text = file.raw[:30000].decode("utf-8", errors="ignore").lower()
            hits = sum(token in text for token in ("hora_python", "índice_uv_api", "uva=", "uvb=", "opt3001", "veml_"))
            score = max(score, min(1.0, hits / 4.0))
        return score

    def run(self, files: list[UploadedData], options: dict[str, Any]) -> AnalysisResult:
        if not files:
            raise ValueError("Select a UV measurement file.")

        gap_seconds = int(options.get("gap_seconds", 90))
        with materialized_uploads([files[0]]) as (_root, paths):
            df = legacy.carregar_dados(paths[0])

        raw_df = df.copy()
        df = legacy.inserir_lacunas(df, tolerancia=pd.Timedelta(seconds=gap_seconds).to_pytimedelta())

        figures: list[tuple[str, Any]] = []

        # Fixed 0–12 UV Index display scale.
        uvi_candidates = []
        for col in ("uvi", "uvi_api"):
            if col in raw_df.columns:
                vals = pd.to_numeric(raw_df[col], errors="coerce").dropna()
                if not vals.empty:
                    uvi_candidates.append((col, vals))
        if uvi_candidates:
            preferred_col, preferred_vals = next((item for item in uvi_candidates if item[0] == "uvi"), uvi_candidates[0])
            mean_uvi = float(preferred_vals.mean())
            figures.append((
                "UV Index · 0–12 scale",
                uv_index_gauge_figure(mean_uvi, f"Average UV Index · {preferred_col}"),
            ))

        uv_fig = go.Figure()
        for column, label in (("uva", "UVA"), ("uvb", "UVB")):
            if column in df.columns:
                uv_fig.add_trace(go.Scatter(x=df["timestamp"], y=df[column], mode="lines", name=label))

        # The original parser already computes this correction. Use it explicitly in the dashboard.
        uvc_column = "uvc_corrigido" if "uvc_corrigido" in df.columns else "uvc"
        if uvc_column in df.columns:
            uv_fig.add_trace(go.Scatter(x=df["timestamp"], y=df[uvc_column], mode="lines", name="Corrected UVC" if uvc_column == "uvc_corrigido" else "UVC"))

        if "uvi" in df.columns:
            uv_fig.add_trace(go.Scatter(x=df["timestamp"], y=df["uvi"], mode="lines", name="UVI sensor", yaxis="y2"))
        if "uvi_api" in df.columns:
            uv_fig.add_trace(go.Scatter(x=df["timestamp"], y=df["uvi_api"], mode="lines+markers", name="API UVI reference", yaxis="y2"))

        uv_fig.update_layout(
            title="UV irradiance and UV Index · sensor vs reference",
            xaxis_title="Hora da medição",
            yaxis_title="Irradiância",
            yaxis2=dict(title="Índice UV", overlaying="y", side="right"),
            legend=dict(orientation="h"),
        )
        figures.append(("UV / UVI", uv_fig))

        if "opt3001_lux" in df.columns:
            figures.append((
                "Luminosidade ambiente",
                px.line(df, x="timestamp", y="opt3001_lux", title="OPT3001 · Luminosidade ambiente", labels={"opt3001_lux": "Lux"}),
            ))

        rgb_cols = [c for c in ("veml_C", "veml_R", "veml_G", "veml_B", "veml_I") if c in df.columns]
        if rgb_cols:
            figures.append((
                "VEML3328 · canais",
                px.line(df, x="timestamp", y=rgb_cols, title="VEML3328 · Canais RGB + Clear + IV"),
            ))

        if "veml_lux" in df.columns or "veml_cct" in df.columns:
            fig = go.Figure()
            if "veml_lux" in df.columns:
                fig.add_trace(go.Scatter(x=df["timestamp"], y=df["veml_lux"], name="Lux VEML"))
            if "veml_cct" in df.columns:
                fig.add_trace(go.Scatter(x=df["timestamp"], y=df["veml_cct"], name="CCT (K)", yaxis="y2"))
            fig.update_layout(
                title="VEML3328 · Lux e CCT",
                yaxis_title="Lux",
                yaxis2=dict(title="CCT (K)", overlaying="y", side="right"),
            )
            figures.append(("VEML3328 · Lux/CCT", fig))

        if "uvTemp" in df.columns:
            figures.append((
                "UV sensor temperature",
                px.line(df, x="timestamp", y="uvTemp", title="AS7331 · Internal temperature", labels={"uvTemp": "°C"}),
            ))

        metrics: dict[str, Any] = {
            "Records": len(raw_df),
            "Start": raw_df["timestamp"].min(),
            "End": raw_df["timestamp"].max(),
        }
        if "uvi" in raw_df.columns:
            _uvi = pd.to_numeric(raw_df["uvi"], errors="coerce").dropna()
            if not _uvi.empty:
                metrics["Mean UVI sensor"] = round(float(_uvi.mean()), 4)
        if "uvi_api" in raw_df.columns:
            _uvi_api = pd.to_numeric(raw_df["uvi_api"], errors="coerce").dropna()
            if not _uvi_api.empty:
                metrics["Mean UVI reference"] = round(float(_uvi_api.mean()), 4)
        for column, label in (
            ("uvi", "UVI máx. sensor"),
            ("uvi_api", "UVI máx. API"),
            ("uva", "UVA máx."),
            ("uvb", "UVB máx."),
            (uvc_column, "UVC máx."),
            ("opt3001_lux", "Lux máx."),
        ):
            if column in raw_df.columns and pd.to_numeric(raw_df[column], errors="coerce").notna().any():
                metrics[label] = round(float(pd.to_numeric(raw_df[column], errors="coerce").max()), 4)

        gaps = raw_df["timestamp"].sort_values().diff().dt.total_seconds()
        gap_df = pd.DataFrame({"timestamp": raw_df["timestamp"], "gap_s": gaps})
        gap_df = gap_df[gap_df["gap_s"] > gap_seconds].reset_index(drop=True)

        tables = {
            "Medições": raw_df,
            "Sampling gaps": gap_df,
        }

        notes = [
            f"Gaps longer than {gap_seconds} s are inserted explicitly so charts do not connect missing periods as continuous data.",
            f"A correção de crosstalk UVC usa α={legacy.ALFA_CROSSTALK_UVC:g}, conforme o teu código.",
        ]

        return AnalysisResult(
            title=self.name,
            summary="Integrated UV, ambient-light and colour-channel analysis using the existing UV parser.",
            metrics=metrics,
            tables=tables,
            figures=figures,
            notes=notes,
            raw={"dataframe": raw_df},
        )


PLUGIN = UVMultiSensorPlugin()
