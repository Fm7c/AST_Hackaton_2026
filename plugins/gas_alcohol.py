from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core.models import AnalysisResult, UploadedData


NUMERIC_ROW = re.compile(
    r"^\s*([-+]?\d+(?:[.,]\d+)?)\s*,\s*"
    r"([-+]?\d+(?:[.,]\d+)?)\s*,\s*"
    r"([-+]?\d+(?:[.,]\d+)?)\s*,\s*"
    r"([-+]?\d+(?:[.,]\d+)?)\s*,\s*"
    r"([-+]?\d+(?:[.,]\d+)?)\s*,\s*"
    r"([-+]?\d+(?:[.,]\d+)?)\s*$"
)


def _float(value: str) -> float:
    return float(value.replace(",", "."))


def _target_from_name(name: str) -> str:
    low = name.lower()
    if "ch4" in low:
        return "CH4 dataset"
    if "alcohol" in low or "alcool" in low:
        return "Alcohol dataset"
    return "Gas dataset"


def _short_file_label(name: str) -> str:
    """Human-readable label for archive paths such as Data/Alcohol/file.txt."""
    return PurePosixPath(str(name).replace("\\", "/")).stem


def parse_gas(file: UploadedData) -> tuple[pd.DataFrame, dict[str, Any]]:
    text = file.raw.decode("utf-8", errors="replace")
    rows = []
    header_line = ""
    for line in text.splitlines():
        low = line.lower()
        if "valoradc" in low or "ppm" in low:
            header_line = line
        m = NUMERIC_ROW.match(line)
        if not m:
            continue
        vals = [_float(v) for v in m.groups()]
        rows.append(vals)

    df = pd.DataFrame(
        rows,
        columns=[
            "valor_adc",
            "tensao_v",
            "resistencia_kohm",
            "concentracao_ppm",
            "temperatura_c",
            "humidade_pct",
        ],
    )
    df.insert(0, "sample", range(len(df)))
    df.insert(0, "file", file.name)
    return df, {"header": header_line, "target": _target_from_name(file.name)}


class GasAlcoholPlugin:
    id = "gas_alcohol"
    name = "Gas · Alcohol / CH4"
    description = "Dedicated parser for the Arduino gas logs: ADC, voltage, resistance, reported ppm, temperature and humidity."

    def confidence(self, files: list[UploadedData]) -> float:
        score = 0.0
        for file in files[:8]:
            text = file.raw[:25000].decode("utf-8", errors="ignore").lower()
            hits = sum(token in text for token in ("valoradc", "resist", "ppm", "humidade", "iniciando sensor alcohol"))
            score = max(score, min(1.0, hits / 4.0))
        return score

    def run(self, files: list[UploadedData], options: dict[str, Any]) -> AnalysisResult:
        frames = []
        metadata = []
        for file in files:
            if self.confidence([file]) < 0.45:
                continue
            df, meta = parse_gas(file)
            if not df.empty:
                interval_s = float(options.get("sample_interval_s", 1.0))
                df["elapsed_s"] = df["sample"] * interval_s
                df["target"] = meta["target"]
                frames.append(df)
                metadata.append((file.name, meta))

        if not frames:
            raise ValueError("No valid gas-sensor rows were found in the selected files.")

        data = pd.concat(frames, ignore_index=True)
        data["file_short"] = data["file"].map(_short_file_label)
        figures: list[tuple[str, Any]] = []

        figures.append((
            "Reported concentration",
            px.line(
                data,
                x="elapsed_s",
                y="concentracao_ppm",
                color="file_short",
                hover_data={"file": True, "file_short": False},
                title="Reported sensor concentration",
                labels={"elapsed_s": "Elapsed time (s)", "concentracao_ppm": "Reported concentration (ppm)", "file_short": "File"},
            ),
        ))

        env = go.Figure()
        one_file = data["file"].nunique() == 1
        for file_name, group in data.groupby("file", sort=False):
            short_name = _short_file_label(file_name)
            temp_label = "Temperature" if one_file else f"T · {short_name}"
            rh_label = "Humidity" if one_file else f"RH · {short_name}"
            env.add_trace(
                go.Scatter(
                    x=group["elapsed_s"],
                    y=group["temperatura_c"],
                    name=temp_label,
                    customdata=[[file_name]] * len(group),
                    hovertemplate="%{y}<br>%{customdata[0]}<extra>" + temp_label + "</extra>",
                )
            )
            env.add_trace(
                go.Scatter(
                    x=group["elapsed_s"],
                    y=group["humidade_pct"],
                    name=rh_label,
                    yaxis="y2",
                    customdata=[[file_name]] * len(group),
                    hovertemplate="%{y}<br>%{customdata[0]}<extra>" + rh_label + "</extra>",
                )
            )
        env.update_layout(
            title="Temperature and humidity during measurement",
            xaxis=dict(title=None),
            yaxis=dict(title="Temperature (°C)"),
            yaxis2=dict(title="Humidity (%)", overlaying="y", side="right"),
            # Keep series labels away from the X-axis.  Long archive paths are
            # shortened above; the complete path remains available on hover.
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1.0,
                title=None,
            ),
            margin=dict(l=70, r=80, t=105, b=85),
        )
        # Plotly centres axis titles by default.  Put this one at the lower-right
        # so it cannot collide with the legend, even on a narrow browser window.
        env.add_annotation(
            text="Elapsed time (s)",
            xref="paper",
            yref="paper",
            x=1.0,
            y=-0.12,
            showarrow=False,
            xanchor="right",
            yanchor="top",
        )
        figures.append(("Environment", env))

        figures.append((
            "Electrical response",
            px.scatter(
                data,
                x="resistencia_kohm",
                y="concentracao_ppm",
                color="file_short",
                hover_data={"file": True, "file_short": False},
                title="Reported concentration vs sensor resistance",
                labels={"resistencia_kohm": "Resistance", "concentracao_ppm": "ppm", "file_short": "File"},
            ),
        ))

        if data["file"].nunique() > 1:
            comparison = (
                data.groupby(["file", "file_short", "target"], as_index=False)
                .agg(
                    media_ppm=("concentracao_ppm", "mean"),
                    max_ppm=("concentracao_ppm", "max"),
                    p95_ppm=("concentracao_ppm", lambda s: s.quantile(0.95)),
                )
            )
            figures.append((
                "File comparison",
                px.bar(
                    comparison,
                    x="file_short",
                    y=["media_ppm", "p95_ppm", "max_ppm"],
                    barmode="group",
                    title="Response by run",
                    labels={"file_short": "File"},
                    hover_data={"file": True},
                ),
            ))
        else:
            comparison = pd.DataFrame()

        metrics = {
            "Files": int(data["file"].nunique()),
            "Samples": len(data),
            "Mean ppm": round(float(data["concentracao_ppm"].mean()), 4),
            "Maximum ppm": round(float(data["concentracao_ppm"].max()), 4),
            "Mean temperature (°C)": round(float(data["temperatura_c"].mean()), 3),
            "Mean humidity (%)": round(float(data["humidade_pct"].mean()), 3),
        }

        warnings_out = []
        for name, meta in metadata:
            header = str(meta.get("header", "")).lower()
            if "ch4" in name.lower() and ("etanol" in header or "alcohol" in header):
                warnings_out.append(
                    f"{name}: the filename identifies a CH4 dataset, but the source header describes the ppm field as Ethanol. "
                    "The value is therefore shown as reported concentration without assuming the chemical species."
                )

        tables = {"Measurements": data if len(data) <= 15000 else data.iloc[:: max(1, len(data) // 15000)].copy()}
        if not comparison.empty:
            tables["Comparison"] = comparison

        return AnalysisResult(
            title=self.name,
            summary="Analysis of the supplied Arduino gas logs, preserving the meaning stated in the source files.",
            metrics=metrics,
            tables=tables,
            figures=figures,
            warnings=warnings_out,
            notes=[
                "The logs do not contain a timestamp per sample; elapsed time uses the configured sample interval.",
                "CH4/Ethanol is not inferred when the filename and source header disagree.",
            ],
            raw={"dataframe": data},
        )


PLUGIN = GasAlcoholPlugin()
