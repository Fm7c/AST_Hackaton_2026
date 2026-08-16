from __future__ import annotations

import csv
import io
import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core.models import AnalysisResult, UploadedData


CONDITION_PATTERNS = [
    ("Acetone", re.compile(r"acetone", re.I)),
    ("Isopropanol", re.compile(r"isoprop", re.I)),
    ("Clean", re.compile(r"clean", re.I)),
]


def condition_from_name(name: str) -> str:
    for label, pattern in CONDITION_PATTERNS:
        if pattern.search(name):
            return label
    return "Unlabelled"


def parse_volatiles(file: UploadedData) -> pd.DataFrame:
    text = file.raw.decode("utf-8", errors="replace")
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 2:
        return pd.DataFrame()

    data_rows = [row for row in rows[1:] if len(row) >= 2]
    if not data_rows:
        return pd.DataFrame()
    width = max(len(row) for row in data_rows)
    names = ["timestamp"] + [f"channel_{idx:02d}" for idx in range(1, width)]

    normalized_rows = []
    for row in data_rows:
        if len(row) < width:
            row = row + [None] * (width - len(row))
        normalized_rows.append(row[:width])

    df = pd.DataFrame(normalized_rows, columns=names)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in names[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.insert(0, "condition", condition_from_name(file.name))
    df.insert(0, "file", file.name)
    return df


def _top_dynamic_channels(df: pd.DataFrame, channels: list[str], n: int = 6) -> list[str]:
    scores = []
    for col in channels:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() < 3:
            continue
        q05, q95 = s.quantile([0.05, 0.95])
        median = float(np.nanmedian(np.abs(s)))
        scale = median if median > 1e-9 else max(float(s.std()), 1e-9)
        score = abs(float(q95 - q05)) / scale
        scores.append((score, col))
    return [col for _score, col in sorted(scores, reverse=True)[:n]]


class VolatilesPlugin:
    id = "volatiles_multisensor"
    name = "Volatiles · Multi-sensor"
    description = "Explores the official CH4Air5Alc3 CSV series, labelled exposure runs and anonymous sensor channels without inventing undocumented units."

    def confidence(self, files: list[UploadedData]) -> float:
        score = 0.0
        for file in files[:10]:
            low_name = file.name.lower()
            text = file.raw[:5000].decode("utf-8", errors="ignore")
            first = text.splitlines()[0].lower() if text.splitlines() else ""
            second = text.splitlines()[1] if len(text.splitlines()) > 1 else ""
            s = 0.0
            if "ch4air5alc3" in low_name:
                s += 0.7
            if first.strip() == "timestamp,data":
                s += 0.2
            if second.count(",") >= 10:
                s += 0.2
            score = max(score, min(1.0, s))
        return score

    def run(self, files: list[UploadedData], options: dict[str, Any]) -> AnalysisResult:
        frames = []
        for file in files:
            if self.confidence([file]) >= 0.5:
                df = parse_volatiles(file)
                if not df.empty:
                    frames.append(df)
        if not frames:
            raise ValueError("No compatible CH4Air5Alc3 CSV files were found.")

        data = pd.concat(frames, ignore_index=True, sort=False)
        channels = [c for c in data.columns if c.startswith("channel_")]
        top_n = int(options.get("top_channels", 6))
        top = _top_dynamic_channels(data, channels, n=top_n)
        figures: list[tuple[str, Any]] = []

        if top:
            long = data[["timestamp", "file", "condition", *top]].melt(
                id_vars=["timestamp", "file", "condition"], var_name="channel", value_name="value"
            )
            figures.append((
                "Canais mais dinâmicos",
                px.line(
                    long,
                    x="timestamp",
                    y="value",
                    color="channel",
                    line_group="file",
                    facet_row="condition" if data["condition"].nunique() > 1 else None,
                    title="Canais com maior variação relativa",
                    labels={"value": "Valor bruto"},
                ),
            ))

        # Robust z-score heatmap allows channels with very different raw scales to
        # be compared without pretending they share physical units.
        heat_source = data[channels].apply(pd.to_numeric, errors="coerce")
        med = heat_source.median()
        mad = (heat_source - med).abs().median().replace(0, np.nan)
        z = (heat_source - med) / (1.4826 * mad)
        z = z.clip(-8, 8)
        max_points = 1500
        step = max(1, len(z) // max_points)
        z_small = z.iloc[::step]
        x = data["timestamp"].iloc[::step] if data["timestamp"].notna().any() else np.arange(len(z_small))
        heat = go.Figure(
            data=go.Heatmap(z=z_small.to_numpy().T, x=x, y=channels, colorbar_title="robust z")
        )
        heat.update_layout(title="Normalised multi-sensor response profile", xaxis_title="Time", yaxis_title="Channel")
        figures.append(("Fingerprint", heat))

        # Compare labelled exposure files using per-file medians. Raw units remain
        # separate per channel, but the standardized response is comparable.
        file_medians = data.groupby(["file", "condition"], as_index=False)[channels].median(numeric_only=True)
        if len(file_medians) >= 2:
            chan_med = file_medians[channels].median()
            chan_scale = (file_medians[channels] - chan_med).abs().median().replace(0, np.nan)
            standardized = (file_medians[channels] - chan_med) / (1.4826 * chan_scale)
            comp = pd.concat([file_medians[["file", "condition"]], standardized], axis=1)
            comp_long = comp.melt(id_vars=["file", "condition"], var_name="channel", value_name="relative_response")
            comp_long = comp_long.dropna().sort_values("relative_response", key=lambda s: s.abs(), ascending=False)
            # Keep the most informative channels to avoid an unreadable figure.
            informative = comp_long.groupby("channel")["relative_response"].apply(lambda s: s.abs().max()).nlargest(8).index
            comp_long = comp_long[comp_long["channel"].isin(informative)]
            figures.append((
                "Exposure comparison",
                px.bar(
                    comp_long,
                    x="channel",
                    y="relative_response",
                    color="condition",
                    barmode="group",
                    hover_data=["file"],
                    title="Relative response by labelled exposure condition",
                ),
            ))
        else:
            comp_long = pd.DataFrame()

        metrics = {
            "Files": int(data["file"].nunique()),
            "Samples": len(data),
            "Canais numéricos": len(channels),
            "Condições identificadas": int(data["condition"].nunique()),
            "Início": data["timestamp"].min(),
            "Fim": data["timestamp"].max(),
        }

        conditions = (
            data[["file", "condition"]].drop_duplicates().sort_values(["condition", "file"]).reset_index(drop=True)
        )
        channel_stats = data[channels].describe(percentiles=[0.05, 0.5, 0.95]).T.rename_axis("channel").reset_index()
        tables = {
            "Ensaios": conditions,
            "Estatística dos canais": channel_stats,
            "Medições": data if len(data) <= 12000 else data.iloc[:: max(1, len(data) // 12000)].copy(),
        }
        if not comp_long.empty:
            tables["Resposta relativa"] = comp_long

        return AnalysisResult(
            title=self.name,
            summary="Analysis of the supplied volatile-sensor CSV files, including comparison of labelled Clean, Acetone and Isopropanol runs when selected together.",
            metrics=metrics,
            tables=tables,
            figures=figures,
            notes=[
                "The source CSV labels its second field as 'data', while each row contains 17 additional numeric sensor values; the parser handles that structure explicitly.",
                "The supplied data does not document the physical meaning or unit of each of the 17 channels, so they remain labelled channel_01…channel_17.",
                "Exposure comparison uses robust normalisation so response patterns remain comparable across channels with different scales.",
            ],
            raw={"dataframe": data},
        )


PLUGIN = VolatilesPlugin()
