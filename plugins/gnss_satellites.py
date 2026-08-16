from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px

from core.models import AnalysisResult, UploadedData


CONSTELLATIONS = {"GPS", "GLONASS", "Galileo", "SBAS"}


def _excel_sheets(file: UploadedData, max_rows_per_sheet: int | None = None) -> dict[str, pd.DataFrame]:
    # Open the workbook once. Re-opening a 10+ MB observation workbook for each
    # constellation is unnecessarily slow. ExcelFile.parse reuses the same reader.
    book = pd.ExcelFile(io.BytesIO(file.raw), engine="openpyxl")
    result: dict[str, pd.DataFrame] = {}
    for sheet in book.sheet_names:
        if sheet not in CONSTELLATIONS:
            continue
        try:
            df = book.parse(sheet_name=sheet, nrows=max_rows_per_sheet)
        except Exception:
            continue
        if not df.empty:
            df["Constellation"] = sheet
            result[sheet] = df
    return result


def _snr_column(df: pd.DataFrame) -> str | None:
    # Observation files use S1C / S1X as carrier-to-noise style measurements;
    # *_SSI columns are signal-strength indicators, not the same value.
    for col in df.columns:
        text = str(col)
        if text.startswith("S1") and not text.endswith("_SSI"):
            return col
    return None


class GNSSSatellitesPlugin:
    id = "gnss_satellites"
    name = "GNSS · Satellites"
    description = "Multi-constellation analysis of the supplied Observacoes/Navegacao Excel files: visibility, satellite count, signal level and ephemeris health."

    def confidence(self, files: list[UploadedData]) -> float:
        score = 0.0
        for file in files[:5]:
            low = file.name.lower()
            if low.endswith((".xlsx", ".xlsm", ".xls")):
                if "observacoes" in low or "navegacao" in low or "gnsssatellites" in low:
                    score = max(score, 0.95)
                elif file.dataframe is not None and "satelliteid" in file.dataframe.columns:
                    score = max(score, 0.7)
        return score

    def run(self, files: list[UploadedData], options: dict[str, Any]) -> AnalysisResult:
        observations = []
        navigation = []
        file_sheet_summary = []

        max_rows_per_sheet = int(options.get("max_rows_per_sheet", 15000))
        for file in files:
            if not file.name.lower().endswith((".xlsx", ".xlsm", ".xls")):
                continue
            sheets = _excel_sheets(file, max_rows_per_sheet=max_rows_per_sheet)
            for constellation, df in sheets.items():
                df = df.copy()
                df["SourceFile"] = file.name
                if "Time" in df.columns:
                    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
                is_observation = "EpochFlag" in df.columns and "SatelliteID" in df.columns
                is_navigation = "SVClockBias" in df.columns and "SatelliteID" in df.columns
                if is_observation:
                    snr = _snr_column(df)
                    if snr:
                        df["PrimarySignal"] = pd.to_numeric(df[snr], errors="coerce")
                    observations.append(df)
                elif is_navigation:
                    navigation.append(df)
                file_sheet_summary.append({
                    "file": file.name,
                    "constellation": constellation,
                    "rows": len(df),
                    "type": "Observations" if is_observation else "Navigation" if is_navigation else "Other",
                })

        if not observations and not navigation:
            raise ValueError("No compatible GNSS sheets were found in the selected Excel files.")

        figures: list[tuple[str, Any]] = []
        tables: dict[str, pd.DataFrame] = {
            "Files / sheets": pd.DataFrame(file_sheet_summary),
        }
        metrics: dict[str, Any] = {}
        notes: list[str] = []

        if observations:
            obs = pd.concat(observations, ignore_index=True, sort=False)
            obs = obs.dropna(subset=["Time", "SatelliteID"]).copy()
            metrics["Observations"] = len(obs)
            metrics["Unique satellites"] = int(obs.groupby("Constellation")["SatelliteID"].nunique().sum())
            metrics["Observed constellations"] = int(obs["Constellation"].nunique())
            metrics["First epoch"] = obs["Time"].min()
            metrics["Last epoch"] = obs["Time"].max()

            counts = (
                obs.groupby(["Time", "Constellation"])["SatelliteID"]
                .nunique()
                .reset_index(name="satellites")
            )
            max_points = int(options.get("max_plot_points", 8000))
            if len(counts) > max_points:
                counts = counts.iloc[:: max(1, len(counts) // max_points)].copy()
            figures.append((
                "Visible satellites",
                px.line(counts, x="Time", y="satellites", color="Constellation", title="Observed satellites by constellation"),
            ))

            timeline_cols = ["Time", "SatelliteID", "Constellation", "SourceFile"]
            if "PrimarySignal" in obs.columns:
                timeline_cols.append("PrimarySignal")
            timeline = obs[timeline_cols].copy()
            if len(timeline) > max_points:
                timeline = timeline.iloc[:: max(1, len(timeline) // max_points)].copy()
            figures.append((
                "Satellite timeline",
                px.scatter(
                    timeline,
                    x="Time",
                    y="SatelliteID",
                    color="PrimarySignal" if "PrimarySignal" in timeline.columns else "Constellation",
                    symbol="Constellation",
                    title="Observed satellites over time",
                ),
            ))

            if "PrimarySignal" in obs.columns and obs["PrimarySignal"].notna().any():
                signal = obs.dropna(subset=["PrimarySignal"])
                figures.append((
                    "Signal level",
                    px.box(signal, x="Constellation", y="PrimarySignal", points=False, title="Primary signal distribution by constellation"),
                ))
                metrics["Mean primary signal"] = round(float(signal["PrimarySignal"].mean()), 2)

            sat_summary = (
                obs.groupby(["Constellation", "SatelliteID"], as_index=False)
                .agg(
                    epochs=("Time", "size"),
                    first_seen=("Time", "min"),
                    last_seen=("Time", "max"),
                    mean_signal=("PrimarySignal", "mean") if "PrimarySignal" in obs.columns else ("SatelliteID", "size"),
                )
            )
            if "PrimarySignal" not in obs.columns:
                sat_summary = sat_summary.rename(columns={"mean_signal": "records"})
            tables["Satellites"] = sat_summary
            tables["Observations (sample)"] = obs if len(obs) <= 12000 else obs.iloc[:: max(1, len(obs) // 12000)].copy()
            notes.append("For observation sheets, the first available S1* field is used as the primary signal level; *_SSI fields are excluded from that choice.")
            notes.append(f"At most {max_rows_per_sheet:,} rows per constellation are read from each workbook in this run to keep the dashboard responsive.")

        if navigation:
            nav = pd.concat(navigation, ignore_index=True, sort=False)
            nav = nav.dropna(subset=["SatelliteID"]).copy()
            metrics["Navigation records"] = len(nav)
            metrics["Satellites with ephemerides"] = int(nav.groupby("Constellation")["SatelliteID"].nunique().sum())
            nav_counts = nav.groupby("Constellation", as_index=False).size().rename(columns={"size": "records"})
            figures.append((
                "Ephemerides",
                px.bar(nav_counts, x="Constellation", y="records", title="Navigation records by constellation"),
            ))

            if "SVClockBias" in nav.columns:
                bias = nav.dropna(subset=["SVClockBias"]).copy()
                figures.append((
                    "Clock bias",
                    px.box(bias, x="Constellation", y="SVClockBias", points=False, title="SV clock bias by constellation"),
                ))

            health_columns = [c for c in ("SVHealth", "Health", "HealthFlags") if c in nav.columns]
            if health_columns:
                health_rows = []
                for col in health_columns:
                    vals = pd.to_numeric(nav[col], errors="coerce")
                    health_rows.append({"field": col, "nonzero": int((vals.fillna(0) != 0).sum()), "valid": int(vals.notna().sum())})
                tables["Health"] = pd.DataFrame(health_rows)

            tables["Navigation (sample)"] = nav if len(nav) <= 12000 else nav.iloc[:: max(1, len(nav) // 12000)].copy()
            notes.append("Navigation fields are presented without reinterpreting orbital parameters; source field names are preserved.")

        return AnalysisResult(
            title=self.name,
            summary="Multi-constellation analysis of the supplied GNSS observation and navigation workbooks.",
            metrics=metrics,
            tables=tables,
            figures=figures,
            notes=notes,
            raw={},
        )


PLUGIN = GNSSSatellitesPlugin()
