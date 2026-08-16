from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core.models import AnalysisResult, UploadedData
from core.tempfiles import materialized_uploads
from legacy import analise_radiacao as legacy


class RadiationPlugin:
    id = "radiation_events"
    name = "Radiation · Events and dose"
    description = "Event intervals, CPM/Hz, estimated dose, Poisson behaviour and possible decay chains."

    def confidence(self, files: list[UploadedData]) -> float:
        if len(files) != 1:
            return 0.1 if files else 0.0
        file = files[0]
        if Path(file.name).suffix.lower() != ".txt":
            return 0.05
        text = file.raw[:20000].decode("utf-8", errors="ignore")
        values = []
        for line in text.splitlines()[:300]:
            try:
                values.append(float(line.strip().replace(",", ".")))
            except ValueError:
                pass
        if len(values) >= 10:
            return 0.85
        if len(values) >= 2:
            return 0.55
        return 0.05

    def run(self, files: list[UploadedData], options: dict[str, Any]) -> AnalysisResult:
        if not files:
            raise ValueError("Select a radiation timestamp file.")

        unit = str(options.get("time_unit", legacy.UNIDADE_TEMPO_ENTRADA))
        chain_limit = float(options.get("chain_limit_s", legacy.LIMITE_EVENTO_EM_CADEIA))
        window_min = float(options.get("window_min", 70.0))

        with materialized_uploads([files[0]]) as (_root, paths):
            timestamps = legacy.carregar_timestamps(paths[0])

        if len(timestamps) < 2:
            raise ValueError("At least two valid timestamps are required.")

        # Reuse the original calculations but make time-unit and chain threshold configurable in the UI.
        # calcular_estatisticas() uses a default argument captured as minutes in the
        # original module, so normalize explicitly before calling it.
        timestamps_min = np.asarray(timestamps, dtype=float) * legacy.fator_tempo_para_minutos(unit)
        original_limit = legacy.LIMITE_EVENTO_EM_CADEIA
        try:
            legacy.LIMITE_EVENTO_EM_CADEIA = chain_limit
            stats = legacy.calcular_estatisticas(timestamps_min)
        finally:
            legacy.LIMITE_EVENTO_EM_CADEIA = original_limit

        interval_df = pd.DataFrame({"intervalo_min": stats["intervalos_min"]})
        interval_df["intervalo_s"] = stats["intervalos_s"]

        event_df = pd.DataFrame({
            "evento": np.arange(1, len(stats["timestamps_min"]) + 1),
            "tempo_min": stats["timestamps_min"],
        })
        event_df["contagem_acumulada"] = event_df["evento"]

        centers, rate_cpm, actual_window = legacy.calcular_taxa_por_janela(stats, largura_janela_min=window_min)
        dose = legacy.cpm_para_usv_h(rate_cpm)
        window_df = pd.DataFrame({
            "tempo_min": centers,
            "taxa_cpm": rate_cpm,
            "taxa_dose_usv_h": dose,
        })

        figures: list[tuple[str, Any]] = []

        if len(interval_df):
            max_interval = float(interval_df["intervalo_min"].max())
            bin_width = float(options.get("hist_bin_min", legacy.LARGURA_BIN_HISTOGRAMA_MIN))
            nbins = max(5, int(np.ceil(max_interval / max(bin_width, 1e-9))))
            hist = px.histogram(
                interval_df,
                x="intervalo_min",
                nbins=nbins,
                histnorm="probability density",
                title="Distribution of detection intervals",
                labels={"intervalo_min": "Detection interval (min)"},
            )
            x = np.linspace(0, max_interval, 1000)
            y = stats["lambda_por_min"] * np.exp(-stats["lambda_por_min"] * x)
            hist.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"Exponencial λ={stats['lambda_por_min']:.3f}/min"))
            figures.append(("Distribuição de intervalos", hist))

        figures.append((
            "Contagem acumulada",
            px.line(
                event_df,
                x="tempo_min",
                y="contagem_acumulada",
                title="Processo de contagem acumulada",
                labels={"tempo_min": "Time (min)", "contagem_acumulada": "Cumulative events"},
            ),
        ))

        rate_fig = px.bar(
            window_df,
            x="tempo_min",
            y="taxa_cpm",
            title=f"Count rate over time · {actual_window:g} min window",
            labels={"tempo_min": "Time (min)", "taxa_cpm": "Count rate (CPM)"},
        )
        rate_fig.add_hline(y=stats["taxa_cpm"], line_dash="dash", annotation_text="Global mean")
        figures.append(("Taxa de contagem", rate_fig))

        valid_dose = window_df.dropna(subset=["taxa_dose_usv_h"])
        if not valid_dose.empty:
            dose_fig = px.line(
                valid_dose,
                x="tempo_min",
                y="taxa_dose_usv_h",
                markers=True,
                title="Estimated dose rate over time",
                labels={"tempo_min": "Time (min)", "taxa_dose_usv_h": "µSv/h"},
            )
            if not np.isnan(stats["taxa_dose_usv_h"]):
                dose_fig.add_hline(y=stats["taxa_dose_usv_h"], line_dash="dash", annotation_text="Global mean")
            figures.append(("Dose rate", dose_fig))

        chains = pd.DataFrame(stats["cadeias_de_decaimento"])
        metrics = {
            "Events": int(stats["n_eventos"]),
            "Duration (min)": round(float(stats["duracao_min"]), 3),
            "Mean rate (CPM)": round(float(stats["taxa_cpm"]), 4),
            "Mean rate (Hz)": round(float(stats["taxa_hz"]), 6),
            "Estimated dose (µSv/h)": None if np.isnan(stats["taxa_dose_usv_h"]) else round(float(stats["taxa_dose_usv_h"]), 6),
            "Possible chains": int(stats["n_cadeias_de_decaimento"]),
            "Isolated events": int(stats["n_eventos_isolados"]),
            "Interval σ/μ": round(float(stats["razao_std_media"]), 3),
        }

        warnings = []
        if np.isnan(stats["taxa_dose_usv_h"]):
            warnings.append("The mean CPM is outside the conversion curve used by the original analysis, so an overall dose estimate is not reported.")

        return AnalysisResult(
            title=self.name,
            summary="Statistical analysis of radiation event timestamps, based on the original radiation script.",
            metrics=metrics,
            tables={
                "Events": event_df,
                "Intervals": interval_df,
                "Windowed rate": window_df,
                "Possible chains": chains,
            },
            figures=figures,
            warnings=warnings,
            notes=[
                "CPM → µSv/h conversion uses the calibration curve and low-count extrapolation from the original analysis script.",
                "The interval used to group nearby detections into a possible chain can be adjusted from the interface.",
            ],
            raw=stats,
        )


PLUGIN = RadiationPlugin()
