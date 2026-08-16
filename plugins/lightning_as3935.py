from __future__ import annotations

import io
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core.models import AnalysisResult, UploadedData
from core.tempfiles import materialized_uploads
from legacy import analisar_relampagos as legacy


class LightningPlugin:
    id = "lightning_as3935"
    name = "Lightning · AS3935"
    description = "AS3935 lightning detections, disturbances, noise, distance and sensitivity changes."

    def confidence(self, files: list[UploadedData]) -> float:
        if not files:
            return 0.0
        score = 0.0
        for file in files[:5]:
            text = file.raw[:12000].decode("utf-8", errors="ignore").lower()
            hits = sum(
                token in text
                for token in (
                    "lightning strike detected", "franklin lightning detector",
                    "tun_cap", "noise floor", "disturber", "relampago",
                )
            )
            score = max(score, min(1.0, hits / 3.0))
        return score

    @staticmethod
    def _event_time(row: pd.Series):
        if pd.notna(row.get("data_hora")):
            return row.get("data_hora")
        ref = row.get("data_referencia")
        sec = row.get("segundos")
        if pd.notna(ref) and pd.notna(sec):
            try:
                return pd.Timestamp(ref) + pd.to_timedelta(float(sec), unit="s")
            except Exception:
                pass
        return ref

    def run(self, files: list[UploadedData], options: dict[str, Any]) -> AnalysisResult:
        txt_files = [f for f in files if Path(f.name).suffix.lower() == ".txt"]
        if not txt_files:
            raise ValueError("Select at least one AS3935 .txt log.")

        with materialized_uploads(txt_files) as (root, _paths):
            resultado = legacy.analisar_pasta(root)

            events = pd.DataFrame(resultado["eventos"])
            configs = pd.DataFrame(resultado["configuracoes"])
            file_summary = pd.DataFrame(resultado["resumo_ficheiros"])
            unknown = pd.DataFrame(resultado["desconhecidas"])

            figures: list[tuple[str, Any]] = []
            metrics: dict[str, Any] = {
                "Files": len(resultado["ficheiros"]),
                "Sessões": resultado["sessoes"],
            }

            if not file_summary.empty:
                metrics.update({
                    "Lightning": int(file_summary["relampagos"].sum()),
                    "Disturbances": int(file_summary["disturbios"].sum()),
                    "Noise": int(file_summary["ruido"].sum()),
                    "Total detections": int(file_summary["total_deteccoes"].sum()),
                })

            if not events.empty:
                events = events.copy()
                events["momento"] = events.apply(self._event_time, axis=1)

                counts = events["tipo"].value_counts().rename_axis("tipo").reset_index(name="quantidade")
                figures.append((
                    "Detections by type",
                    px.bar(counts, x="tipo", y="quantidade", title="Detections by type"),
                ))

                lightning = events[events["tipo"] == "Relâmpago"].copy()
                if not lightning.empty:
                    numeric_distance = pd.to_numeric(lightning["distancia_km"], errors="coerce")
                    if numeric_distance.notna().any():
                        distance_df = lightning.loc[numeric_distance.notna(), ["momento"]].copy()
                        distance_df["distancia_km"] = numeric_distance[numeric_distance.notna()].to_numpy()
                        figures.append((
                            "Distância estimada dos relâmpagos",
                            px.scatter(
                                distance_df,
                                x="momento",
                                y="distancia_km",
                                title="Distância estimada dos relâmpagos",
                                labels={"momento": "Time", "distancia_km": "Distance (km)"},
                            ),
                        ))

                timeline = events.dropna(subset=["momento"]).copy()
                if not timeline.empty:
                    timeline["ordem_visual"] = range(1, len(timeline) + 1)
                    figures.append((
                        "Detection timeline",
                        px.scatter(
                            timeline,
                            x="momento",
                            y="ordem_visual",
                            color="tipo",
                            hover_data=["ficheiro", "sessao", "estado_sensibilidade", "distancia_km"],
                            title="Detection timeline",
                            labels={"momento": "Time", "ordem_visual": "Event"},
                        ),
                    ))

                sensitivity_counts = (
                    events.groupby(["estado_sensibilidade", "tipo"], dropna=False)
                    .size()
                    .reset_index(name="quantidade")
                )
                if len(sensitivity_counts):
                    figures.append((
                        "Detections by sensitivity",
                        px.bar(
                            sensitivity_counts,
                            x="estado_sensibilidade",
                            y="quantidade",
                            color="tipo",
                            barmode="group",
                            title="Detections by sensitivity setting",
                            labels={"estado_sensibilidade": "Active sensitivity"},
                        ),
                    ))

            # Preserve one of the useful outputs from the original code: Excel report.
            excel_buffer = io.BytesIO()
            # create_excel works with a path, so materialize it inside the temp folder.
            excel_path = root / "analise_relampagos.xlsx"
            legacy.criar_excel(resultado, excel_path)
            excel_buffer.write(excel_path.read_bytes())

            tables = {
                "Eventos": events,
                "Configuration / sensitivity": configs,
                "File summary": file_summary,
            }
            if not unknown.empty:
                tables["Unrecognised lines"] = unknown

            return AnalysisResult(
                title=self.name,
                summary=(
                    "Interpretação dos logs do AS3935 com a lógica do teu analisador original, "
                    "agora apresentada de forma interativa no dashboard."
                ),
                metrics=metrics,
                tables=tables,
                figures=figures,
                downloads={"Relatório Excel AS3935": (excel_buffer.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
                notes=[
                    "The original Lightning / Disturbance / Noise classification and sensitivity state are preserved.",
                    "When no explicit timestamp exists, the reference time from the filename is combined with elapsed seconds when possible.",
                ],
                raw=resultado,
            )


PLUGIN = LightningPlugin()
