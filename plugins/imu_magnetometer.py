from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.data_loader import enrich_dataframe
from core.generic_views import (
    classify_vector_group,
    detect_vector_groups,
    magnetic_vector_map_figure,
    vector_timeseries_figure,
)
from core.models import AnalysisResult, UploadedData


class IMUMagnetometerPlugin:
    id = "imu_magnetometer"
    name = "IMU · Accelerometer / Gyroscope / Magnetometer"
    description = (
        "Dedicated three-axis sensor dashboard: accelerometer, gyroscope and magnetometer, "
        "including vector magnitude and geomagnetic direction on a map when GNSS coordinates exist."
    )

    def _groups(self, file: UploadedData):
        if file.dataframe is None or file.dataframe.empty:
            return {}
        df, meta = enrich_dataframe(file.dataframe)
        groups = detect_vector_groups(list(meta.get("numeric_columns", [])))
        return {name: cols for name, cols in groups.items() if classify_vector_group(name, cols) in {"accelerometer", "gyroscope", "magnetometer"}}

    def confidence(self, files: list[UploadedData]) -> float:
        best = 0.0
        for file in files[:8]:
            groups = self._groups(file)
            if not groups:
                continue
            kinds = {classify_vector_group(name, cols) for name, cols in groups.items()}
            score = min(1.0, 0.35 + 0.22 * len(kinds))
            if "magnetometer" in kinds:
                score += 0.08
            best = max(best, min(1.0, score))
        return best

    def run(self, files: list[UploadedData], options: dict[str, Any]) -> AnalysisResult:
        frames: list[pd.DataFrame] = []
        source_names: list[str] = []
        for file in files:
            if file.dataframe is None or file.dataframe.empty:
                continue
            groups = self._groups(file)
            if not groups:
                continue
            frame = file.dataframe.copy()
            frame.insert(0, "source_file", file.name)
            frames.append(frame)
            source_names.append(file.name)

        if not frames:
            raise ValueError("Select data containing an accelerometer, gyroscope or magnetometer XYZ vector.")

        data = pd.concat(frames, ignore_index=True, sort=False)
        data, meta = enrich_dataframe(data)
        groups = detect_vector_groups(list(meta.get("numeric_columns", [])))
        sensor_groups = {
            name: cols
            for name, cols in groups.items()
            if classify_vector_group(name, cols) in {"accelerometer", "gyroscope", "magnetometer"}
        }
        if not sensor_groups:
            raise ValueError("Vector fields were detected but no complete XYZ samples remained after normalisation.")

        figures: list[tuple[str, Any]] = []
        metrics: dict[str, Any] = {"Files": len(source_names), "Samples": len(data)}
        notes: list[str] = []

        for name, columns in sensor_groups.items():
            kind = classify_vector_group(name, columns)
            label = {
                "accelerometer": "Acelerómetro",
                "gyroscope": "Giroscópio",
                "magnetometer": "Magnetómetro",
            }[kind]
            figures.append((f"{label} · componentes e magnitude", vector_timeseries_figure(data, meta, columns, label)))

            numeric = data[list(columns)].apply(pd.to_numeric, errors="coerce").dropna()
            if numeric.empty:
                continue
            magnitude = np.sqrt((numeric ** 2).sum(axis=1))
            metrics[f"{label} · |v| médio"] = round(float(magnitude.mean()), 5)
            metrics[f"{label} · |v| máximo"] = round(float(magnitude.max()), 5)

            if kind == "magnetometer":
                map_fig = magnetic_vector_map_figure(data, meta, columns)
                if map_fig is not None:
                    figures.append(("Magnetic field on map", map_fig))
                    notes.append(
                        "On the map, arrow length is normalised for readability; direction uses the horizontal components and measured magnitude remains available on hover."
                    )

        kinds = {classify_vector_group(name, cols) for name, cols in sensor_groups.items()}
        missing = []
        if "accelerometer" not in kinds:
            missing.append("acelerómetro")
        if "gyroscope" not in kinds:
            missing.append("giroscópio")
        if "magnetometer" not in kinds:
            missing.append("magnetómetro")
        if missing:
            notes.append("This dataset does not contain: " + ", ".join(missing) + ". Only sensors present in the data are shown.")

        return AnalysisResult(
            title=self.name,
            summary="IMU/vector analysis based only on components present in the selected data.",
            metrics=metrics,
            figures=figures,
            tables={"IMU / magnetometer data": data.head(20000)},
            notes=notes,
            raw={"dataframe": data},
        )


PLUGIN = IMUMagnetometerPlugin()
