from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data_loader import build_uploaded_data, enrich_dataframe
from core.generic_views import (
    classify_vector_group,
    detect_vector_groups,
    magnetic_vector_map_figure,
    meteorological_columns,
    meteorological_overlay_figure,
    multi_bar_figure,
    uv_index_gauge_figure,
)
from core.plugin_manager import ranked_plugins


def main() -> int:
    failures: list[str] = []

    fusion_path = ROOT / "sample_data" / "generic_meteo_gnss.csv"
    fusion = build_uploaded_data(fusion_path.name, fusion_path.read_bytes())
    if fusion.dataframe is None:
        failures.append("synthetic sensor-fusion CSV did not load")
    else:
        df, meta = enrich_dataframe(fusion.dataframe)
        groups = detect_vector_groups(list(meta["numeric_columns"]))
        kinds = {classify_vector_group(name, cols) for name, cols in groups.items()}
        for expected in ("accelerometer", "gyroscope", "magnetometer"):
            if expected not in kinds:
                failures.append(f"missing vector group: {expected}")

        meteo = meteorological_columns(list(df.columns))
        if set(meteo) != {"Temperature", "Humidity", "Pressure"}:
            failures.append(f"meteo layers incomplete: {meteo}")
        weather_overlay = meteorological_overlay_figure(df, meta, meteo, ["Temperature", "Humidity", "Pressure"])
        if weather_overlay is None or len(weather_overlay.data) != 3:
            failures.append("temperature/humidity/pressure overlay was not generated as three geographic layers")

        magnetic = next((cols for name, cols in groups.items() if classify_vector_group(name, cols) == "magnetometer"), None)
        if magnetic is None or magnetic_vector_map_figure(df, meta, magnetic) is None:
            failures.append("magnetic vector map not generated")

        if multi_bar_figure(df, ["temperature", "humidity", "pressure"], barmode="group") is None:
            failures.append("clustered bars not generated")
        if multi_bar_figure(df, ["temperature", "humidity", "pressure"], barmode="stack") is None:
            failures.append("stacked bars not generated")

        gauge = uv_index_gauge_figure(7.0)
        try:
            axis_range = list(gauge.data[0]["gauge"]["axis"]["range"])
        except Exception:
            axis_range = []
        if axis_range != [0, 12]:
            failures.append(f"UV gauge range is not 0–12: {axis_range}")

        ranked = ranked_plugins([fusion])
        if not ranked or ranked[0].plugin.id != "imu_magnetometer" or ranked[0].confidence < 0.8:
            failures.append("IMU/Magnetometer plugin is not strongly recognized")

    uv_path = ROOT / "sample_data" / "medicoes_uv.txt"
    uv = build_uploaded_data(uv_path.name, uv_path.read_bytes())
    uv_plugin = next((item.plugin for item in ranked_plugins([uv]) if item.plugin.id == "uv_multisensor"), None)
    if uv_plugin is None:
        failures.append("UV plugin missing")
    else:
        result = uv_plugin.run([uv], {"gap_seconds": 90})
        if not any("0–12" in title for title, _fig in result.figures):
            failures.append("UV special mode does not expose fixed 0–12 gauge")

    if failures:
        print("FAIL")
        for failure in failures:
            print(" -", failure)
        return 1

    print("Hackathon requirement smoke test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
