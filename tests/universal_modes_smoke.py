from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data_loader import build_uploaded_data, enrich_dataframe
from core.generic_views import base_mode_capabilities
from core.universal_data import build_universal_datasets


def load(*names: str):
    official = ROOT / "sample_data" / "official"
    base = ROOT / "sample_data"
    result = []
    for name in names:
        path = (official / name) if (official / name).exists() else (base / name)
        result.append(build_uploaded_data(name, path.read_bytes()))
    return result


def available(files):
    datasets = build_universal_datasets(files)
    if not datasets:
        return [], []
    summaries = []
    for dataset in datasets:
        df, meta = enrich_dataframe(dataset.dataframe)
        caps = base_mode_capabilities(df, meta)
        summaries.append((dataset.name, {k for k, v in caps.items() if v["available"]}))
    return datasets, summaries


def main() -> int:
    cases = [
        ("Volatiles", load("CH4Air5Alc3_2026-05-18_13-28-47_Clean.csv"), {"Overview", "Time Series", "Warnings", "Playback"}),
        ("Particles", load("28June2025_Bike_Taveiro_Coord.txt"), {"Overview", "Time Series", "Map", "Warnings", "Playback"}),
        ("Gas", load("Akel_CH4_20250818.txt"), {"Overview", "Time Series", "Warnings", "Playback"}),
        ("GNSS Precision", load("GNSS_20260524_022750.mat"), {"Overview", "Time Series", "Map", "Vectors", "Playback"}),
        ("GNSS RTK", load("RoverData_300s_2m.mat"), {"Overview", "Time Series", "Map", "Playback"}),
        ("RTK MCOS", load("Data_1_Base_2025-11-25_14-40-22_Taveiro.mat"), {"Overview", "Raw Data"}),
        ("UV", load("medicoes_uv.txt"), {"Overview", "Time Series", "Warnings", "Playback"}),
        ("Radiation", load("radiation_timestamps.txt"), {"Overview", "Time Series", "Warnings", "Playback"}),
        ("Lightning", load("20260810_1400_lightning.txt", "20260811_1400_lightning.txt"), {"Overview", "Time Series", "Warnings", "Playback"}),
        ("Generic meteo", load("generic_meteo_gnss.csv"), {"Overview", "Time Series", "Map", "Vectors", "Playback"}),
    ]

    failures = []
    for label, files, required in cases:
        datasets, summaries = available(files)
        union = set().union(*(modes for _name, modes in summaries)) if summaries else set()
        missing = required - union
        print(f"{label:<18} datasets={len(datasets):>2} modes={', '.join(sorted(union))}")
        if missing:
            failures.append(f"{label}: missing {sorted(missing)}")

    if failures:
        print("\nFAIL")
        for failure in failures:
            print(" -", failure)
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
