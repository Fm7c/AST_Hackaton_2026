from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data_loader import build_uploaded_data
from plugins.gas_alcohol import parse_gas
from plugins.particles_sps import parse_particles
from plugins.volatiles_multisensor import parse_volatiles


def real_names(zf: zipfile.ZipFile) -> list[str]:
    names = []
    for info in zf.infolist():
        if info.is_dir() or info.filename.startswith("__MACOSX/"):
            continue
        if PurePosixPath(info.filename).name in {".DS_Store", "Thumbs.db"}:
            continue
        names.append(info.filename)
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the platform against the official Hackathon Data.zip")
    parser.add_argument("data_zip", type=Path)
    args = parser.parse_args()
    path = args.data_zip.expanduser().resolve()
    if not path.is_file():
        print(f"Data.zip not found: {path}", file=sys.stderr)
        return 2

    failures: list[str] = []
    counters: dict[str, int] = {}

    with zipfile.ZipFile(path) as zf:
        names = real_names(zf)

        volatiles = [n for n in names if n.startswith("Data/Volatiles/") and n.endswith(".csv")]
        for name in volatiles:
            file = build_uploaded_data(name, zf.read(name))
            df = parse_volatiles(file)
            if df.empty or len([c for c in df.columns if c.startswith("channel_")]) != 17:
                failures.append(f"Volatiles: {name}")
        counters["Volatiles"] = len(volatiles)

        particles = [n for n in names if n.startswith("Data/Particles/") and n.endswith(".txt")]
        for name in particles:
            file = build_uploaded_data(name, zf.read(name))
            if parse_particles(file).empty:
                failures.append(f"Particles: {name}")
        counters["Particles"] = len(particles)

        gas_files = [
            n for n in names
            if (n.startswith("Data/CH4/") or n.startswith("Data/Alcohol/")) and n.endswith(".txt")
        ]
        for name in gas_files:
            file = build_uploaded_data(name, zf.read(name))
            df, _ = parse_gas(file)
            if df.empty:
                failures.append(f"Gas: {name}")
        counters["CH4/Alcohol"] = len(gas_files)

        precision = [n for n in names if n.startswith("Data/GNSSprecision/") and n.endswith(".mat")]
        for name in precision:
            file = build_uploaded_data(name, zf.read(name))
            if file.dataframe is None or not {"xpos", "ypos", "zpos"}.issubset(file.dataframe.columns):
                failures.append(f"GNSSprecision: {name}")
        counters["GNSSprecision"] = len(precision)

        base_rover = [n for n in names if n.startswith("Data/GNSSresRTK/RTK_BaseRover/") and n.endswith(".mat")]
        for name in base_rover:
            file = build_uploaded_data(name, zf.read(name))
            if file.dataframe is None or not {"lat", "lon"}.issubset(file.dataframe.columns):
                failures.append(f"RTK_BaseRover: {name}")
        counters["RTK_BaseRover"] = len(base_rover)

        mcos_candidates = [
            n for n in names
            if n.startswith("Data/GNSSresRTK/") and "/RTK_BaseRover/" not in n and n.endswith(".mat")
        ]
        # Reading all ~18k files is unnecessary for a smoke test; verify one from each subgroup.
        seen_subgroups: set[str] = set()
        checked = 0
        for name in mcos_candidates:
            parts = PurePosixPath(name).parts
            subgroup = parts[2] if len(parts) >= 4 else "root"
            if subgroup in seen_subgroups:
                continue
            seen_subgroups.add(subgroup)
            file = build_uploaded_data(name, zf.read(name))
            if file.kind != "matlab_mcos" or not file.metadata.get("matlab_tables"):
                failures.append(f"RTK MCOS detection: {name}")
            checked += 1
        counters["RTK MCOS subgroups checked"] = checked

    print("Official Data.zip smoke test")
    for key, value in counters.items():
        print(f"  {key:<28} {value:>6}")

    if failures:
        print("\nFAILURES:")
        for item in failures:
            print(" -", item)
        return 1

    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
