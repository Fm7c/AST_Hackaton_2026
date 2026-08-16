from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
import plotly.express as px

from core.models import AnalysisResult, UploadedData


DATE_RE = re.compile(r"(20\d{2})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})")
STATION_RE = re.compile(r"kEsc_(\d+)_Station", re.I)
STREAM_RE = re.compile(r"Data_([12])", re.I)


def _parts(file: UploadedData) -> dict[str, Any]:
    path = PurePosixPath(file.name)
    parts = path.parts
    subgroup = "(root)"
    if len(parts) >= 4 and parts[0].lower() == "data" and parts[1].lower() == "gnssresrtk":
        subgroup = parts[2]

    base = path.name
    m_date = DATE_RE.search(base)
    timestamp = None
    if m_date:
        try:
            timestamp = datetime(*map(int, m_date.groups()))
        except ValueError:
            timestamp = None

    m_station = STATION_RE.search(base)
    m_stream = STREAM_RE.search(base)
    tables = [str(x) for x in file.metadata.get("matlab_tables", [])]
    table_families = sorted({re.sub(r"DataValid\d+$", "", t, flags=re.I).upper() for t in tables})

    role = "Unknown"
    low = base.lower()
    if "_base_" in low:
        role = "Base"
    elif "_rove_" in low or "rover" in low:
        role = "Rover"
    elif m_stream:
        role = f"Data {m_stream.group(1)}"

    return {
        "file": file.name,
        "name": base,
        "subgroup": subgroup,
        "timestamp": timestamp,
        "station": int(m_station.group(1)) if m_station else None,
        "stream": int(m_stream.group(1)) if m_stream else None,
        "role": role,
        "nmea_tables": ", ".join(table_families),
        "table_count": len(tables),
        "has_gga": "GGA" in table_families,
        "has_gll": "GLL" in table_families,
        "has_gsa": "GSA" in table_families,
        "has_gsv": "GSV" in table_families,
        "has_rmc": "RMC" in table_families,
        "has_vtg": "VTG" in table_families,
    }


class RTKMCOSCatalogPlugin:
    id = "rtk_mcos_catalog"
    name = "GNSS · RTK experiment catalog"
    description = (
        "Explores the large GNSSresRTK MATLAB-table collection using file metadata, experiment groups, "
        "station/stream identifiers and the NMEA table families embedded in each MAT file."
    )

    def confidence(self, files: list[UploadedData]) -> float:
        score = 0.0
        for file in files[:12]:
            low = file.name.lower()
            tables = file.metadata.get("matlab_tables") or []
            if file.kind == "matlab_mcos" and tables:
                score = max(score, 0.99)
            elif file.kind == "matlab_mcos" and "gnssresrtk" in low:
                score = max(score, 0.95)
            elif "gnssresrtk" in low and low.endswith(".mat"):
                # A standard MAT inside GNSSresRTK belongs to the same family, but
                # the point-by-point GNSS RTK plugin is normally the correct mode.
                score = max(score, 0.15)
        return score

    def run(self, files: list[UploadedData], options: dict[str, Any]) -> AnalysisResult:
        # Keep this catalog usable even if the user selects a standard RTK MAT by
        # mistake.  Previously that produced a hard ValueError after switching
        # datasets because Streamlit could retain the previous plugin selection.
        compatible = [
            f for f in files
            if Path(f.name).suffix.lower() == ".mat"
            and ("gnssresrtk" in f.name.lower() or f.metadata.get("matlab_tables"))
        ]
        if not compatible:
            raise ValueError("This analysis is intended for MAT files from the GNSSresRTK collection.")

        catalog = pd.DataFrame([_parts(file) for file in compatible])
        catalog["storage"] = [
            "MATLAB table/MCOS" if file.kind == "matlab_mcos" else "Standard MAT"
            for file in compatible
        ]
        figures: list[tuple[str, Any]] = []

        subgroup_counts = catalog.groupby("subgroup", as_index=False).size().rename(columns={"size": "files"})
        figures.append((
            "Experiências",
            px.bar(subgroup_counts, x="subgroup", y="files", title="Selected files by experiment group"),
        ))

        role_counts = catalog.groupby("role", as_index=False).size().rename(columns={"size": "files"})
        figures.append((
            "Base / Rover / streams",
            px.bar(role_counts, x="role", y="files", title="Distribuição por papel/stream identificado no nome"),
        ))

        family_cols = ["has_gga", "has_gll", "has_gsa", "has_gsv", "has_rmc", "has_vtg"]
        family = pd.DataFrame({
            "NMEA": [col.replace("has_", "").upper() for col in family_cols],
            "files": [int(catalog[col].sum()) for col in family_cols],
        })
        figures.append((
            "NMEA coverage",
            px.bar(family, x="NMEA", y="files", title="Famílias NMEA materializadas como MATLAB table"),
        ))

        dated = catalog.dropna(subset=["timestamp"]).copy()
        if not dated.empty:
            dated["date"] = pd.to_datetime(dated["timestamp"]).dt.date.astype(str)
            timeline = dated.groupby(["date", "subgroup"], as_index=False).size().rename(columns={"size": "files"})
            figures.append((
                "Timeline",
                px.bar(timeline, x="date", y="files", color="subgroup", title="Selected experiments over time"),
            ))

        mcos_count = int((catalog["storage"] == "MATLAB table/MCOS").sum())
        standard_count = int((catalog["storage"] == "Standard MAT").sum())

        metrics = {
            "Selected files": len(catalog),
            "MCOS files": mcos_count,
            "MAT standard": standard_count,
            "Groups": int(catalog["subgroup"].nunique()),
            "With GGA": int(catalog["has_gga"].sum()),
            "With GSV": int(catalog["has_gsv"].sum()),
            "With RMC": int(catalog["has_rmc"].sum()),
            "First experiment": dated["timestamp"].min() if not dated.empty else None,
            "Last experiment": dated["timestamp"].max() if not dated.empty else None,
        }

        return AnalysisResult(
            title=self.name,
            summary=(
                "Catalogue of the GNSSresRTK collection. For MATLAB table/MCOS files it compares structure and NMEA families; "
                "standard MAT files remain in the catalogue and are better analysed with GNSS · RTK / Rover for point-by-point data."
            ),
            metrics=metrics,
            tables={"Catalogue": catalog, "NMEA coverage": family},
            figures=figures,
            warnings=(
                [
                    "This analysis reports metadata and MATLAB-table structure. SciPy does not convert internal MCOS table rows into a DataFrame; "
                    "use RTK_BaseRover files for point-by-point trajectories and precision, which are fully read by GNSS · RTK / Rover."
                ]
                + ([
                    f"{standard_count} selected file(s) are standard MAT rather than MCOS. GNSS · RTK / Rover is the preferred analysis for those files."
                ] if standard_count else [])
            ),
            notes=[
                "GGA/GLL/RMC indicate NMEA families with position/status information, but values that the loader cannot materialise are not inferred.",
                "RTK_Right, RTK_Left, RTK_std, RTK_swap, RTK_Rain and date-based campaigns are preserved from the Data.zip folder structure.",
            ],
            raw={"catalog": catalog},
        )


PLUGIN = RTKMCOSCatalogPlugin()
