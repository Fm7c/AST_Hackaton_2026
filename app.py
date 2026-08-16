from __future__ import annotations

import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import base64
import hashlib
import io
import logging
import os
import math
import struct
import time
import wave
import zipfile
from pathlib import PurePosixPath
from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit.components.v1 import html as components_html

from core.data_loader import build_uploaded_data, dataframe_profile, enrich_dataframe
from core.database import DatabaseConfig, list_tables, load_dataframe, save_dataframe, test_connection
from core.generic_views import (
    bar_figure,
    base_mode_capabilities,
    classify_vector_group,
    default_timeseries,
    describe_numeric,
    detect_vector_groups,
    gauge_figure,
    magnetic_vector_map_figure,
    map_figure,
    meteorological_columns,
    meteorological_overlay_figure,
    multi_bar_figure,
    summary_metrics,
    threshold_figure,
    uv_index_gauge_figure,
    vector_snapshot_figure,
    vector_timeseries_figure,
)
from core.data_pack import category_for_member, find_default_pack, members_from_zip, subgroup_for_member, summarize_infos
from core.models import AnalysisResult, NormalizedDataset, UploadedData
from core.plugin_manager import legacy_script_statuses, ranked_plugins
from core.universal_data import build_universal_datasets, file_signature


SAMPLE_ROOT = APP_ROOT / "sample_data"
LOG_ROOT = APP_ROOT / "logs"
LOG_ROOT.mkdir(exist_ok=True)
LOGGER = logging.getLogger("ast_sensor_analytics")
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_ROOT / "sensor_analytics.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(handler)


def _log_exception(context: str, exc: BaseException) -> None:
    LOGGER.exception("%s: %s", context, exc)

SUPPORTED_FILE_TYPES = [
    "csv", "txt", "tsv", "dat", "log", "json", "xlsx", "xlsm", "xls", "mat", "zip"
]

st.set_page_config(
    page_title="AST Sensor Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 3rem;}
      [data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); padding: .7rem .9rem; border-radius: .65rem;}
      .ast-badge {display:inline-block;padding:.18rem .5rem;border-radius:.45rem;border:1px solid rgba(128,128,128,.35);font-size:.78rem;margin-right:.35rem;}
      .ast-hero {padding: .55rem 0 .75rem 0; margin-bottom:.4rem;}
      .ast-muted {opacity:.72;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False, max_entries=20)
def _cached_build_uploaded(name: str, raw: bytes) -> UploadedData:
    """Cache expensive parsing of MAT/Excel/text members across Streamlit reruns."""
    return build_uploaded_data(name, raw)


@st.cache_data(show_spinner=False, max_entries=12)
def _cached_local_zip_member(path_text: str, mtime_ns: int, file_size: int, member: str) -> UploadedData:
    # mtime + size invalidate the cache if Data.zip is replaced in place.
    del mtime_ns, file_size
    with zipfile.ZipFile(path_text) as zf:
        return build_uploaded_data(member, zf.read(member))


@st.cache_data(show_spinner=False, max_entries=4)
def _cached_local_zip_index(path_text: str, mtime_ns: int, file_size: int) -> tuple[tuple[str, int], ...]:
    """Cache the ZIP central directory so sidebar reruns do not rescan ~19k files."""
    del mtime_ns, file_size
    with zipfile.ZipFile(path_text) as zf:
        return tuple((info.filename, int(info.file_size)) for info in members_from_zip(zf))




def _sample_files(key: str) -> list[UploadedData]:
    mapping = {
        "Synthetic · Sensor Fusion + Weather + GNSS": ["generic_meteo_gnss.csv"],
        "Synthetic · Radiation": ["radiation_timestamps.txt"],
        "Synthetic · UV multi-sensor": ["medicoes_uv.txt"],
        "Synthetic · Lightning AS3935": ["20260810_1400_lightning.txt", "20260811_1400_lightning.txt"],
        "Official · Volatiles comparison": [
            "official/CH4Air5Alc3_2026-05-18_14-08-50_Acetone.csv",
            "official/CH4Air5Alc3_2026-05-18_11-08-38_Isoprop.csv",
            "official/CH4Air5Alc3_2026-05-18_13-28-47_Clean.csv",
        ],
        "Official · Particles + GNSS": ["official/28June2025_Bike_Taveiro_Coord.txt"],
        "Official · Gas CH4 + Alcohol": ["official/Akel_CH4_20250818.txt", "official/Akel_Alcohol3_20250810.txt"],
        "Official · GNSS precision": ["official/GNSS_20260524_022750.mat"],
        "Official · GNSS RTK": ["official/RoverData_300s_2m.mat"],
        "Official · GNSS RTK MCOS": ["official/Data_1_Base_2025-11-25_14-40-22_Taveiro.mat"],
        "Official · GNSS satellites": ["official/Observacoes.xlsx", "official/Navegacao.xlsx"],
    }
    files: list[UploadedData] = []
    for relative in mapping[key]:
        path = SAMPLE_ROOT / relative
        files.append(_cached_build_uploaded(path.name, path.read_bytes()))
    return files


def _zip_category(name: str) -> str:
    return category_for_member(name)


def _zip_subgroup(name: str, category: str | None = None) -> str:
    return subgroup_for_member(name)


def _archive_summary_from_records(records: list[tuple[str, int]]) -> pd.DataFrame:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for name, size in records:
        category = _zip_category(name)
        subgroup = _zip_subgroup(name, category)
        key = (category, subgroup)
        row = grouped.setdefault(key, {"category": category, "subgroup": subgroup, "files": 0, "size_mb": 0.0})
        row["files"] = int(row["files"]) + 1
        row["size_mb"] = float(row["size_mb"]) + float(size) / 1_000_000.0
    if not grouped:
        return pd.DataFrame(columns=["category", "subgroup", "files", "size_mb"])
    frame = pd.DataFrame(grouped.values())
    frame["size_mb"] = frame["size_mb"].round(3)
    return frame.sort_values(["category", "subgroup"], key=lambda col: col.astype(str).str.lower()).reset_index(drop=True)


def _remember_archive_catalog_records(records: list[tuple[str, int]], label: str) -> None:
    try:
        signature = (
            label,
            len(records),
            sum(size for _name, size in records),
            records[0][0] if records else "",
            records[-1][0] if records else "",
        )
        if st.session_state.get("archive_catalog_signature") == signature:
            return
        st.session_state["archive_catalog"] = _archive_summary_from_records(records)
        st.session_state["archive_catalog_label"] = label
        st.session_state["archive_catalog_signature"] = signature
    except Exception as exc:
        _log_exception("archive catalog", exc)


def _select_from_records(
    records: list[tuple[str, int]],
    archive_label: str,
    archive_index: int,
    key_prefix: str,
    member_loader: Callable[[str], UploadedData],
) -> list[UploadedData]:
    selected_files: list[UploadedData] = []
    if not records:
        st.sidebar.warning(f"{archive_label}: no supported files were found.")
        return []

    _remember_archive_catalog_records(records, archive_label)
    members = [name for name, _size in records]
    size_by_name = {name: size for name, size in records}

    categories = sorted({_zip_category(name) for name in members}, key=str.casefold)
    category = st.sidebar.selectbox(
        f"Category · {archive_label}",
        categories,
        key=f"{key_prefix}_category_{archive_index}",
    )
    category_members = [name for name in members if _zip_category(name) == category]

    subgroups = sorted({_zip_subgroup(name, category) for name in category_members}, key=str.casefold)
    if len(subgroups) > 1:
        subgroup = st.sidebar.selectbox(
            "Group",
            subgroups,
            key=f"{key_prefix}_subgroup_{archive_index}_{category}",
        )
        category_members = [name for name in category_members if _zip_subgroup(name, category) == subgroup]

    total_bytes = sum(size_by_name.get(name, 0) for name in category_members)
    st.sidebar.caption(f"{len(category_members):,} files · {total_bytes / 1_000_000:.1f} MB")

    search = st.sidebar.text_input(
        "Filter filenames",
        key=f"{key_prefix}_search_{archive_index}_{category}",
        placeholder="e.g. Acetone, File6, RoverData...",
    ).strip().lower()
    if search:
        category_members = [name for name in category_members if search in PurePosixPath(name).name.lower()]

    category_members = sorted(category_members, key=lambda name: PurePosixPath(name).name.lower())
    if len(category_members) > 1200 and not search:
        st.sidebar.caption("Filter by filename to narrow this large group.")
        visible = category_members[:1200]
    else:
        visible = category_members

    labels: dict[str, str] = {}
    for name in visible:
        basename = PurePosixPath(name).name
        label = basename
        if label in labels:
            label = f"{basename} · {PurePosixPath(name).parent}"
        labels[label] = name

    options = list(labels)
    # Do not auto-load the first file. On the official pack this used to trigger
    # a full scientific analysis during the very first Streamlit render.
    chosen = st.sidebar.multiselect(
        "Files",
        options,
        default=[],
        max_selections=25,
        key=f"{key_prefix}_files_{archive_index}_{category}",
    )
    if not chosen:
        st.sidebar.caption("Select one or more files to load.")
        return []

    for label in chosen:
        selected_files.append(member_loader(labels[label]))
    return selected_files


def _select_from_open_zip(
    zf: zipfile.ZipFile,
    archive_label: str,
    archive_index: int,
    key_prefix: str,
    member_loader: Callable[[str], UploadedData] | None = None,
) -> list[UploadedData]:
    records = [(info.filename, int(info.file_size)) for info in members_from_zip(zf)]
    loader = member_loader or (lambda member: _cached_build_uploaded(member, zf.read(member)))
    return _select_from_records(records, archive_label, archive_index, key_prefix, loader)


def _select_from_uploaded_zip(uploaded: Any, archive_index: int) -> list[UploadedData]:
    try:
        uploaded.seek(0)
        with zipfile.ZipFile(uploaded) as zf:
            return _select_from_open_zip(
                zf,
                uploaded.name,
                archive_index,
                key_prefix=f"uploadzip_{abs(hash(uploaded.name))}",
            )
    except zipfile.BadZipFile:
        st.sidebar.error(f"{uploaded.name}: invalid ZIP file.")
        return []
    except Exception as exc:
        st.sidebar.error(f"Could not read {uploaded.name}. See logs/sensor_analytics.log for details."); _log_exception("uploaded ZIP", exc)
        return []
    finally:
        try:
            uploaded.seek(0)
        except Exception:
            pass


def _select_from_local_zip(path: Path) -> list[UploadedData]:
    try:
        stat = path.stat()
        resolved = str(path.resolve())
        records = list(_cached_local_zip_index(resolved, int(stat.st_mtime_ns), int(stat.st_size)))
        return _select_from_records(
            records,
            path.name,
            0,
            key_prefix=f"localzip_{abs(hash(resolved))}",
            member_loader=lambda member: _cached_local_zip_member(
                resolved, int(stat.st_mtime_ns), int(stat.st_size), member
            ),
        )
    except zipfile.BadZipFile:
        st.sidebar.error("The selected file is not a valid ZIP archive.")
    except Exception as exc:
        st.sidebar.error("Could not open Data.zip. See logs/sensor_analytics.log for details.")
        _log_exception("local Data.zip", exc)
    return []


def _database_config_from_settings() -> DatabaseConfig | None:
    """Read server-side database settings from Streamlit secrets or environment."""
    values: dict[str, Any] = {}
    try:
        if "database" in st.secrets:
            section = st.secrets["database"]
            values = {
                "host": section.get("host", ""),
                "port": section.get("port", 5432),
                "database": section.get("database", ""),
                "username": section.get("username", section.get("user", "")),
                "password": section.get("password", ""),
                "sslmode": section.get("sslmode", "require"),
            }
    except Exception:
        values = {}

    if not values.get("host"):
        values = {
            "host": os.getenv("AST_DB_HOST", ""),
            "port": os.getenv("AST_DB_PORT", "5432"),
            "database": os.getenv("AST_DB_NAME", ""),
            "username": os.getenv("AST_DB_USER", ""),
            "password": os.getenv("AST_DB_PASSWORD", ""),
            "sslmode": os.getenv("AST_DB_SSLMODE", "require"),
        }
    if not all(str(values.get(k, "")).strip() for k in ("host", "database", "username", "password")):
        return None
    try:
        return DatabaseConfig(
            host=str(values["host"]).strip(),
            port=int(values.get("port", 5432)),
            database=str(values["database"]).strip(),
            username=str(values["username"]).strip(),
            password=str(values["password"]),
            sslmode=str(values.get("sslmode", "require")).strip() or "require",
        )
    except Exception:
        return None


def _current_database_config() -> DatabaseConfig | None:
    configured = _database_config_from_settings()
    if configured is not None:
        st.session_state["remote_database_config"] = configured
        return configured
    value = st.session_state.get("remote_database_config")
    return value if isinstance(value, DatabaseConfig) else None


def _database_connection_form() -> DatabaseConfig | None:
    configured = _database_config_from_settings()
    if configured is not None:
        st.session_state["remote_database_config"] = configured
        st.sidebar.caption("Remote PostgreSQL database configured")
        return configured

    with st.sidebar.expander("Database connection", expanded=True):
        st.caption("PostgreSQL · the server connects over TCP; the browser does not connect directly to the database.")
        host = st.text_input("Host / IP", key="remote_db_host", placeholder="db.example.com or 203.0.113.10")
        port = st.number_input("TCP port", min_value=1, max_value=65535, value=5432, step=1, key="remote_db_port")
        database = st.text_input("Database", key="remote_db_name")
        username = st.text_input("Username", key="remote_db_user")
        password = st.text_input("Password", type="password", key="remote_db_password")
        sslmode = st.selectbox(
            "TLS / SSL mode",
            ["require", "verify-full", "verify-ca", "prefer", "disable"],
            index=0,
            key="remote_db_sslmode",
        )
    if not all(str(v).strip() for v in (host, database, username, password)):
        st.sidebar.caption("Enter the database connection details to continue.")
        return None
    config = DatabaseConfig(
        host=str(host).strip(),
        port=int(port),
        database=str(database).strip(),
        username=str(username).strip(),
        password=str(password),
        sslmode=str(sslmode),
    )
    st.session_state["remote_database_config"] = config
    return config


@st.cache_data(show_spinner=False, ttl=30, max_entries=8)
def _cached_database_tables(config: DatabaseConfig) -> tuple[str, ...]:
    test_connection(config)
    return tuple(list_tables(config))


@st.cache_data(show_spinner=False, ttl=30, max_entries=6)
def _cached_database_dataframe(config: DatabaseConfig, table: str) -> pd.DataFrame:
    return load_dataframe(table, config)


def _load_from_database() -> list[UploadedData]:
    config = _database_connection_form()
    if config is None:
        return []
    try:
        with st.sidebar.status("Connecting to database...", expanded=False) as status:
            tables = list(_cached_database_tables(config))
            status.update(label="Database connected", state="complete")
    except Exception as exc:
        st.sidebar.error("Could not connect to PostgreSQL. Check host/IP, TCP port, credentials, firewall and TLS settings.")
        _log_exception("PostgreSQL connection", exc)
        return []
    if not tables:
        st.sidebar.info("The database is reachable but contains no readable tables yet.")
        return []
    table = st.sidebar.selectbox("Database table", tables, key="remote_database_table")
    try:
        df = _cached_database_dataframe(config, table)
    except Exception as exc:
        st.sidebar.error("Could not read the selected database table. See logs/sensor_analytics.log for details.")
        _log_exception("PostgreSQL table load", exc)
        return []
    return [UploadedData(name=f"Database/{table}", raw=b"", dataframe=df, kind="database")]


def _uploads() -> list[UploadedData]:
    default_pack = find_default_pack(APP_ROOT)
    source_options = ["Official dataset (Data.zip)", "Upload files", "Remote database", "Built-in examples"]
    source = st.sidebar.radio("Data source", source_options, index=0 if default_pack else 1)
    st.session_state["active_data_source"] = source

    if source == "Built-in examples":
        st.session_state.pop("archive_catalog", None)
        st.session_state.pop("archive_catalog_label", None)
        st.session_state.pop("archive_catalog_signature", None)
        demo = st.sidebar.selectbox(
            "Example dataset",
            [
                "Official · Volatiles comparison",
                "Official · Particles + GNSS",
                "Official · Gas CH4 + Alcohol",
                "Official · GNSS precision",
                "Official · GNSS RTK",
                "Official · GNSS RTK MCOS",
                "Official · GNSS satellites",
                "Synthetic · Sensor Fusion + Weather + GNSS",
                "Synthetic · Radiation",
                "Synthetic · UV multi-sensor",
                "Synthetic · Lightning AS3935",
            ],
        )
        return _sample_files(demo)

    if source == "Remote database":
        st.session_state.pop("archive_catalog", None)
        st.session_state.pop("archive_catalog_label", None)
        st.session_state.pop("archive_catalog_signature", None)
        return _load_from_database()

    if source == "Official dataset (Data.zip)":
        suggested = default_pack or (APP_ROOT / "Data.zip")
        path_text = st.sidebar.text_input(
            "Path to Data.zip",
            value=str(suggested),
            help="Place Data.zip next to the project or enter its full path.",
        ).strip()
        if not path_text:
            return []
        path = Path(path_text).expanduser()
        if not path.is_file():
            st.sidebar.warning("Data.zip was not found at this path.")
            return []
        st.sidebar.success("Data.zip found")
        return _select_from_local_zip(path)

    st.session_state.pop("archive_catalog", None)
    st.session_state.pop("archive_catalog_label", None)
    st.session_state.pop("archive_catalog_signature", None)
    uploaded = st.sidebar.file_uploader(
        "Data files",
        type=SUPPORTED_FILE_TYPES,
        accept_multiple_files=True,
        help="CSV, TXT, Excel, MAT and ZIP are supported. ZIP members are loaded only when selected.",
    )
    files: list[UploadedData] = []
    for index, item in enumerate(uploaded):
        if Path(item.name).suffix.lower() == ".zip":
            files.extend(_select_from_uploaded_zip(item, index))
        else:
            files.append(_cached_build_uploaded(item.name, item.getvalue()))
    return files


def _normalized_datasets(files: list[UploadedData]) -> list[NormalizedDataset]:
    """Build/cached universal dataframes for the current file selection."""
    signature = file_signature(files)
    cache = st.session_state.setdefault("universal_dataset_cache", {})
    if signature not in cache:
        # Keep the cache bounded because some GNSS datasets are large.
        if len(cache) >= 4:
            cache.clear()
        cache[signature] = build_universal_datasets(files)
    return cache[signature]


MODE_LABELS = {
    "Overview": "Overview",
    "Time Series": "Time Series",
    "Bars & Gauges": "Bars & Indicators",
    "Map": "Map",
    "Vectors": "Vectors",
    "Warnings": "Alerts",
    "Playback": "Playback",
    "Raw Data": "Raw Data",
}


def _capability_summary(df: pd.DataFrame, meta: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return base_mode_capabilities(df, meta)


def _stable_ui_key(*parts: Any) -> str:
    """Build a short deterministic Streamlit key from arbitrary context parts.

    Plotly elements need explicit unique keys because Streamlit renders all top-level
    tabs on every script run.  Without a key, two identical figures in different tabs
    can receive the same auto-generated element ID.
    """
    payload = "|".join(str(part) for part in parts).encode("utf-8", errors="replace")
    return hashlib.sha1(payload).hexdigest()[:20]


def _format_metric(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if hasattr(value, "strftime") and not isinstance(value, str):
        try:
            return value.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "—"
        return f"{value:,.4g}"
    return str(value)


def _metric_grid(metrics: dict[str, Any], per_row: int = 4) -> None:
    items = list(metrics.items())
    for start in range(0, len(items), per_row):
        row = items[start:start + per_row]
        cols = st.columns(len(row))
        for col, (label, value) in zip(cols, row):
            col.metric(label, _format_metric(value))


def _render_result(result: AnalysisResult, key_prefix: str = "result") -> None:
    st.subheader(result.title)
    if result.summary:
        st.write(result.summary)
    if result.metrics:
        _metric_grid(result.metrics)
    for warning in result.warnings:
        st.warning(warning)

    if result.figures:
        for fig_index, (title, fig) in enumerate(result.figures):
            with st.container(border=True):
                st.caption(title)
                if isinstance(fig, (bytes, bytearray)):
                    # Auto-discovered legacy scripts may leave Matplotlib figures.
                    # They are captured as PNG bytes so arbitrary scientific code
                    # can be shown without requiring a custom adapter.
                    st.image(bytes(fig), caption=title)
                elif hasattr(fig, "to_plotly_json"):
                    st.plotly_chart(
                        fig,
                        width="stretch",
                        config={"scrollZoom": True, "displaylogo": False},
                        key=f"plot_{_stable_ui_key(key_prefix, result.title, fig_index, title)}",
                    )
                elif hasattr(fig, "savefig"):
                    # Defensive fallback for a plugin that returns a Matplotlib
                    # figure directly rather than through the legacy adapter.
                    st.pyplot(fig, clear_figure=False)
                else:
                    st.caption(f"{title}: unsupported figure type {type(fig).__name__}")

    if result.tables:
        st.markdown("### Results")
        names = list(result.tables)
        if len(names) == 1:
            selected_name = names[0]
        elif hasattr(st, "segmented_control"):
            selected_name = st.segmented_control(
                "Table", names, default=names[0], key=f"table_{_stable_ui_key(key_prefix, result.title)}",
                label_visibility="collapsed",
            ) or names[0]
        else:
            selected_name = st.selectbox("Table", names, key=f"table_{_stable_ui_key(key_prefix, result.title)}")
        table = result.tables[selected_name]
        preview = table if len(table) <= 50000 else table.head(50000)
        st.dataframe(preview, width="stretch", hide_index=True)
        if len(table) > len(preview):
            st.caption(f"Showing 50,000 of {len(table):,} rows to keep the interface responsive.")
        try:
            csv_payload = table.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Export CSV", csv_payload,
                file_name=f"{selected_name.lower().replace(' ', '_').replace('/', '_')}.csv",
                mime="text/csv",
                key=f"download_{_stable_ui_key(key_prefix, result.title, selected_name, 'csv')}",
            )
        except Exception as exc:
            _log_exception("result CSV export", exc)

    if result.downloads:
        st.markdown("### Downloads")
        for label, (payload, mime) in result.downloads.items():
            suffix = ".xlsx" if "spreadsheet" in mime else ".bin"
            st.download_button(
                label, payload,
                file_name=f"{label.lower().replace(' ', '_')}{suffix}",
                mime=mime,
                key=f"download_{_stable_ui_key(key_prefix, result.title, label, mime)}",
            )

    if result.notes:
        with st.expander("Method notes"):
            for note in result.notes:
                st.markdown(f"- {note}")


def _is_uv_index_column(name: str) -> bool:
    text = str(name).lower().replace(" ", "_")
    return text in {"uvi", "uv_index", "indice_uv", "indice_uvi"} or "uv_index" in text


def _uv_risk_label(value: float) -> str:
    if value < 3:
        return "Low"
    if value < 6:
        return "Moderate"
    if value < 8:
        return "High"
    if value < 11:
        return "Very high"
    return "Extreme"


def _warning_beep_html() -> str:
    """Return a tiny generated WAV as an autoplay data URI (no external asset)."""
    sample_rate = 8000
    duration = 0.18
    frequency = 880.0
    frames = int(sample_rate * duration)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for i in range(frames):
            envelope = max(0.0, 1.0 - i / frames)
            value = int(14000 * envelope * math.sin(2 * math.pi * frequency * i / sample_rate))
            wav.writeframesraw(struct.pack("<h", value))
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f'<audio autoplay aria-label="warning tone"><source src="data:audio/wav;base64,{encoded}" type="audio/wav"></audio>'


def _play_warning_beep() -> None:
    components_html(_warning_beep_html(), height=0, width=0)


def _render_overview(df: pd.DataFrame, meta: dict[str, Any]) -> None:
    _metric_grid(summary_metrics(df, meta))
    tags = []
    if meta.get("time_column"):
        tags.append("Time series")
    if meta.get("latitude_column") and meta.get("longitude_column"):
        tags.append("Geographic")
    if meta.get("numeric_columns"):
        tags.append(f"{len(meta['numeric_columns'])} numeric fields")
    if tags:
        st.markdown(" ".join(f'<span class="ast-badge">{x}</span>' for x in tags), unsafe_allow_html=True)

    st.markdown("### Descriptive statistics")
    stats = describe_numeric(df, list(meta.get("numeric_columns", [])))
    if stats.empty:
        st.caption("No numeric fields are available for descriptive statistics.")
    else:
        st.dataframe(stats, width="stretch", hide_index=True)

    st.markdown("### Data quality")
    missing = (
        df.isna().sum().sort_values(ascending=False)
        .rename("missing_values").rename_axis("column").reset_index()
    )
    st.dataframe(missing, width="stretch", hide_index=True)


def _render_timeseries(df: pd.DataFrame, meta: dict[str, Any], prefix: str = "") -> None:
    numeric = list(meta.get("numeric_columns", []))
    if not numeric:
        st.caption("No numeric variables are available for this view.")
        return
    columns = st.multiselect("Variables", numeric, default=numeric[:min(4, len(numeric))], key=f"{prefix}ts_cols", max_selections=8)
    if not columns:
        return
    range_options = ["Auto", "Full range", "Manual"]
    if any(_is_uv_index_column(c) for c in columns):
        range_options.insert(1, "UV 0–12")
    range_mode = st.selectbox("Y-axis range", range_options, key=f"{prefix}ts_range_mode")
    fig = default_timeseries(df, meta, columns)
    if fig is None:
        return
    if range_mode == "UV 0–12":
        fig.update_yaxes(range=[0, 12])
    elif range_mode == "Full range":
        merged = pd.concat([pd.to_numeric(df[c], errors="coerce") for c in columns], ignore_index=True).dropna()
        if not merged.empty:
            lo, hi = float(merged.min()), float(merged.max())
            pad = max((hi-lo)*0.04, abs(hi)*0.01, 1e-9)
            fig.update_yaxes(range=[lo-pad, hi+pad])
    elif range_mode == "Manual":
        merged = pd.concat([pd.to_numeric(df[c], errors="coerce") for c in columns], ignore_index=True).dropna()
        if not merged.empty:
            lo0, hi0 = float(merged.min()), float(merged.max())
            if lo0 == hi0:
                hi0 = lo0 + 1.0
            c1, c2 = st.columns(2)
            lo = c1.number_input("Y minimum", value=lo0, key=f"{prefix}ts_ymin")
            hi = c2.number_input("Y maximum", value=hi0, key=f"{prefix}ts_ymax")
            if float(lo) < float(hi):
                fig.update_yaxes(range=[float(lo), float(hi)])
            else:
                st.warning("Y minimum must be lower than Y maximum.")
    st.plotly_chart(fig, width="stretch", config={"scrollZoom": True, "displaylogo": False}, key=f"plot_{_stable_ui_key(prefix, 'base_timeseries', tuple(columns), range_mode)}")


def _render_bars_gauges(df: pd.DataFrame, meta: dict[str, Any], prefix: str = "") -> None:
    numeric = list(meta.get("numeric_columns", []))
    if not numeric:
        st.caption("No numeric variables are available for this view.")
        return
    options = ["Single variable", "Clustered", "Stacked"]
    chart_mode = st.segmented_control("Bar layout", options, default=options[0], key=f"{prefix}bar_mode", label_visibility="collapsed") if hasattr(st, "segmented_control") else st.radio("Bar layout", options, horizontal=True, key=f"{prefix}bar_mode")
    chart_mode = chart_mode or options[0]
    aggregation = st.selectbox("Segment statistic", ["mean", "median", "max", "min"], key=f"{prefix}bar_aggregation")
    gauge_variable = None
    if chart_mode == "Single variable":
        gauge_variable = st.selectbox("Variable", numeric, key=f"{prefix}bar_variable")
        fig = bar_figure(df, gauge_variable, aggregation=aggregation)
    else:
        selected = st.multiselect("Variables", numeric, default=numeric[:min(4, len(numeric))], max_selections=8, key=f"{prefix}bar_variables_multi")
        if len(selected) < 2:
            st.caption("Select at least two variables.")
            fig = None
        else:
            fig = multi_bar_figure(df, selected, aggregation=aggregation, barmode="stack" if chart_mode == "Stacked" else "group")
            gauge_variable = selected[0]
    if fig is not None:
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False}, key=f"plot_{_stable_ui_key(prefix, 'bars', chart_mode, aggregation, gauge_variable)}")
    if not gauge_variable:
        return
    values = pd.to_numeric(df[gauge_variable], errors="coerce").dropna()
    if values.empty:
        return
    # Summary indicators represent the whole selected measurement, not only
    # the final sample.  The arithmetic mean is therefore the value shown in
    # the gauge and metric cards.
    st.markdown("### Mean value")
    mean_value = float(values.mean())
    if _is_uv_index_column(gauge_variable):
        c1, c2 = st.columns([2, 1])
        with c1:
            st.plotly_chart(uv_index_gauge_figure(mean_value, f"Average UV Index · {gauge_variable}"), width="stretch", key=f"plot_{_stable_ui_key(prefix, 'uv_gauge', gauge_variable)}")
        c2.metric("Mean UVI", f"{mean_value:.2f}")
        c2.metric("Level", _uv_risk_label(mean_value))
        if mean_value > 12:
            c2.caption("The fixed UV scale ends at 12; the measurement average is higher.")
    else:
        low, high = float(values.min()), float(values.max())
        if low == high:
            high = low + 1.0
        st.plotly_chart(gauge_figure(mean_value, f"Mean · {gauge_variable}", low, high), width="stretch", key=f"plot_{_stable_ui_key(prefix, 'gauge', gauge_variable)}")


def _render_map(df: pd.DataFrame, meta: dict[str, Any], prefix: str = "") -> None:
    if not (meta.get("latitude_column") and meta.get("longitude_column")):
        return
    numeric = list(meta.get("numeric_columns", []))
    coordinate_cols = {meta.get("latitude_column"), meta.get("longitude_column"), meta.get("altitude_column")}
    color_options = [c for c in numeric if c not in coordinate_cols]
    meteo = meteorological_columns(list(df.columns))
    vector_groups = detect_vector_groups(numeric)
    magnetic_groups = {name: cols for name, cols in vector_groups.items() if classify_vector_group(name, cols) == "magnetometer"}
    map_modes = ["Measurements"]
    if meteo:
        map_modes.append("Weather")
    if magnetic_groups:
        map_modes.append("Magnetic field")
    if len(map_modes) > 1:
        map_mode = st.segmented_control("Map layer", map_modes, default="Weather" if "Weather" in map_modes else map_modes[0], key=f"{prefix}map_mode", label_visibility="collapsed") if hasattr(st, "segmented_control") else st.radio("Map layer", map_modes, horizontal=True, key=f"{prefix}map_mode")
        map_mode = map_mode or map_modes[0]
    else:
        map_mode = map_modes[0]
    if map_mode == "Weather":
        available_layers = list(meteo)
        layers = st.multiselect(
            "Weather layers",
            available_layers,
            default=available_layers,
            key=f"{prefix}meteo_layers",
            help="Temperature, humidity and pressure are drawn together at their real GNSS coordinates. Use the legend to hide/show individual layers.",
        )
        if not layers:
            st.caption("Select at least one weather layer.")
            return
        means: dict[str, float] = {}
        for layer in layers:
            values = pd.to_numeric(df[meteo[layer]], errors="coerce").dropna()
            if not values.empty:
                means[f"Mean {layer.lower()}"] = float(values.mean())
        if means:
            _metric_grid(means)
        fig = meteorological_overlay_figure(df, meta, meteo, layers)
        if len(layers) > 1:
            st.caption("Meteorological layers are overlaid on the same geographic samples; marker sizes differ so simultaneous layers remain visible.")
    elif map_mode == "Magnetic field":
        group_name = st.selectbox("Magnetic vector", list(magnetic_groups), key=f"{prefix}mag_map_group")
        fig = magnetic_vector_map_figure(df, meta, magnetic_groups[group_name])
        st.caption("Arrow length is normalised for readability; the measured magnitude remains available on hover.")
    else:
        value = st.selectbox("Colour by", ["None"] + color_options, key=f"{prefix}map_color")
        fig = map_figure(df, meta, None if value == "None" else value)
    if fig is not None:
        st.plotly_chart(fig, width="stretch", config={"scrollZoom": True, "displaylogo": False}, key=f"plot_{_stable_ui_key(prefix, 'map', map_mode)}")


def _render_vectors(df: pd.DataFrame, meta: dict[str, Any], prefix: str = "") -> None:
    numeric = list(meta.get("numeric_columns", []))
    groups = detect_vector_groups(numeric)
    if not groups:
        return
    kinds = {name: classify_vector_group(name, cols) for name, cols in groups.items()}
    group_name = st.selectbox("Vector", list(groups), key=f"{prefix}vector_group")
    columns = groups[group_name]
    kind = kinds[group_name]
    friendly = {"accelerometer": "Accelerometer", "gyroscope": "Gyroscope", "magnetometer": "Magnetometer", "ecef": "ECEF", "enu": "ENU", "vector": group_name}.get(kind, group_name)
    fig = vector_timeseries_figure(df, meta, columns, friendly)
    st.plotly_chart(fig, width="stretch", config={"scrollZoom": True, "displaylogo": False}, key=f"plot_{_stable_ui_key(prefix, 'vector_timeseries', group_name)}")
    valid = df[list(columns)].apply(pd.to_numeric, errors="coerce").dropna()
    if valid.empty:
        st.caption("No complete XYZ samples are available.")
        return
    magnitude_series = np.sqrt((valid ** 2).sum(axis=1))
    _metric_grid({"Mean |v|": float(magnitude_series.mean()), "Maximum |v|": float(magnitude_series.max()), "Minimum |v|": float(magnitude_series.min()), "Valid samples": len(valid)})
    index = st.slider("3D vector sample", 0, len(valid)-1, len(valid)-1, key=f"{prefix}vector_index")
    row = valid.iloc[index]
    x, y, z = (float(row[c]) for c in columns)
    magnitude = float(np.sqrt(x*x + y*y + z*z))
    _metric_grid({"X": x, "Y": y, "Z": z, "Magnitude": magnitude})
    st.plotly_chart(vector_snapshot_figure((x, y, z), f"Vector snapshot · {friendly}"), width="stretch", config={"displaylogo": False}, key=f"plot_{_stable_ui_key(prefix, 'vector_snapshot', group_name)}")
    if kind == "magnetometer" and meta.get("latitude_column") and meta.get("longitude_column"):
        map_fig = magnetic_vector_map_figure(df, meta, columns)
        if map_fig is not None:
            st.markdown("### Magnetic field on map")
            st.plotly_chart(map_fig, width="stretch", config={"scrollZoom": True, "displaylogo": False}, key=f"plot_{_stable_ui_key(prefix, 'magnetic_vector_map', group_name)}")


def _render_warnings(df: pd.DataFrame, meta: dict[str, Any], prefix: str = "") -> None:
    # Alerts should monitor sensor measurements, not the time axis itself.
    time_column = meta.get("time_column")
    numeric = [c for c in meta.get("numeric_columns", []) if c != time_column]
    if not numeric:
        st.caption("No numeric measurement variables are available for alerts.")
        return
    variable = st.selectbox("Monitored variable", numeric, key=f"{prefix}warning_variable")
    values = pd.to_numeric(df[variable], errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return
    q05, q95 = float(valid.quantile(0.05)), float(valid.quantile(0.95))
    low_default = float(valid.min()) if q05 == q95 else q05
    high_default = float(valid.max()) if q05 == q95 else q95
    c1, c2 = st.columns(2)
    use_low = c1.checkbox("Lower threshold", value=False, key=f"{prefix}warning_use_low")
    use_high = c2.checkbox("Upper threshold", value=True, key=f"{prefix}warning_use_high")
    low = c1.number_input("Minimum value", value=low_default, key=f"{prefix}warning_low") if use_low else None
    high = c2.number_input("Maximum value", value=high_default, key=f"{prefix}warning_high") if use_high else None
    if low is not None and high is not None and float(low) >= float(high):
        st.warning("Lower threshold must be below upper threshold.")
        return
    a1, a2, a3, a4 = st.columns([1,1,1,1])
    use_color = a1.checkbox("Colour", value=True, key=f"{prefix}warning_color")
    use_text = a2.checkbox("Message", value=True, key=f"{prefix}warning_text")
    use_sound = a3.checkbox("Sound", value=False, key=f"{prefix}warning_sound")
    if a4.button("Test sound", key=f"{prefix}warning_test_sound"):
        _play_warning_beep()
    breach = pd.Series(False, index=df.index)
    if low is not None:
        breach |= values < float(low)
    if high is not None:
        breach |= values > float(high)
    current = float(valid.iloc[-1])
    current_alarm = (low is not None and current < float(low)) or (high is not None and current > float(high))
    _metric_grid({"Latest sample": current, "Minimum": float(valid.min()), "Maximum": float(valid.max()), "Out-of-range samples": int(breach.sum())})
    if current_alarm:
        message, level = f"{variable}: latest sample is outside the configured range.", "alarm"
    elif breach.any():
        message, level = f"{int(breach.sum())} historical samples are outside the configured range. Latest sample is normal.", "history"
    else:
        message, level = "All samples are within the configured range.", "normal"
    if use_text:
        if use_color and level == "alarm": st.error(message)
        elif use_color and level == "history": st.warning(message)
        elif use_color: st.success(message)
        else: st.write(message)
    elif use_color:
        if level == "alarm": st.error("ALERT")
        elif level == "history": st.warning("Historical threshold breaches")
        else: st.success("NORMAL")
    alarm_state_key = f"{prefix}warning_last_sound"
    if current_alarm and use_sound:
        alarm_signature = (variable, None if low is None else float(low), None if high is None else float(high), round(current, 8))
        if st.session_state.get(alarm_state_key) != alarm_signature:
            _play_warning_beep(); st.session_state[alarm_state_key] = alarm_signature
    elif not current_alarm:
        st.session_state.pop(alarm_state_key, None)
    fig = threshold_figure(df, meta, variable, low, high)
    st.plotly_chart(fig, width="stretch", config={"scrollZoom": True, "displaylogo": False}, key=f"plot_{_stable_ui_key(prefix, 'warnings', variable, low, high)}")
    if breach.any():
        columns = []
        if meta.get("time_column") and meta["time_column"] in df.columns:
            columns.append(meta["time_column"])
        columns.append(variable)
        # Preserve order while guaranteeing unique column names. This also makes
        # the table safe if a stale widget state ever points at the time column.
        columns = list(dict.fromkeys(columns))
        warning_rows = df.loc[breach, columns].copy().head(5000)
        st.dataframe(warning_rows, width="stretch", hide_index=True)


def _playback_frame(
    df: pd.DataFrame,
    meta: dict[str, Any],
    index: int,
    selected: list[str] | None = None,
    key_prefix: str = "playback",
) -> None:
    index = max(0, min(index, len(df) - 1))
    visible = df.iloc[: index + 1]
    current = df.iloc[index]
    numeric = list(meta.get("numeric_columns", []))
    selected = [c for c in (selected or numeric[:4]) if c in numeric][:6]

    metrics = {c: current[c] for c in selected if c in current.index}
    if meta.get("time_column"):
        metrics = {"Time": current[meta["time_column"]], **metrics}
    _metric_grid(metrics)

    fig = default_timeseries(visible, meta, selected)
    if fig is not None:
        # Keep the same frontend element and the same view while playback advances.
        # This avoids rebuilding the whole chart as a brand-new Streamlit element.
        fig.update_layout(uirevision=key_prefix)
        st.plotly_chart(
            fig,
            width="stretch",
            config={"scrollZoom": True, "displaylogo": False},
            key=f"plot_{_stable_ui_key(key_prefix, 'series')}",
        )

    if meta.get("latitude_column") and meta.get("longitude_column"):
        m = map_figure(visible, meta, selected[0] if selected else None)
        if m is not None:
            # Highlight the current playback position with a larger red marker.
            lat_col = meta.get("latitude_column")
            lon_col = meta.get("longitude_column")
            try:
                current_lat = float(current[lat_col]) if lat_col else None
                current_lon = float(current[lon_col]) if lon_col else None
            except (TypeError, ValueError):
                current_lat = current_lon = None
            if current_lat is not None and current_lon is not None:
                import plotly.graph_objects as go
                m.add_trace(
                    go.Scattermapbox(
                        lat=[current_lat],
                        lon=[current_lon],
                        mode="markers",
                        marker={"size": 18, "color": "#E53935"},
                        name="Current position",
                        hovertemplate="Current position<extra></extra>",
                    )
                )
            m.update_layout(uirevision=key_prefix)
            st.plotly_chart(
                m,
                width="stretch",
                key=f"plot_{_stable_ui_key(key_prefix, 'map')}",
            )


def _render_playback(df: pd.DataFrame, meta: dict[str, Any], dataset_key: str) -> None:
    """Interactive playback without full-page reruns.

    The old implementation used ``time.sleep()`` followed by ``st.rerun()``.
    Because Streamlit eagerly executes every top-level tab, that rebuilt the whole
    application many times per second, which caused the page to flash and also left
    the timeline slider stuck on its own widget state.  Playback now uses a Streamlit
    fragment: only the frame area reruns on a timer, while the rest of the UI stays
    mounted.
    """
    if len(df) < 2:
        st.info("Playback requires at least two samples.")
        return

    safe_key = _stable_ui_key("playback", dataset_key)
    pos_key = f"playback_position_{safe_key}"
    run_key = f"playback_running_{safe_key}"
    dir_key = f"playback_direction_{safe_key}"
    tick_key = f"playback_last_tick_{safe_key}"

    st.session_state.setdefault(pos_key, 0)
    st.session_state.setdefault(run_key, False)
    st.session_state.setdefault(dir_key, 1)
    st.session_state.setdefault(tick_key, time.monotonic())
    st.session_state[pos_key] = max(0, min(int(st.session_state[pos_key]), len(df) - 1))

    # Use callbacks for transport controls. Streamlit executes callbacks before
    # rendering the rerun, so the Play/Pause label always reflects the new state
    # immediately after the click.
    def _go_start() -> None:
        st.session_state[pos_key] = 0
        st.session_state[run_key] = False
        st.session_state[tick_key] = time.monotonic()

    def _go_previous() -> None:
        st.session_state[pos_key] = max(0, int(st.session_state[pos_key]) - 1)
        st.session_state[run_key] = False
        st.session_state[tick_key] = time.monotonic()

    def _toggle_playback() -> None:
        starting = not bool(st.session_state[run_key])
        if starting and int(st.session_state[dir_key]) > 0 and int(st.session_state[pos_key]) >= len(df) - 1:
            st.session_state[pos_key] = 0
        elif starting and int(st.session_state[dir_key]) < 0 and int(st.session_state[pos_key]) <= 0:
            st.session_state[pos_key] = len(df) - 1
        st.session_state[run_key] = starting
        st.session_state[tick_key] = time.monotonic()

    def _go_next() -> None:
        st.session_state[pos_key] = min(len(df) - 1, int(st.session_state[pos_key]) + 1)
        st.session_state[run_key] = False
        st.session_state[tick_key] = time.monotonic()

    def _go_end() -> None:
        st.session_state[pos_key] = len(df) - 1
        st.session_state[run_key] = False
        st.session_state[tick_key] = time.monotonic()

    c0, c1, c2, c3, c4, c5 = st.columns([1, 1, 1.25, 1, 1, 4])
    c0.button("⏮", help="Start", key=f"pb_start_{safe_key}", on_click=_go_start)
    c1.button("◀", help="Previous", key=f"pb_prev_{safe_key}", on_click=_go_previous)
    c2.button(
        "Pause" if st.session_state[run_key] else "Play",
        key=f"pb_play_{safe_key}",
        on_click=_toggle_playback,
    )
    c3.button("▶", help="Next", key=f"pb_next_{safe_key}", on_click=_go_next)
    c4.button("⏭", help="End", key=f"pb_end_{safe_key}", on_click=_go_end)

    direction_text = c5.radio(
        "Direction",
        ["Forward", "Reverse"],
        horizontal=True,
        index=0 if st.session_state[dir_key] >= 0 else 1,
        key=f"pb_dir_radio_{safe_key}",
    )
    st.session_state[dir_key] = 1 if direction_text == "Forward" else -1
    speed = st.slider("Playback speed", 1, 20, 5, key=f"pb_speed_{safe_key}")

    numeric = list(meta.get("numeric_columns", []))
    excluded = {
        meta.get("latitude_column"),
        meta.get("longitude_column"),
        meta.get("altitude_column"),
        meta.get("time_column"),
    }
    preferred = [c for c in numeric if c not in excluded and c not in {"sample", "evento", "contagem_acumulada"}]
    defaults = (preferred or numeric)[:4]
    selected_metrics = st.multiselect(
        "Displayed variables",
        numeric,
        default=defaults,
        key=f"pb_metrics_{safe_key}",
        max_selections=6,
    )

    # Only the fragment below refreshes during autoplay.  The full Streamlit page,
    # sidebar, tabs and controls no longer rerun for every frame.
    run_every = max(0.08, 1.0 / float(speed)) if st.session_state[run_key] else None

    @st.fragment(run_every=run_every)
    def _playback_fragment() -> None:
        now = time.monotonic()
        interval = max(0.08, 1.0 / float(speed))

        if st.session_state[run_key] and now - float(st.session_state[tick_key]) >= interval * 0.8:
            next_index = int(st.session_state[pos_key]) + int(st.session_state[dir_key])
            if next_index < 0 or next_index >= len(df):
                st.session_state[run_key] = False
                st.session_state[tick_key] = now
                # Rebuild once with run_every=None so the timer stops cleanly.
                st.rerun()
            else:
                st.session_state[pos_key] = next_index
                st.session_state[tick_key] = now

        # The slider itself is now the single source of truth for the current frame.
        # Updating its session-state value before widget creation lets autoplay move it.
        index = st.slider(
            "Timeline",
            min_value=0,
            max_value=len(df) - 1,
            key=pos_key,
        )
        progress = (index + 1) / len(df)
        st.progress(progress, text=f"Sample {index + 1:,} of {len(df):,}")
        _playback_frame(df, meta, index, selected_metrics, key_prefix=f"playback_{safe_key}")

    _playback_fragment()


def _render_generic(files: list[UploadedData]) -> None:
    datasets = _normalized_datasets(files)
    if not datasets:
        st.caption("No tabular dataset could be prepared from the current selection.")
        return
    labels = {f"{d.name} · {len(d.dataframe):,} rows": d for d in datasets}
    selected_label = st.selectbox("Dataset", list(labels), key="base_normalized_dataset")
    source = labels[selected_label]
    df, meta = enrich_dataframe(source.dataframe)
    capabilities = _capability_summary(df, meta)
    available_modes = [name for name, info in capabilities.items() if info["available"] and name not in {"Overview", "Raw Data"}]
    if not available_modes:
        st.caption("No compatible visualisations are available for this dataset.")
        return
    prefix = str(abs(hash(source.id)))
    mode_labels = [MODE_LABELS[name] for name in available_modes]
    if hasattr(st, "segmented_control"):
        chosen_label = st.segmented_control("View", mode_labels, default=mode_labels[0], selection_mode="single", key=f"base_mode_{prefix}", label_visibility="collapsed")
    else:
        chosen_label = st.radio("View", mode_labels, horizontal=True, key=f"base_mode_{prefix}", label_visibility="collapsed")
    if chosen_label not in mode_labels:
        chosen_label = mode_labels[0]
    mode = next(name for name in available_modes if MODE_LABELS[name] == chosen_label)
    if mode == "Time Series": _render_timeseries(df, meta, prefix=prefix)
    elif mode == "Bars & Gauges": _render_bars_gauges(df, meta, prefix=prefix)
    elif mode == "Map": _render_map(df, meta, prefix=prefix)
    elif mode == "Vectors": _render_vectors(df, meta, prefix=prefix)
    elif mode == "Warnings": _render_warnings(df, meta, prefix=prefix)
    elif mode == "Playback": _render_playback(df, meta, source.id)


def _special_options(plugin_id: str) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if plugin_id == "radiation_events":
        cols = st.columns(4)
        options["time_unit"] = cols[0].selectbox("Unidade dos timestamps", ["min", "s", "h"], index=0)
        options["chain_limit_s"] = cols[1].number_input("Limiar cadeia (s)", min_value=0.001, value=2.0, step=0.5)
        options["window_min"] = cols[2].number_input("Janela CPM (min)", min_value=0.01, value=70.0, step=1.0)
        options["hist_bin_min"] = cols[3].number_input("Bin histograma (min)", min_value=0.001, value=0.05, step=0.01)
    elif plugin_id == "uv_multisensor":
        options["gap_seconds"] = st.number_input("Maximum sample gap (s)", min_value=1, value=90, step=10)
    elif plugin_id in {"particles_sps", "gas_alcohol"}:
        options["sample_interval_s"] = st.number_input("Assumed sample interval (s)", min_value=0.001, value=1.0, step=0.5)
    elif plugin_id == "volatiles_multisensor":
        options["top_channels"] = st.slider("Highlighted channels", 3, 12, 6)
    elif plugin_id in {"gnss_precision", "gnss_rtk"}:
        options["max_plot_points"] = st.slider("Maximum plot points", 1000, 20000, 8000, step=1000)
    elif plugin_id == "gnss_satellites":
        cols = st.columns(2)
        options["max_plot_points"] = cols[0].slider("Maximum plot points", 1000, 20000, 8000, step=1000)
        options["max_rows_per_sheet"] = cols[1].slider("Rows per constellation", 2000, 35000, 15000, step=1000)
    return options


def _render_special(files: list[UploadedData]) -> None:
    if not files:
        st.caption("Select data to enable dedicated analyses.")
        return
    ranked = ranked_plugins(files)
    compatible = [item for item in ranked if item.confidence >= 0.30]
    if not compatible:
        st.caption("No dedicated analysis is available for this format. Generic visualisations may still be used.")
        return
    labels = [item.plugin.name for item in compatible]
    by_label = {item.plugin.name: item for item in compatible}
    file_sig = tuple((f.name, f.kind, tuple(map(str, f.metadata.get("matlab_tables", []) or []))) for f in files)
    choice = st.selectbox("Analysis", labels, key=f"special_analysis_{abs(hash(file_sig))}")
    selected = by_label[choice]
    st.caption(selected.plugin.description)
    options = _special_options(selected.plugin.id)
    if st.button("Run analysis", type="primary", width="stretch"):
        with st.spinner("Running analysis..."):
            try:
                result = selected.plugin.run(files, options)
                st.session_state["last_special_result"] = result
                st.session_state["last_special_plugin"] = selected.plugin.id
                st.session_state["last_special_files"] = tuple(f.name for f in files)
            except ValueError as exc:
                LOGGER.info("Analysis input rejected by %s: %s", selected.plugin.id, exc)
                st.warning(str(exc))
            except Exception as exc:
                _log_exception(f"analysis {selected.plugin.id}", exc)
                st.error("The analysis could not be completed. See logs/sensor_analytics.log for details.")
    result = st.session_state.get("last_special_result")
    if result is not None and st.session_state.get("last_special_plugin") == selected.plugin.id and st.session_state.get("last_special_files") == tuple(f.name for f in files):
        _render_result(result, key_prefix=f"special_{selected.plugin.id}")

    # Do not silently ignore a newly copied legacy script. If its sensor or entry
    # point cannot be determined, keep the normal UI clean and expose the reason
    # only in a small troubleshooting expander.
    unsupported = [status for status in legacy_script_statuses() if not status.supported]
    if unsupported:
        with st.expander(f"Legacy scripts needing configuration ({len(unsupported)})"):
            for status in unsupported:
                sensor = status.sensor or "unknown sensor"
                st.write(f"**{status.filename}** · {sensor}")
                st.caption(status.reason)
            st.caption("A sidecar JSON file can specify sensor, function and input without changing the original Python script.")


def _best_universal_dataset(datasets: list[NormalizedDataset]) -> tuple[NormalizedDataset | None, pd.DataFrame | None, dict[str, Any] | None, dict[str, dict[str, Any]] | None]:
    best = None
    best_score = -1
    best_df = None
    best_meta = None
    best_caps = None
    for dataset in datasets:
        df, meta = enrich_dataframe(dataset.dataframe)
        caps = _capability_summary(df, meta)
        score = sum(int(info["available"]) for info in caps.values())
        score += 2 * int(caps["Map"]["available"]) + int(caps["Vectors"]["available"])
        if score > best_score:
            best_score = score
            best = dataset
            best_df = df
            best_meta = meta
            best_caps = caps
    return best, best_df, best_meta, best_caps


def _render_overview_page(files: list[UploadedData]) -> None:
    if not files:
        st.caption("Select a data source from the sidebar to begin.")
        return
    prepare_started = time.perf_counter()
    with st.spinner("Loading selected data..."):
        datasets = _normalized_datasets(files)
    LOGGER.info("Prepared overview dataset in %.3fs for %s", time.perf_counter() - prepare_started, [f.name for f in files])
    best, df, meta, caps = _best_universal_dataset(datasets)
    ranked = ranked_plugins(files)
    recommended = next((item for item in ranked if item.confidence >= 0.30), None)
    if best is None or df is None or meta is None or caps is None:
        st.caption("The selected files do not expose a tabular dataset. Try Analysis for format-specific handling.")
        return
    profile = dataframe_profile(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Measurements", f"{profile['row_count']:,}")
    c2.metric("Variables", profile["column_count"])
    c3.metric("Numeric fields", profile["numeric_count"])
    c4.metric("Dataset", best.name)
    available = [MODE_LABELS[name] for name, info in caps.items() if info["available"] and name not in {"Overview", "Raw Data"}]
    details = []
    if meta.get("time_column"): details.append("time axis")
    if caps.get("Map", {}).get("available"): details.append("geographic coordinates")
    if caps.get("Vectors", {}).get("available"): details.append("vector data")
    if details:
        st.caption("Available data: " + ", ".join(details) + ".")
    if available:
        st.write("**Available views:** " + " · ".join(available))
    if recommended is not None:
        st.write(f"**Recommended analysis:** {recommended.plugin.name}")
    st.markdown("### Preview")
    if caps.get("Map", {}).get("available"):
        # Overview is a location preview, not a thematic map. Keep a single,
        # high-contrast marker style so the measurement positions remain easy
        # to see regardless of the first numeric field present in the dataset.
        fig = map_figure(df, meta, None)
        if fig is not None:
            fig.update_traces(marker={"size": 14, "color": "#E53935"})
    elif caps.get("Vectors", {}).get("available"):
        groups = detect_vector_groups(list(meta.get("numeric_columns", []))); label, columns = next(iter(groups.items())); fig = vector_timeseries_figure(df, meta, columns, label)
    else:
        fig = default_timeseries(df, meta, list(meta.get("numeric_columns", []))[:4])
    if fig is not None:
        st.plotly_chart(fig, width="stretch", config={"scrollZoom": True, "displaylogo": False}, key=f"plot_{_stable_ui_key('overview_preview', best.id)}")
    with st.expander("Dataset summary"):
        _render_overview(df, meta)


def _short_source_name(value: Any) -> str:
    text = str(value).replace("\\", "/")
    return PurePosixPath(text).stem or PurePosixPath(text).name or text


def _render_compare(files: list[UploadedData]) -> None:
    
    if len(files) < 2:
        st.info("Select at least two files to compare them.")
        return

    prepare_started = time.perf_counter()
    with st.spinner("Loading selected data..."):
        datasets = _normalized_datasets(files)
    LOGGER.info("Prepared comparison datasets in %.3fs for %s", time.perf_counter() - prepare_started, [f.name for f in files])
    if not datasets:
        st.info("The selected files could not be prepared for comparison.")
        return

    # Prefer a dedicated parser that already joined several experiments and kept
    # the source-file identity. This is ideal for Volatiles, Gas and Particles.
    combined: list[tuple[NormalizedDataset, str]] = []
    for dataset in datasets:
        for group_col in ("file", "source_file"):
            if group_col in dataset.dataframe.columns and dataset.dataframe[group_col].nunique(dropna=True) >= 2:
                combined.append((dataset, group_col))
                break

    if combined:
        labels = {f"{d.name} · {d.dataframe[g].nunique()} experiências": (d, g) for d, g in combined}
        choice = st.selectbox("Dataset", list(labels), key="compare_combined_dataset")
        dataset, group_col = labels[choice]
        df, meta = enrich_dataframe(dataset.dataframe)
        excluded = {
            meta.get("time_column"), meta.get("latitude_column"), meta.get("longitude_column"), meta.get("altitude_column"),
            "sample", "elapsed_s", "elapsed_min",
        }
        numeric = [c for c in meta.get("numeric_columns", []) if c not in excluded]
        if not numeric:
            st.info("No comparable numeric variables were found.")
            return
        variables = st.multiselect(
            "Variables",
            numeric,
            default=numeric[: min(4, len(numeric))],
            max_selections=8,
            key="compare_variables",
        )
        statistic = st.selectbox("Statistic", ["mean", "median", "max", "min"], key="compare_statistic")
        if not variables:
            return
        grouped_obj = df.groupby(group_col, dropna=False)[variables]
        if statistic == "median":
            summary = grouped_obj.median(numeric_only=True)
        elif statistic == "max":
            summary = grouped_obj.max(numeric_only=True)
        elif statistic == "min":
            summary = grouped_obj.min(numeric_only=True)
        else:
            summary = grouped_obj.mean(numeric_only=True)
        summary = summary.rename_axis("source").reset_index()
        summary["source_name"] = summary["source"].map(_short_source_name)
        long = summary.melt(id_vars=["source", "source_name"], value_vars=variables, var_name="variable", value_name="value")
        fig = px.bar(
            long,
            x="source_name",
            y="value",
            color="variable",
            barmode="group",
            hover_data={"source": True},
            title=f"Comparison · {statistic}",
        )
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), margin=dict(t=95, b=80))
        st.plotly_chart(fig, width="stretch", key=f"plot_{_stable_ui_key('compare', dataset.id, tuple(variables), statistic)}")
        st.dataframe(summary[["source_name", "source", *variables]], width="stretch", hide_index=True)
        return

    # Fallback: compare separate normalized datasets when they share a numeric field.
    if len(datasets) < 2:
        st.info("The selected parser produced a single combined dataset with no independent groups to compare.")
        return
    labels = {f"{d.name} · {len(d.dataframe):,} rows": d for d in datasets}
    selected_labels = st.multiselect(
        "Datasets",
        list(labels),
        default=list(labels)[: min(2, len(labels))],
        max_selections=4,
        key="compare_dataset_selection",
    )
    selected = [labels[label] for label in selected_labels]
    if len(selected) < 2:
        st.info("Select at least two datasets.")
        return
    numeric_sets = []
    for dataset in selected:
        _df, meta = enrich_dataframe(dataset.dataframe)
        numeric_sets.append(set(meta.get("numeric_columns", [])))
    common = sorted(set.intersection(*numeric_sets)) if numeric_sets else []
    if not common:
        st.info("The selected datasets do not share a numeric variable with the same name.")
        return
    variable = st.selectbox("Common variable", common, key="compare_common_variable")
    rows = []
    for dataset in selected:
        values = pd.to_numeric(dataset.dataframe[variable], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append({
            "dataset": dataset.name,
            "mean": float(values.mean()),
            "median": float(values.median()),
            "maximum": float(values.max()),
            "minimum": float(values.min()),
            "latest": float(values.iloc[-1]),
        })
    summary = pd.DataFrame(rows)
    if summary.empty:
        return
    fig = px.bar(summary, x="dataset", y=["mean", "median", "maximum", "minimum"], barmode="group", title=f"Comparison · {variable}")
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), margin=dict(t=95, b=100))
    st.plotly_chart(fig, width="stretch", key=f"plot_{_stable_ui_key('compare_fallback', variable, tuple(selected_labels))}")
    st.dataframe(summary, width="stretch", hide_index=True)


def _render_data(files: list[UploadedData]) -> None:
    if not files:
        st.caption("Select files from the sidebar to inspect or export data.")
        catalog = st.session_state.get("archive_catalog")
        if isinstance(catalog, pd.DataFrame) and not catalog.empty:
            with st.expander("Data.zip summary"):
                summary = catalog.groupby("category", as_index=False).agg(files=("files", "sum"), size_mb=("size_mb", "sum")).sort_values("files", ascending=False)
                st.dataframe(summary, width="stretch", hide_index=True)
        return
    datasets = _normalized_datasets(files)
    if not datasets:
        st.caption("No tabular dataset is available for the current selection.")
        return
    labels = {f"{d.name} · {len(d.dataframe):,} rows": d for d in datasets}
    source = labels[st.selectbox("Dataset", list(labels), key="data_dataset")]
    df, _meta = enrich_dataframe(source.dataframe)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Source files", len(source.source_files))
    preview = df if len(df) <= 50000 else df.head(50000)
    st.dataframe(preview, width="stretch", hide_index=True)
    if len(df) > len(preview):
        st.caption(f"Showing 50,000 of {len(df):,} rows.")
    c1, c2 = st.columns(2)
    csv_payload = df.to_csv(index=False).encode("utf-8-sig")
    c1.download_button("Export CSV", csv_payload, file_name=f"{source.id}.csv", mime="text/csv", key=f"data_csv_{_stable_ui_key(source.id)}", width="stretch")
    db_config = _current_database_config()
    save_clicked = c2.button(
        "Save to remote database",
        key=f"data_database_{_stable_ui_key(source.id)}",
        width="stretch",
        disabled=db_config is None,
        help=None if db_config is not None else "Select Remote database as the data source once, or configure server-side database secrets.",
    )
    if save_clicked and db_config is not None:
        try:
            table = save_dataframe(source.name, df, db_config)
            _cached_database_tables.clear()
            _cached_database_dataframe.clear()
            st.success(f"Saved to PostgreSQL table {table}.")
        except Exception as exc:
            _log_exception("PostgreSQL save", exc)
            st.error("Could not save this dataset to the remote database. See logs/sensor_analytics.log for details.")
    with st.expander("Source details"):
        st.write(f"Parser: `{source.parser}`")
        for name in source.source_files:
            st.write(name)
        for note in source.notes[:6]:
            st.caption(note)
    catalog = st.session_state.get("archive_catalog")
    if isinstance(catalog, pd.DataFrame) and not catalog.empty:
        with st.expander("Data.zip summary"):
            summary = catalog.groupby("category", as_index=False).agg(files=("files", "sum"), size_mb=("size_mb", "sum")).sort_values("files", ascending=False)
            st.dataframe(summary, width="stretch", hide_index=True)


files = _uploads()

st.markdown(
    """
    <div class="ast-hero">
      <h2 style="margin:0">AST Sensor Analytics</h2>
      <div class="ast-muted">Sensor data visualisation and analysis</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if files:
    st.caption("Active files: " + ", ".join(f.name for f in files))
else:
    st.caption("Select a data source from the sidebar.")

section_options = ["Overview", "Visualisation", "Analysis"]
if len(files) >= 2:
    section_options.append("Compare")
section_options.append("Data")
if st.session_state.get("main_section") not in (None, *section_options):
    st.session_state.pop("main_section", None)

if hasattr(st, "segmented_control"):
    section = st.segmented_control("Section", section_options, default=section_options[0], key="main_section", label_visibility="collapsed") or section_options[0]
else:
    section = st.radio("Section", section_options, horizontal=True, key="main_section")

if section == "Overview":
    _render_overview_page(files)
elif section == "Visualisation":
    _render_generic(files)
elif section == "Analysis":
    _render_special(files)
elif section == "Compare":
    _render_compare(files)
else:
    _render_data(files)
