from __future__ import annotations

import csv
import io
import json
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.io.matlab import MatReadWarning, varmats_from_mat
from scipy.io.matlab._mio5_params import MatlabOpaque

from .models import UploadedData


TIME_CANDIDATES = (
    "timestamp", "datetime", "date_time", "time", "tempo", "data_hora",
    "hora", "date", "data",
    # Relative/sample time axes used by sensor parsers. These remain numeric.
    "elapsed_s", "elapsed_min", "tempo_min", "seconds", "segundos", "itow", "sample",
)
LAT_CANDIDATES = ("latitude", "lat")
LON_CANDIDATES = ("longitude", "lon", "lng", "long")
ALT_CANDIDATES = ("altitude", "alt", "elevation")


def _normalized_column(name: object) -> str:
    text = str(name).strip().lower()
    text = (
        text.replace("º", "")
        .replace("°", "")
        .replace("µ", "u")
        .replace("²", "2")
    )
    return re.sub(r"[^a-z0-9_]+", "_", text).strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    normalized: list[str] = []
    used: Counter[str] = Counter()
    for column in result.columns:
        base = _normalized_column(column) or "column"
        used[base] += 1
        normalized.append(base if used[base] == 1 else f"{base}_{used[base]}")
    result.columns = normalized
    return result


def _decode_text(raw: bytes) -> str | None:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def _try_csv(raw: bytes) -> pd.DataFrame | None:
    """Read common delimited sensor files, including malformed two-column headers.

    The official Volatiles pack has a header ``timestamp,data`` while each data row
    actually contains timestamp + 17 numeric values. Pandas otherwise turns the
    leading values into a MultiIndex. Detect that case before the generic parser.
    """
    text = _decode_text(raw)
    if text is None:
        return None

    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) >= 2:
        try:
            first = next(csv.reader([lines[0]]))
            second = next(csv.reader([lines[1]]))
        except Exception:
            first, second = [], []

        if len(second) >= 3 and len(second) > len(first):
            names: list[str]
            if first and first[0].strip().lower() == "timestamp":
                names = ["timestamp"] + [f"channel_{idx:02d}" for idx in range(1, len(second))]
            else:
                names = [f"column_{idx:02d}" for idx in range(len(second))]
            try:
                df = pd.read_csv(
                    io.StringIO(text),
                    header=None,
                    names=names,
                    skiprows=1,
                    usecols=range(len(names)),
                    engine="python",
                )
                if len(df) >= 1:
                    return df
            except Exception:
                pass

    # Flexible enough for comma/semicolon/tab files typically used by sensors.
    for sep in (None, ";", ",", "\t"):
        try:
            df = pd.read_csv(
                io.StringIO(text),
                sep=sep,
                engine="python" if sep is None else "c",
            )
            # Reject the common case where a free-form log was interpreted as a
            # one-column table. Special plugins can still parse those raw bytes.
            if len(df.columns) >= 2 and len(df) >= 1:
                return df
        except Exception:
            pass
    return None


def _opaque_table_names(raw: bytes) -> list[str]:
    names: list[str] = []
    try:
        for _var_name, stream in varmats_from_mat(io.BytesIO(raw)):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    data = loadmat(stream, squeeze_me=True, struct_as_record=False)
                value = data.get(_var_name)
                if isinstance(value, MatlabOpaque) and value.dtype.names and "s0" in value.dtype.names:
                    item = value["s0"]
                    if getattr(item, "size", 0):
                        candidate = item.flat[0]
                        if isinstance(candidate, bytes):
                            candidate = candidate.decode("utf-8", errors="replace")
                        candidate = str(candidate).strip()
                        if candidate and candidate not in names:
                            names.append(candidate)
            except Exception:
                continue
    except Exception:
        pass
    return names


def _vector_from_value(value: object) -> np.ndarray | None:
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.ndim == 0 or arr.size < 2:
        return None
    arr = np.squeeze(arr)
    if arr.ndim != 1:
        return None
    # Object arrays (MATLAB cells/tables) are not useful as generic columns.
    if arr.dtype.kind in {"O", "V"}:
        return None
    return arr


def _mat_dataframe(raw: bytes) -> tuple[pd.DataFrame | None, dict[str, object]]:
    metadata: dict[str, object] = {"matlab_tables": [], "matlab_variables": []}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MatReadWarning)
            mat = loadmat(io.BytesIO(raw), squeeze_me=True, struct_as_record=False)
    except Exception as exc:
        metadata["mat_error"] = str(exc)
        return None, metadata

    variables = [(key, value) for key, value in mat.items() if not key.startswith("__")]
    metadata["matlab_variables"] = [str(key) for key, _ in variables]

    candidates: dict[str, np.ndarray] = {}
    for key, value in variables:
        # Typical official RTK_BaseRover files contain one MATLAB struct named data.
        field_names = getattr(value, "_fieldnames", None)
        if field_names:
            for field in field_names:
                arr = _vector_from_value(getattr(value, field, None))
                if arr is not None:
                    candidates[str(field)] = arr
            continue

        arr = _vector_from_value(value)
        if arr is not None:
            candidates[str(key)] = arr

    if candidates:
        size_counts = Counter(int(arr.size) for arr in candidates.values())
        # Prefer the length shared by the greatest number of variables; break ties
        # in favour of the longest vector.
        common_size = max(size_counts, key=lambda size: (size_counts[size], size))
        aligned = {key: arr for key, arr in candidates.items() if int(arr.size) == common_size}
        if len(aligned) >= 2:
            try:
                return pd.DataFrame(aligned), metadata
            except Exception:
                pass

    table_names = _opaque_table_names(raw)
    if table_names:
        metadata["matlab_tables"] = table_names
        metadata["matlab_mcos"] = True
    return None, metadata


def load_tabular_file(name: str, raw: bytes) -> pd.DataFrame | None:
    suffix = Path(name).suffix.lower()
    try:
        if suffix in {".csv", ".txt", ".tsv", ".dat", ".log"}:
            return _try_csv(raw)
        if suffix in {".xlsx", ".xlsm", ".xls"}:
            return pd.read_excel(io.BytesIO(raw))
        if suffix == ".json":
            obj = json.loads(raw.decode("utf-8-sig"))
            if isinstance(obj, list):
                return pd.DataFrame(obj)
            if isinstance(obj, dict):
                for key in ("data", "records", "measurements", "values"):
                    if isinstance(obj.get(key), list):
                        return pd.DataFrame(obj[key])
                return pd.json_normalize(obj)
        if suffix == ".mat":
            df, _metadata = _mat_dataframe(raw)
            return df
    except Exception:
        return None
    return None


def build_uploaded_data(name: str, raw: bytes) -> UploadedData:
    suffix = Path(name).suffix.lower()
    metadata: dict[str, object] = {}

    if suffix == ".mat":
        df, metadata = _mat_dataframe(raw)
    else:
        df = load_tabular_file(name, raw)

    if df is not None:
        df = normalize_columns(df)
        return UploadedData(name=name, raw=raw, dataframe=df, kind="tabular", metadata=metadata)

    kind = "matlab_mcos" if metadata.get("matlab_mcos") else "raw_text"
    return UploadedData(name=name, raw=raw, dataframe=None, kind=kind, metadata=metadata)


def detect_time_column(df: pd.DataFrame) -> str | None:
    lower = {_normalized_column(c): c for c in df.columns}
    for candidate in TIME_CANDIDATES:
        if candidate in lower:
            return lower[candidate]

    # Fallback: only test text/object columns. Numeric sensor columns can otherwise
    # be silently interpreted as nanoseconds since 1970.
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        series = df[col].dropna()
        if len(series) < 2:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(series, errors="coerce")
            if parsed.notna().mean() >= 0.8:
                return col
        except Exception:
            pass
    return None


def detect_coordinate_columns(df: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
    normalized = {_normalized_column(c): c for c in df.columns}

    def pick(candidates: Iterable[str]) -> str | None:
        for candidate in candidates:
            if candidate in normalized:
                return normalized[candidate]
        return None

    return pick(LAT_CANDIDATES), pick(LON_CANDIDATES), pick(ALT_CANDIDATES)


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return measurement-like numeric columns.

    Datetime columns must not be treated as numeric measurements. Pandas can
    convert ``datetime64`` values to integer nanoseconds, which previously made
    a detected ``timestamp`` appear in selectors such as Alerts and could create
    duplicate ``[timestamp, timestamp]`` tables.
    """
    cols: list[str] = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series):
            continue
        converted = pd.to_numeric(series, errors="coerce")
        min_valid = 1 if len(df) < 2 else max(2, int(len(df) * 0.5))
        if converted.notna().sum() >= min_valid:
            cols.append(col)
    return cols


def enrich_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    result = df.copy()
    time_col = detect_time_column(result)
    time_is_datetime = False
    if time_col is not None:
        # Do not turn relative numeric axes (elapsed_s, sample, iTOW...) into
        # nanoseconds since 1970.  Datetime parsing is only useful for text/date
        # columns or already-datetime series.
        if pd.api.types.is_datetime64_any_dtype(result[time_col]):
            time_is_datetime = True
        elif not pd.api.types.is_numeric_dtype(result[time_col]):
            parsed = pd.to_datetime(result[time_col], errors="coerce")
            if parsed.notna().sum() >= 2:
                result[time_col] = parsed
                time_is_datetime = True

    lat, lon, alt = detect_coordinate_columns(result)
    nums = numeric_columns(result)
    metadata = {
        "time_column": time_col,
        "time_is_datetime": time_is_datetime,
        "latitude_column": lat,
        "longitude_column": lon,
        "altitude_column": alt,
        "numeric_columns": nums,
        "row_count": len(result),
        "column_count": len(result.columns),
    }
    return result, metadata


def dataframe_profile(df: pd.DataFrame) -> dict[str, object]:
    enriched, meta = enrich_dataframe(df)
    numeric = meta["numeric_columns"]
    return {
        **meta,
        "columns": list(enriched.columns),
        "missing_values": int(enriched.isna().sum().sum()),
        "numeric_count": len(numeric),
        "has_time": meta["time_column"] is not None,
        "has_map": meta["latitude_column"] is not None and meta["longitude_column"] is not None,
    }
