from __future__ import annotations

import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Iterable

import pandas as pd


IGNORED_NAMES = {".DS_Store", "Thumbs.db"}
SUPPORTED_EXTENSIONS = {
    ".csv", ".txt", ".tsv", ".dat", ".log", ".json",
    ".xlsx", ".xlsm", ".xls", ".mat",
}


def is_real_member(info: zipfile.ZipInfo) -> bool:
    if info.is_dir():
        return False
    name = info.filename
    if name.startswith("__MACOSX/"):
        return False
    path = PurePosixPath(name)
    if path.name in IGNORED_NAMES:
        return False
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def category_for_member(name: str) -> str:
    parts = PurePosixPath(name).parts
    if len(parts) >= 2 and parts[0].lower() == "data":
        return parts[1]
    return parts[0] if parts else "Archive"


def subgroup_for_member(name: str) -> str:
    parts = PurePosixPath(name).parts
    if len(parts) >= 4 and parts[0].lower() == "data":
        return parts[2]
    if len(parts) >= 3 and parts[0].lower() != "data":
        return parts[1]
    return "(root)"


def members_from_zip(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    return [info for info in zf.infolist() if is_real_member(info)]


def summarize_infos(infos: Iterable[zipfile.ZipInfo]) -> pd.DataFrame:
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    for info in infos:
        category = category_for_member(info.filename)
        subgroup = subgroup_for_member(info.filename)
        key = (category, subgroup)
        row = grouped.setdefault(
            key,
            {
                "category": category,
                "subgroup": subgroup,
                "files": 0,
                "size_bytes": 0,
                "extensions": set(),
            },
        )
        row["files"] = int(row["files"]) + 1
        row["size_bytes"] = int(row["size_bytes"]) + int(info.file_size)
        cast_set = row["extensions"]
        if isinstance(cast_set, set):
            cast_set.add(PurePosixPath(info.filename).suffix.lower())

    rows = []
    for row in grouped.values():
        extensions = row.pop("extensions")
        rows.append(
            {
                **row,
                "size_mb": round(int(row["size_bytes"]) / 1_000_000, 3),
                "extensions": ", ".join(sorted(extensions)) if isinstance(extensions, set) else "",
            }
        )
    if not rows:
        return pd.DataFrame(columns=["category", "subgroup", "files", "size_mb", "extensions"])
    return (
        pd.DataFrame(rows)
        .drop(columns=["size_bytes"])
        .sort_values(["category", "subgroup"], key=lambda s: s.astype(str).str.lower())
        .reset_index(drop=True)
    )


def summarize_zip(path: str | Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        return summarize_infos(members_from_zip(zf))


def default_pack_candidates(app_root: Path) -> list[Path]:
    candidates = [
        app_root / "Data.zip",
        app_root.parent / "Data.zip",
        Path.cwd() / "Data.zip",
        Path.home() / "Downloads" / "Data.zip",
        Path.home() / "Desktop" / "Data.zip",
    ]
    seen: set[Path] = set()
    result: list[Path] = []
    for path in candidates:
        try:
            resolved = path.expanduser().resolve()
        except Exception:
            resolved = path.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(resolved)
    return result


def find_default_pack(app_root: Path) -> Path | None:
    for path in default_pack_candidates(app_root):
        if path.is_file():
            return path
    return None
