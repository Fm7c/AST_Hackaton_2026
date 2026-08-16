from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class UploadedData:
    """Representation of one uploaded/source file."""

    name: str
    raw: bytes
    dataframe: pd.DataFrame | None = None
    kind: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedDataset:
    """A dataframe made available to the universal/base dashboard engine.

    Special parsers can turn free-form logs, MATLAB structures or multi-sheet
    workbooks into this common representation.  Base modes only need a dataframe;
    they do not need to know which sensor created it.
    """

    id: str
    name: str
    dataframe: pd.DataFrame
    source_files: list[str] = field(default_factory=list)
    parser: str = "Generic table loader"
    description: str = ""
    special_plugin_id: str | None = None
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Common contract returned by every special-analysis plugin."""

    title: str
    summary: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: list[tuple[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    downloads: dict[str, tuple[bytes, str]] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
