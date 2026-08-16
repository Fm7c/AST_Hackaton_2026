from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io.matlab._mio5_params import MatlabOpaque

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data_loader import _opaque_table_names


def _make_opaque(table_name: str) -> MatlabOpaque:
    # Minimal stand-in for what scipy hands back for a real MCOS-encoded
    # MATLAB table: a structured array exposing an "s0" field.
    dtype = np.dtype([("s0", object)])
    arr = np.empty((1,), dtype=dtype)
    arr["s0"][0] = table_name
    return MatlabOpaque(arr)


def main() -> None:
    # The real variable name inside the .mat file is "GGA_Table_1" -- never
    # the literal string "None". If _opaque_table_names still looked it up
    # under data.get("None") (the bug), this dict lookup would miss and the
    # function would return [] even though a matching MatlabOpaque is present.
    fake_stream = io.BytesIO(b"irrelevant-mat-bytes")
    with patch(
        "core.data_loader.varmats_from_mat",
        return_value=[("GGA_Table_1", fake_stream)],
    ) as mock_varmats, patch(
        "core.data_loader.loadmat",
        return_value={"GGA_Table_1": _make_opaque("GGA")},
    ) as mock_loadmat:
        names = _opaque_table_names(b"irrelevant-raw-bytes")

    assert names == ["GGA"], f"expected ['GGA'], got {names}"
    mock_varmats.assert_called_once()
    mock_loadmat.assert_called_once()

    # Second variable in the same file with no matching name: still found,
    # proving the fix uses each loop iteration's own real name.
    with patch(
        "core.data_loader.varmats_from_mat",
        return_value=[("RMC_Table_9", io.BytesIO(b"irrelevant"))],
    ), patch(
        "core.data_loader.loadmat",
        return_value={"RMC_Table_9": _make_opaque("RMC")},
    ):
        names_2 = _opaque_table_names(b"irrelevant-raw-bytes")
    assert names_2 == ["RMC"], f"expected ['RMC'], got {names_2}"

    print("MCOS table detection (_opaque_table_names): PASS")


if __name__ == "__main__":
    main()
