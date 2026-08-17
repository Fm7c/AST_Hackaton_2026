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


def _make_opaque() -> MatlabOpaque:
    # Minimal stand-in for what scipy hands back for a real MCOS-encoded
    # MATLAB table: a structured array exposing "_TypeSystem", "_Class" and
    # "_ObjectMetadata" fields, matching real MAT files (confirmed against
    # sample_data/official/Data_1_Base_2025-11-25_14-40-22_Taveiro.mat).
    # "_ObjectMetadata" is an opaque blob scipy cannot decode into text, so
    # the table's name has to come from the real variable name instead.
    dtype = np.dtype([("_TypeSystem", object), ("_Class", object), ("_ObjectMetadata", object)])
    arr = np.empty((1,), dtype=dtype)
    arr["_TypeSystem"][0] = "MCOS"
    arr["_Class"][0] = "table"
    arr["_ObjectMetadata"][0] = np.array([0], dtype=np.uint32)
    return MatlabOpaque(arr)


def main() -> None:
    # The real variable name inside the .mat file is "GGA_Table_1". The opaque
    # blob has no field holding a human-readable name -- the fix uses the real
    # variable name from varmats_from_mat directly as the table identifier.
    fake_stream = io.BytesIO(b"irrelevant-mat-bytes")
    with patch(
        "core.data_loader.varmats_from_mat",
        return_value=[("GGA_Table_1", fake_stream)],
    ) as mock_varmats, patch(
        "core.data_loader.loadmat",
        return_value={"GGA_Table_1": _make_opaque()},
    ) as mock_loadmat:
        names = _opaque_table_names(b"irrelevant-raw-bytes")

    assert names == ["GGA_Table_1"], f"expected ['GGA_Table_1'], got {names}"
    mock_varmats.assert_called_once()
    mock_loadmat.assert_called_once()

    # Second variable in the same file with a different name: still found,
    # proving the fix uses each loop iteration's own real name.
    with patch(
        "core.data_loader.varmats_from_mat",
        return_value=[("RMC_Table_9", io.BytesIO(b"irrelevant"))],
    ), patch(
        "core.data_loader.loadmat",
        return_value={"RMC_Table_9": _make_opaque()},
    ):
        names_2 = _opaque_table_names(b"irrelevant-raw-bytes")
    assert names_2 == ["RMC_Table_9"], f"expected ['RMC_Table_9'], got {names_2}"

    print("MCOS table detection (_opaque_table_names): PASS")


if __name__ == "__main__":
    main()
