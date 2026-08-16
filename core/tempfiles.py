from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path

from .models import UploadedData


@contextmanager
def materialized_uploads(files: list[UploadedData]):
    with tempfile.TemporaryDirectory(prefix="ast_sensor_") as tmp:
        root = Path(tmp)
        paths: list[Path] = []
        for index, file in enumerate(files):
            # Avoid path traversal and duplicate names.
            safe_name = Path(file.name).name or f"upload_{index}.dat"
            target = root / safe_name
            if target.exists():
                target = root / f"{index}_{safe_name}"
            target.write_bytes(file.raw)
            paths.append(target)
        yield root, paths
