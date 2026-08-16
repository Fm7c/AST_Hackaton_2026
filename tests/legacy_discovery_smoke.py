from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.legacy_discovery import discover_legacy_plugins, legacy_script_statuses
from core.models import UploadedData


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="ast_legacy_test_") as tmp:
        root = Path(tmp)

        # Zero-configuration CH4 script: filename + conventional function/parameter.
        (root / "analise_CH4.py").write_text(
            """
import pandas as pd
import matplotlib.pyplot as plt

def analisar_ficheiro(path):
    df = pd.read_csv(path)
    plt.figure()
    plt.plot(df['ch4'])
    return {'Mean CH4': float(df['ch4'].mean()), 'Measurements': df}
""",
            encoding="utf-8",
        )

        # Unusual legacy function made compatible without changing Python source.
        (root / "old_calculation.py").write_text(
            """
def legacy_entry(source):
    return {'status': 'ok', 'source': source}
""",
            encoding="utf-8",
        )
        (root / "old_calculation.json").write_text(
            json.dumps({
                "name": "CH4 Old Calculation",
                "sensor": "ch4",
                "function": "legacy_entry",
                "input": "file",
            }),
            encoding="utf-8",
        )

        # Visible but intentionally not executed because nothing identifies sensor/entry point.
        (root / "mystery.py").write_text("x = 1\n", encoding="utf-8")

        plugins = discover_legacy_plugins(root)
        assert len(plugins) == 2, [plugin.name for plugin in plugins]
        names = {plugin.name for plugin in plugins}
        assert "CH4" in names
        assert "CH4 Old Calculation" in names

        source = UploadedData(
            name="Akel_CH4_test.csv",
            raw=b"ch4\n1\n2\n3\n",
            dataframe=pd.DataFrame({"ch4": [1, 2, 3]}),
            kind="tabular",
        )
        auto = next(plugin for plugin in plugins if plugin.name == "CH4")
        assert auto.confidence([source]) >= 0.9
        result = auto.run([source], {})
        assert result.metrics["Mean CH4"] == 2.0
        assert "Measurements" in result.tables
        assert result.figures and isinstance(result.figures[0][1], bytes)

        old = next(plugin for plugin in plugins if plugin.name == "CH4 Old Calculation")
        old_result = old.run([source], {})
        assert old_result.metrics["status"] == "ok"

        statuses = legacy_script_statuses(root)
        mystery = next(status for status in statuses if status.filename == "mystery.py")
        assert not mystery.supported
        assert mystery.reason

    print("Legacy discovery: PASS")


if __name__ == "__main__":
    main()
