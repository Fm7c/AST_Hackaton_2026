# Automatic legacy analyses

`legacy/` is scanned automatically. Adding a new Python analysis does not require changes to `app.py` or a new file in `plugins/`.

## Zero-configuration path

Use a sensor name in the filename and expose a conventional analysis function.

Example: `legacy/analise_CH4.py`

```python
def analisar_ficheiro(path):
    ...
    return {
        "Mean CH4": mean_value,
        "Measurements": dataframe,
    }
```

Recognised entry-point names include `analisar_ficheiro`, `analyze_file`, `analisar`, `analyze`, `processar`, `process`, `run_analysis` and `run`.

Recognised input parameter names include file/path, files/paths, folder/directory, df/dataframe and options/config. A script can return an `AnalysisResult`, a pandas DataFrame/Series, a dictionary, Plotly figure, Matplotlib figure, scalar, list or tuple. Matplotlib figures left open by the script are captured and displayed in the dashboard; `plt.show()` is suppressed while the analysis runs.

Sensor families are inferred deterministically from the filename/source (for example CH4/methane, UV, radiation, lightning, GNSS, particles, volatiles, weather, accelerometer, gyroscope and magnetometer). The analysis only appears when the selected dataset matches that family.

After copying a `.py` file into `legacy/`, refresh the browser or interact with the app once. The folder signature is checked on every Streamlit rerun, while the parsed script description is cached by filename, modification time and size.

## Optional sidecar manifest

If an older script uses an unusual function name, keep the Python file unchanged and add a JSON file with the same basename.

`legacy/my_old_analysis.py`

`legacy/my_old_analysis.json`

```json
{
  "name": "CH4 Peak Analysis",
  "sensor": "ch4",
  "function": "calculate_peaks",
  "input": "file",
  "description": "Peak analysis for CH4 measurements."
}
```

Valid `input` values are `file`, `files`, `directory`, `dataframe`, `dataframes` and `none`.

If a script cannot be mapped safely, it is not executed automatically. The Analysis page shows it under **Legacy scripts needing configuration** with the reason.

## Existing scientific scripts

The current UV, radiation and lightning scripts remain handled by their dedicated adapters because those adapters preserve their domain-specific outputs. They are not duplicated by the automatic scanner.

## Security

A Python file in `legacy/` is executable code and runs with the same operating-system permissions as AST Sensor Analytics. Only place trusted scripts in that folder.
