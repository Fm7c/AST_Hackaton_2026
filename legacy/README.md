Drop trusted scientific Python analyses in this folder. AST Sensor Analytics scans them automatically.

For zero configuration, include the sensor in the filename (for example `analise_CH4.py`) and expose a function such as `analisar_ficheiro(path)`, `analyze_file(path)`, `analisar(df)` or `analyze(df)`.

If the script uses unusual names, keep the Python unchanged and add a same-name `.json` sidecar. See `../LEGACY_ANALYSES.md`.
