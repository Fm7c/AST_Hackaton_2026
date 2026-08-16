# AST Sensor Analytics

AST Sensor Analytics is a browser-based tool for exploring heterogeneous sensor datasets without switching between separate analysis scripts. It supports generic visualisation and dedicated analyses for the supplied hackathon datasets and the existing UV, radiation and AS3935 analysis code.

## Main workflow

1. Select a data source in the sidebar.
2. Use **Overview** to inspect the selected dataset.
3. Use **Visualisation** for the views supported by that dataset. Only relevant views are shown.
4. Use **Analysis** for a format-specific scientific analysis when one is available.
5. Use **Compare** when two or more compatible files are selected.
6. Use **Data** to inspect, export or save a normalized dataset to the configured remote PostgreSQL database.

## Supported data sources

- `Data.zip` supplied with the challenge
- CSV / TXT / TSV / DAT / LOG / JSON
- Excel
- MATLAB MAT
- ZIP archives
- Remote PostgreSQL tables
- Built-in demonstration datasets

## Generic views

Depending on the selected data, the application exposes only the relevant views: Time Series, Bars & Indicators, Map, Vectors, Alerts and Playback. When GNSS coordinates and meteorological fields are present, the Weather map can overlay temperature, humidity and pressure at the same geographic samples.

## Dedicated analyses

- Volatile sensor response profiles
- Particulate matter / SPS
- Gas / Alcohol / CH4 logs
- GNSS precision (ECEF / WGS84 / ENU)
- GNSS RTK
- GNSS satellite observations and navigation
- RTK MATLAB table catalogue
- IMU / magnetometer
- AS3935 lightning
- Radiation event and dose analysis
- UV multi-sensor analysis

## Windows

Double-click `run_windows.bat`. Python 3.11 or 3.12 is required. The launcher also installs the PostgreSQL client dependencies if this version is copied over an existing `.venv`.

## Remote database

See `DATABASE.md`. PostgreSQL can be addressed by DNS name or IP plus TCP port. For a public deployment, keep credentials in Streamlit secrets rather than in the source code.

## Deployment

See `DEPLOY.md`.

## Logs

Unexpected application errors are written to `logs/sensor_analytics.log`.

## Automatic legacy analyses

New scientific Python scripts can be dropped into `legacy/` and are discovered automatically from their sensor name and analysis entry point. Existing UV, radiation and lightning adapters remain unchanged. See `LEGACY_ANALYSES.md` for the supported zero-configuration conventions and the optional JSON sidecar for older scripts.
