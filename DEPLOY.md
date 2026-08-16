# Deploy AST Sensor Analytics

The app is a browser application and can run locally, on a private server, or on a public Streamlit deployment.

## Streamlit Community Cloud

1. Push this project to a GitHub repository. Do **not** commit `Data.zip` or `.streamlit/secrets.toml`.
2. Create a Streamlit app that points to `app.py` and use Python 3.12.
3. The integrated demo datasets are included, so the public demo works without the large official archive.
4. To use the external database online, add the `[database]` block from `.streamlit/secrets.toml.example` to the app's Secrets settings.
5. The PostgreSQL endpoint must be reachable from the deployed Streamlit server over its configured TCP port.

The resulting URL can be opened from any browser on the Internet, subject to the deployment's access settings.

## Docker / any server

```bash
docker build -t ast-sensor-analytics .
docker run --rm -p 8501:8501 ast-sensor-analytics
```

For production, place HTTPS/reverse proxy or the hosting platform in front of Streamlit and keep database credentials server-side.

## Local network demo

The included Streamlit configuration listens on `0.0.0.0`. If the firewall allows TCP 8501, another device on the same LAN can open:

```text
http://IP-OF-THE-COMPUTER:8501
```

See `DATABASE.md` for the PostgreSQL connection model.
