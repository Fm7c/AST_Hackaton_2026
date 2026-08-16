# Remote database

AST Sensor Analytics can read ordinary PostgreSQL tables and can also save a normalized dataset back to PostgreSQL.

The browser never connects directly to PostgreSQL. The Streamlit application server opens the database TCP connection:

```text
Browser -> HTTPS -> AST Sensor Analytics -> PostgreSQL host:port
```

## Connection values

- Host: DNS name or IP address reachable by the Streamlit server
- Port: normally `5432` for PostgreSQL
- Database name
- Username
- Password
- TLS/SSL mode (`require` is the safe default for hosted databases)

For local testing, select **Remote database** in the sidebar and enter the values. The password is kept only in the Streamlit server session.

For a deployed/public app, put the credentials in Streamlit secrets instead of displaying or committing them. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` locally, or paste the same `[database]` block into the deployment platform's secrets area.

Environment variables are also supported:

```text
AST_DB_HOST
AST_DB_PORT
AST_DB_NAME
AST_DB_USER
AST_DB_PASSWORD
AST_DB_SSLMODE
```

## Network note

A database on `192.168.x.x`, `10.x.x.x` or another private LAN address is not reachable from a public cloud app unless the cloud app has a private route/VPN/tunnel to that network. For an Internet deployment, use a database endpoint reachable from the app server and protect it with TLS, authentication and firewall/network rules. Do not expose PostgreSQL anonymously to the whole Internet.
