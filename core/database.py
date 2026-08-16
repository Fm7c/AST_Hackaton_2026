from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache

import pandas as pd
from sqlalchemy import MetaData, Table, create_engine, inspect, select, text
from sqlalchemy.engine import Engine, URL


REGISTRY_TABLE = "ast_datasets"


@dataclass(frozen=True)
class DatabaseConfig:
    """Connection details for the external PostgreSQL database.

    The dashboard server connects to PostgreSQL over TCP. ``host`` may be a DNS
    name or an IP address. Credentials stay on the server side; the browser never
    connects directly to PostgreSQL.
    """

    host: str
    port: int
    database: str
    username: str
    password: str
    sslmode: str = "require"

    def validate(self) -> None:
        if not self.host.strip():
            raise ValueError("Database host/IP is required.")
        if not (1 <= int(self.port) <= 65535):
            raise ValueError("Database TCP port must be between 1 and 65535.")
        if not self.database.strip():
            raise ValueError("Database name is required.")
        if not self.username.strip():
            raise ValueError("Database username is required.")
        if self.sslmode not in {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}:
            raise ValueError(f"Unsupported PostgreSQL sslmode: {self.sslmode}")


def _database_url(config: DatabaseConfig) -> URL:
    config.validate()
    return URL.create(
        drivername="postgresql+psycopg",
        username=config.username,
        password=config.password,
        host=config.host,
        port=int(config.port),
        database=config.database,
        query={"sslmode": config.sslmode},
    )


@lru_cache(maxsize=8)
def get_engine(config: DatabaseConfig) -> Engine:
    """Return a small, reusable SQLAlchemy pool for one PostgreSQL endpoint."""
    return create_engine(
        _database_url(config),
        pool_pre_ping=True,
        pool_recycle=300,
        connect_args={"connect_timeout": 8},
    )


def test_connection(config: DatabaseConfig) -> None:
    with get_engine(config).connect() as con:
        con.execute(text("SELECT 1"))


def initialize(config: DatabaseConfig) -> None:
    """Create only the small AST registry table; sensor tables remain independent."""
    engine = get_engine(config)
    with engine.begin() as con:
        con.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS {REGISTRY_TABLE} (
                    id BIGSERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    table_name TEXT NOT NULL UNIQUE,
                    imported_at TIMESTAMPTZ NOT NULL,
                    row_count BIGINT NOT NULL,
                    column_count INTEGER NOT NULL,
                    columns_json TEXT NOT NULL
                )
                """
            )
        )


def _safe_table_prefix(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", str(name).lower()).strip("_")
    cleaned = re.sub(r"_+", "_", cleaned)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"data_{cleaned}" if cleaned else "data"
    return cleaned[:30]


def list_tables(config: DatabaseConfig) -> list[str]:
    """List ordinary tables that the dashboard can read, including external tables."""
    engine = get_engine(config)
    tables = inspect(engine).get_table_names(schema="public")
    return sorted(t for t in tables if t != REGISTRY_TABLE)


def list_datasets(config: DatabaseConfig) -> pd.DataFrame:
    initialize(config)
    query = text(
        f"""
        SELECT id, name, table_name, imported_at, row_count, column_count, columns_json
        FROM {REGISTRY_TABLE}
        ORDER BY id DESC
        """
    )
    with get_engine(config).connect() as con:
        return pd.read_sql(query, con)


def _validated_table(config: DatabaseConfig, table_name: str) -> Table:
    available = set(list_tables(config))
    if table_name not in available:
        raise ValueError(f"Database table not found: {table_name}")
    metadata = MetaData()
    return Table(table_name, metadata, schema="public", autoload_with=get_engine(config))


def load_dataframe(
    table_name: str,
    config: DatabaseConfig,
    limit: int | None = None,
) -> pd.DataFrame:
    table = _validated_table(config, table_name)
    statement = select(table)
    if limit is not None:
        statement = statement.limit(max(1, int(limit)))
    with get_engine(config).connect() as con:
        return pd.read_sql(statement, con)


def save_dataframe(name: str, df: pd.DataFrame, config: DatabaseConfig) -> str:
    """Persist an arbitrary normalized dataset as its own PostgreSQL table."""
    initialize(config)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    table_name = f"ast_{_safe_table_prefix(name)}_{timestamp}"[:62]
    engine = get_engine(config)

    # Pandas/SQLAlchemy quote column names safely, so sensor files can keep their
    # original labels. Chunking prevents a very large INSERT statement.
    df.to_sql(
        table_name,
        engine,
        schema="public",
        if_exists="fail",
        index=False,
        chunksize=2000,
        method="multi",
    )

    with engine.begin() as con:
        con.execute(
            text(
                f"""
                INSERT INTO {REGISTRY_TABLE}
                    (name, table_name, imported_at, row_count, column_count, columns_json)
                VALUES
                    (:name, :table_name, :imported_at, :row_count, :column_count, :columns_json)
                """
            ),
            {
                "name": str(name),
                "table_name": table_name,
                "imported_at": datetime.now(timezone.utc),
                "row_count": int(len(df)),
                "column_count": int(len(df.columns)),
                "columns_json": json.dumps(list(map(str, df.columns)), ensure_ascii=False),
            },
        )
    return table_name


def delete_dataset(dataset_id: int, config: DatabaseConfig) -> bool:
    initialize(config)
    engine = get_engine(config)
    with engine.begin() as con:
        row = con.execute(
            text(f"SELECT table_name FROM {REGISTRY_TABLE} WHERE id=:id"),
            {"id": int(dataset_id)},
        ).fetchone()
        if not row:
            return False
        table_name = str(row[0])

    table = _validated_table(config, table_name)
    table.drop(engine, checkfirst=True)
    with engine.begin() as con:
        con.execute(
            text(f"DELETE FROM {REGISTRY_TABLE} WHERE id=:id"),
            {"id": int(dataset_id)},
        )
    return True
