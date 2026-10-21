import sqlite3
from pathlib import Path

import pandas as pd

from cpt.config import PROJECT_ROOT

SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"


def connect(path: Path | str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_PATH.read_text())
    conn.commit()


def upsert(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> int:
    """INSERT OR REPLACE rows; relies on the table's primary key to avoid duplicates."""
    cols = list(df.columns)
    placeholders = ", ".join("?" for _ in cols)
    sql = f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
    rows = df.astype(object).where(df.notna(), None).itertuples(index=False, name=None)
    conn.executemany(sql, rows)
    conn.commit()
    return len(df)


def load_adj_close(conn: sqlite3.Connection, tickers: list[str]) -> pd.DataFrame:
    """Wide frame of adjusted closes, inner-joined on dates where all tickers trade."""
    placeholders = ", ".join("?" for _ in tickers)
    df = pd.read_sql(
        f"SELECT date, ticker, adj_close FROM prices WHERE ticker IN ({placeholders})",
        conn,
        params=tickers,
        parse_dates=["date"],
    )
    return df.pivot(index="date", columns="ticker", values="adj_close")[tickers].dropna()
