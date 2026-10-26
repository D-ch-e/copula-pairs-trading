"""Data manifest: row counts, date ranges and checksums of the downloaded data.

Prices are not committed (Yahoo data may not be redistributed), so the manifest
is what lets someone else check that a fresh download matches the one behind the
published results. yfinance revises history, so a mismatch is expected over time
and means results may differ.
"""
import hashlib
import json
import sqlite3
from pathlib import Path

import pandas as pd

COMPARED_FIELDS = ["rows", "start", "end", "sha256"]


def _digest(dates: pd.Series, values: pd.Series) -> str:
    lines = (dates + "," + values.map(lambda v: f"{v:.10g}")).str.cat(sep="\n")
    return hashlib.sha256(lines.encode()).hexdigest()


def build_manifest(conn: sqlite3.Connection) -> dict:
    prices = pd.read_sql("SELECT ticker, date, adj_close, downloaded_at FROM prices ORDER BY ticker, date", conn)
    entries = {}
    for ticker, df in prices.groupby("ticker"):
        entries[ticker] = {
            "rows": len(df), "start": df["date"].iloc[0], "end": df["date"].iloc[-1],
            "sha256": _digest(df["date"], df["adj_close"]), "downloaded_at": df["downloaded_at"].max(),
        }
    rf = pd.read_sql("SELECT date, rf_daily FROM risk_free ORDER BY date", conn)
    entries["risk_free"] = {
        "rows": len(rf), "start": rf["date"].iloc[0], "end": rf["date"].iloc[-1],
        "sha256": _digest(rf["date"], rf["rf_daily"]),
    }
    return entries


def write_manifest(entries: dict, path: Path) -> None:
    path.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")


def compare_manifest(expected: dict, actual: dict) -> list[str]:
    """Human-readable list of differences; empty if the data matches."""
    problems = []
    for name in sorted(set(expected) | set(actual)):
        if name not in actual:
            problems.append(f"{name}: missing from local data")
        elif name not in expected:
            problems.append(f"{name}: not in manifest")
        else:
            for field in COMPARED_FIELDS:
                if expected[name][field] != actual[name][field]:
                    problems.append(f"{name}: {field} {expected[name][field]} -> {actual[name][field]}")
    return problems
