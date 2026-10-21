import io
import zipfile
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import yfinance as yf


def download_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Daily prices from yfinance. `end` is inclusive."""
    end_exclusive = (pd.Timestamp(end) + timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.Ticker(ticker).history(start=start, end=end_exclusive, auto_adjust=False)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")
    return pd.DataFrame({
        "ticker": ticker,
        "date": raw.index.strftime("%Y-%m-%d"),
        "close": raw["Close"].to_numpy(),
        "adj_close": raw["Adj Close"].to_numpy(),
        "volume": raw["Volume"].astype("int64").to_numpy(),
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })


def parse_french_daily(text: str) -> pd.DataFrame:
    """Parse the Fama/French daily factors CSV; keep only the RF column, in decimals."""
    rows = []
    for line in text.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) >= 5 and len(fields[0]) == 8 and fields[0].isdigit():
            rows.append((fields[0], float(fields[4])))
    df = pd.DataFrame(rows, columns=["yyyymmdd", "rf_pct"])
    return pd.DataFrame({
        "date": pd.to_datetime(df["yyyymmdd"], format="%Y%m%d").dt.strftime("%Y-%m-%d"),
        "rf_daily": df["rf_pct"] / 100.0,
    })


def download_risk_free(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=(10, 300))
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        text = zf.read(zf.namelist()[0]).decode("utf-8", errors="replace")
    return parse_french_daily(text)
