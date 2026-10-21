import pandas as pd

from cpt.db import connect, init_db, load_adj_close, upsert


def _prices(ticker, dates, values):
    return pd.DataFrame({
        "ticker": ticker,
        "date": dates,
        "close": values,
        "adj_close": values,
        "volume": 100,
        "downloaded_at": "2026-01-01T00:00:00+00:00",
    })


def test_upsert_replaces_instead_of_duplicating(tmp_path):
    conn = connect(tmp_path / "t.sqlite")
    init_db(conn)
    upsert(conn, "prices", _prices("GLD", ["2020-01-02"], [1.0]))
    upsert(conn, "prices", _prices("GLD", ["2020-01-02"], [2.0]))
    rows = conn.execute("SELECT adj_close FROM prices").fetchall()
    assert rows == [(2.0,)]


def test_load_adj_close_inner_joins_dates(tmp_path):
    conn = connect(tmp_path / "t.sqlite")
    init_db(conn)
    upsert(conn, "prices", _prices("GLD", ["2020-01-02", "2020-01-03"], [1.0, 1.1]))
    upsert(conn, "prices", _prices("SLV", ["2020-01-03"], [2.0]))
    wide = load_adj_close(conn, ["GLD", "SLV"])
    assert list(wide.columns) == ["GLD", "SLV"]
    assert len(wide) == 1
    assert wide.iloc[0].tolist() == [1.1, 2.0]
