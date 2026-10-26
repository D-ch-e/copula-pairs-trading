import pandas as pd

from cpt.db import connect, init_db, upsert
from cpt.manifest import build_manifest, compare_manifest


def _db(tmp_path, gld_prices):
    conn = connect(tmp_path / "m.sqlite")
    init_db(conn)
    upsert(conn, "prices", pd.DataFrame({
        "ticker": "GLD", "date": ["2020-01-02", "2020-01-03"], "close": gld_prices,
        "adj_close": gld_prices, "volume": 1, "downloaded_at": "2026-01-01T00:00:00+00:00"}))
    upsert(conn, "risk_free", pd.DataFrame({"date": ["2020-01-02"], "rf_daily": [0.0001]}))
    return conn


def test_manifest_is_stable_for_same_data(tmp_path):
    a = build_manifest(_db(tmp_path / "a", [100.0, 101.0]))
    b = build_manifest(_db(tmp_path / "b", [100.0, 101.0]))
    assert compare_manifest(a, b) == []
    assert a["GLD"]["rows"] == 2 and a["GLD"]["end"] == "2020-01-03"


def test_manifest_detects_revised_price(tmp_path):
    a = build_manifest(_db(tmp_path / "a", [100.0, 101.0]))
    b = build_manifest(_db(tmp_path / "b", [100.0, 101.5]))
    problems = compare_manifest(a, b)
    assert len(problems) == 1 and problems[0].startswith("GLD: sha256")


def test_manifest_detects_missing_series(tmp_path):
    a = build_manifest(_db(tmp_path / "a", [100.0, 101.0]))
    b = {k: v for k, v in a.items() if k != "GLD"}
    assert compare_manifest(a, b) == ["GLD: missing from local data"]
