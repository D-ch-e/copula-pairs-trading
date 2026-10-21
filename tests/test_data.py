import pytest

from cpt.config import load_config
from cpt.data import parse_french_daily

SAMPLE = """This file was created by CMPT_ME_BEME_RETS_DAILY using the 202508 CRSP database.

,Mkt-RF,SMB,HML,RF
19260701,    0.09,   -0.25,   -0.27,    0.01
19260702,    0.45,   -0.33,   -0.06,    0.01

Copyright 2025 Eugene F. Fama and Kenneth R. French
"""


def test_parse_french_daily_keeps_rf_in_decimals():
    df = parse_french_daily(SAMPLE)
    assert df["date"].tolist() == ["1926-07-01", "1926-07-02"]
    assert df["rf_daily"].tolist() == pytest.approx([0.0001, 0.0001])


def test_config_has_main_pair_in_tickers():
    cfg = load_config()
    assert set(cfg["pairs"]["main"]) <= set(cfg["data"]["tickers"])
    for pair in cfg["pairs"]["robustness"]:
        assert set(pair) <= set(cfg["data"]["tickers"])
