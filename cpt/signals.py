"""Entry/exit signals from conditional probabilities.

Region A: h(u1|u2) < alpha and h(u2|u1) > 1 - alpha -> long leg 1, short leg 2
Region B: h(u1|u2) > 1 - alpha and h(u2|u1) < alpha -> short leg 1, long leg 2
Region C: otherwise -> no trade
"""
import pandas as pd


def classify_regions(h1: pd.Series, h2: pd.Series, alpha: float) -> pd.Series:
    raise NotImplementedError
