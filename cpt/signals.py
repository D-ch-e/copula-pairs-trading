"""Entry signals from conditional probabilities (Krauss & Stübinger 2015, sec. 3.2.2).

h1 = h(u1|u2) = P(U1 <= u1 | U2 = u2), h2 = h(u2|u1) = P(U2 <= u2 | U1 = u1).

Region A: h1 < alpha and h2 > 1 - alpha -> leg 1 undervalued: long leg 1, short leg 2 (+1)
Region B: h1 > 1 - alpha and h2 < alpha -> leg 1 overvalued: short leg 1, long leg 2 (-1)
Region C: otherwise -> no trade (0)
"""
import numpy as np
import pandas as pd


def classify_regions(h1: pd.Series, h2: pd.Series, alpha: float) -> pd.Series:
    region_a = (h1 < alpha) & (h2 > 1 - alpha)
    region_b = (h1 > 1 - alpha) & (h2 < alpha)
    return pd.Series(np.where(region_a, 1, np.where(region_b, -1, 0)), index=h1.index, name="signal")
