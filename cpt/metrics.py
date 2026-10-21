"""Performance and risk metrics on daily strategy returns."""
import pandas as pd


def max_drawdown(returns: pd.Series) -> float:
    raise NotImplementedError


def sharpe_ratio(returns: pd.Series, rf: pd.Series) -> float:
    raise NotImplementedError


def sortino_ratio(returns: pd.Series, rf: pd.Series) -> float:
    raise NotImplementedError


def historical_var_cvar(returns: pd.Series, level: float) -> tuple[float, float]:
    raise NotImplementedError
