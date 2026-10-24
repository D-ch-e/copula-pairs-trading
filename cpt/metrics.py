"""Performance and risk metrics on daily strategy returns.

Conventions: simple daily returns, 252 trading days per year, VaR/CVaR reported
as return quantiles (negative numbers = losses), as in the paper's tables.
"""
import numpy as np
import pandas as pd

DAYS_PER_YEAR = 252


def _excess(returns: pd.Series, rf: pd.Series | None) -> pd.Series:
    if rf is None:
        return returns
    return returns - rf.reindex(returns.index).fillna(0.0)


def annualized_return(returns: pd.Series) -> float:
    growth = (1 + returns).prod()
    return growth ** (DAYS_PER_YEAR / len(returns)) - 1


def annualized_volatility(returns: pd.Series) -> float:
    return returns.std(ddof=1) * np.sqrt(DAYS_PER_YEAR)


def sharpe_ratio(returns: pd.Series, rf: pd.Series | None = None) -> float:
    ex = _excess(returns, rf)
    std = ex.std(ddof=1)
    return np.nan if std == 0 else ex.mean() / std * np.sqrt(DAYS_PER_YEAR)


def sortino_ratio(returns: pd.Series, rf: pd.Series | None = None) -> float:
    ex = _excess(returns, rf)
    downside = np.sqrt((np.minimum(ex, 0) ** 2).mean())
    return np.nan if downside == 0 else ex.mean() / downside * np.sqrt(DAYS_PER_YEAR)


def drawdown(returns: pd.Series) -> pd.Series:
    wealth = (1 + returns).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough loss, as a positive number."""
    return -drawdown(returns).min()


def historical_var_cvar(returns: pd.Series, level: float) -> tuple[float, float]:
    """Historical VaR and CVaR at confidence `level` (e.g. 0.95)."""
    var = returns.quantile(1 - level)
    return var, returns[returns <= var].mean()


def performance_summary(returns: pd.Series, rf: pd.Series | None = None) -> dict:
    ann_ret = annualized_return(returns)
    mdd = max_drawdown(returns)
    var95, cvar95 = historical_var_cvar(returns, 0.95)
    var99, cvar99 = historical_var_cvar(returns, 0.99)
    return {
        "ann_return": ann_ret,
        "ann_vol": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns, rf),
        "sortino": sortino_ratio(returns, rf),
        "max_drawdown": mdd,
        "calmar": np.nan if mdd == 0 else ann_ret / mdd,
        "var_95": var95, "cvar_95": cvar95,
        "var_99": var99, "cvar_99": cvar99,
        "days": len(returns),
    }
