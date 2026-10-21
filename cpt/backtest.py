"""Walk-forward backtest: refit monthly on a 252-day window, trade the next month.

Rules to respect:
- signal at close t, execute at close t + execution_lag_days
- one open position per pair, $1 long / $1 short
- exit on h back to exit_band, max_holding_days, or stop_loss
- round-trip cost charged per pair trade
"""
import pandas as pd


def run_backtest(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    raise NotImplementedError
