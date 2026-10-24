"""Walk-forward backtest for one pair.

Each month: fit Gaussian and t copulas on the previous `estimation_window_days`
log returns, map that month's returns through the training ECDFs, and compute
h1 = h(u1|u2), h2 = h(u2|u1) with the chosen family.

Trading rules:
- signal at close t, order executed at close t + execution_lag_days
- one open position at a time, $1 long / $1 short (returns are per $1 per leg)
- exit when h1 and h2 are both back across exit_band, the trade return hits
  take_profit or stop_loss, or the position is max_holding_days old
- round-trip cost charged half at entry, half at exit
"""
import numpy as np
import pandas as pd

from cpt import copula
from cpt.signals import classify_regions

H_FUNCTIONS = {
    "gaussian": lambda u, v, fit: copula.h_gaussian(u, v, fit["rho"]),
    "t": lambda u, v, fit: copula.h_t(u, v, fit["rho"], fit["nu"]),
}


def rolling_conditional_probs(
    log_returns: pd.DataFrame, window: int, family: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly refit; returns daily (u1, u2, h1, h2) and one row of fit stats per month."""
    r = log_returns.dropna()
    months = r.index.to_period("M")
    daily, fits = [], []
    for month in months.unique():
        in_month = months == month
        first = int(np.argmax(in_month))
        if first < window:
            continue
        train, test = r.iloc[first - window:first], r[in_month]
        tr1, tr2 = train.iloc[:, 0].to_numpy(), train.iloc[:, 1].to_numpy()
        p1, p2 = copula.pseudo_observations(tr1), copula.pseudo_observations(tr2)
        u1 = copula.ecdf_transform(tr1, test.iloc[:, 0].to_numpy())
        u2 = copula.ecdf_transform(tr2, test.iloc[:, 1].to_numpy())

        g, t = copula.fit_gaussian(p1, p2), copula.fit_t(p1, p2)
        fits.append({
            "month": str(month),
            "gaussian_rho": g["rho"], "gaussian_aic": g["aic"],
            "t_rho": t["rho"], "t_nu": t["nu"], "t_aic": t["aic"],
            "t_tail_dep": copula.tail_dependence_t(t["rho"], t["nu"]),
            "gaussian_oos_loglik": copula.logpdf_gaussian(u1, u2, g["rho"]).sum(),
            "t_oos_loglik": copula.logpdf_t(u1, u2, t["rho"], t["nu"]).sum(),
        })

        fit = g if family == "gaussian" else t
        h = H_FUNCTIONS[family]
        daily.append(pd.DataFrame(
            {"u1": u1, "u2": u2, "h1": h(u1, u2, fit), "h2": h(u2, u1, fit)}, index=test.index
        ))
    return pd.concat(daily), pd.DataFrame(fits)


def _exit_reason(pos, trade_ret, held, h1, h2, s) -> str | None:
    if trade_ret <= s["stop_loss"]:
        return "stop_loss"
    if s.get("take_profit") is not None and trade_ret >= s["take_profit"]:
        return "take_profit"
    band = s.get("exit_band")
    if band is not None and pos * (h1 - band) >= 0 and pos * (h2 - band) <= 0:
        return "reversion"
    if held >= s["max_holding_days"]:
        return "max_holding"
    return None


def simulate(prices: pd.DataFrame, probs: pd.DataFrame, s: dict, cost_bps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily state machine over the trading days in `probs`."""
    px = prices.loc[probs.index]
    p1, p2 = px.iloc[:, 0].to_numpy(), px.iloc[:, 1].to_numpy()
    h1, h2 = probs["h1"].to_numpy(), probs["h2"].to_numpy()
    signal = classify_regions(probs["h1"], probs["h2"], s["alpha"]).to_numpy()
    lag, half_cost = s["execution_lag_days"], cost_bps / 1e4 / 2
    dates = probs.index
    n = len(dates)

    ret, held_pos = np.zeros(n), np.zeros(n, dtype=int)
    trades = []
    pos, entry_idx, e1, e2, trade_val = 0, None, None, None, 0.0
    pending = None  # (exec_idx, action, direction, decided_idx, reason)

    def execute(i, order):
        nonlocal pos, entry_idx, e1, e2, trade_val
        _, action, direction, decided_idx, reason = order
        ret[i] -= half_cost
        if action == "enter":
            pos, entry_idx, e1, e2, trade_val = direction, i, p1[i], p2[i], 0.0
            trades.append({"signal_date": dates[decided_idx], "entry_date": dates[i], "direction": direction})
        else:
            trades[-1].update({
                "exit_date": dates[i], "exit_reason": reason, "holding_days": i - entry_idx,
                "gross_return": trade_val, "net_return": trade_val - 2 * half_cost,
            })
            pos = 0

    for i in range(n):
        if pos != 0:
            value = pos * (p1[i] / e1 - p2[i] / e2)
            ret[i] += value - trade_val
            trade_val = value

        if pending is not None and pending[0] == i:
            execute(i, pending)
            pending = None

        if pending is None:
            if pos == 0 and signal[i] != 0:
                pending = (i + lag, "enter", int(signal[i]), i, None)
            elif pos != 0:
                reason = _exit_reason(pos, trade_val, i - entry_idx, h1[i], h2[i], s)
                if reason is not None:
                    pending = (i + lag, "exit", 0, i, reason)
            if pending is not None and pending[0] == i:
                execute(i, pending)
                pending = None

        if i == n - 1 and pos != 0:
            execute(i, (i, "exit", 0, i, "end_of_data"))
        held_pos[i] = pos

    daily = probs.assign(signal=signal, position=held_pos, ret=ret)
    return daily, pd.DataFrame(trades)


def run_backtest(
    prices: pd.DataFrame, cfg: dict, family: str | None = None, cost_bps: float | None = None
) -> dict:
    """prices: two columns (leg 1, leg 2) of adjusted closes indexed by date."""
    s = cfg["strategy"]
    family = family or s["family"]
    cost_bps = cfg["costs"]["round_trip_bps"] if cost_bps is None else cost_bps
    log_returns = np.log(prices).diff()
    probs, fits = rolling_conditional_probs(log_returns, s["estimation_window_days"], family)
    daily, trades = simulate(prices, probs, s, cost_bps)
    return {"daily": daily, "trades": trades, "fits": fits}
