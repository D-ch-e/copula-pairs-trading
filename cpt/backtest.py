"""Walk-forward backtest for one pair.

Every month the model is refit on the previous `estimation_window_days` of data
and applied to that month only. Two signal models share one trade simulator:

- copula: fit Gaussian and t copulas on log returns, map the month's returns
  through the training ECDFs, compute h1 = h(u1|u2), h2 = h(u2|u1)
- zscore (baseline): OLS hedge ratio on log prices, spread z-score

Trading rules (simulate):
- signal at close t, order executed at close t + execution_lag_days
- one open position at a time; capital is $1 of NAV and each leg has notional
  gross_exposure / 2 (default 2.0: $1 long / $1 short, the paper's convention)
- exit on reversion, take-profit, stop-loss or max_holding_days; take-profit and
  stop-loss are returns per $1 of leg notional, so exits do not depend on exposure
- round-trip cost is quoted per pair at $1 per leg and scaled by leg notional,
  charged half at entry, half at exit
"""
import numpy as np
import pandas as pd

from cpt import copula
from cpt.signals import classify_regions

H_FUNCTIONS = {
    "gaussian": lambda u, v, fit: copula.h_gaussian(u, v, fit["rho"]),
    "t": lambda u, v, fit: copula.h_t(u, v, fit["rho"], fit["nu"]),
}


def _monthly_windows(index: pd.DatetimeIndex, window: int):
    """Yield (month, train positions, test mask) for months with a full training window."""
    months = index.to_period("M")
    for month in months.unique():
        in_month = months == month
        first = int(np.argmax(in_month))
        if first >= window:
            yield month, slice(first - window, first), in_month


def paper_quantile_thresholds(
    p1: np.ndarray, p2: np.ndarray, signal: np.ndarray, horizon: int, tp_q: float, sl_q: float
) -> tuple[float, float, int]:
    """Krauss & Stübinger (2015) exit levels from in-window cumulative return paths.

    For every signal j whose full `horizon`-day path lies inside the window, the
    path is CR_k = d * (p1[j+k]/p1[j] - p2[j+k]/p2[j]), k = 1..horizon.
    TP = mean of the paths' tp_q quantiles, SL = mean of their sl_q quantiles.
    """
    tps, sls = [], []
    for j in np.flatnonzero(signal[: len(signal) - horizon]):
        k = np.arange(j + 1, j + horizon + 1)
        path = signal[j] * (p1[k] / p1[j] - p2[k] / p2[j])
        tps.append(np.quantile(path, tp_q))
        sls.append(np.quantile(path, sl_q))
    if not tps:
        return np.nan, np.nan, 0
    return float(np.mean(tps)), float(np.mean(sls)), len(tps)


def copula_frame(prices: pd.DataFrame, s: dict, family: str, exit_rule: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily signal/exit frame for the copula strategy plus one row of fit stats per month."""
    r = np.log(prices).diff().dropna()
    px = prices.loc[r.index]
    alpha, band = s["alpha"], s.get("exit_band")
    frames, fits = [], []
    for month, train_pos, in_month in _monthly_windows(r.index, s["estimation_window_days"]):
        train, test = r.iloc[train_pos], r[in_month]
        tr1, tr2 = train.iloc[:, 0].to_numpy(), train.iloc[:, 1].to_numpy()
        p1, p2 = copula.pseudo_observations(tr1), copula.pseudo_observations(tr2)
        u1 = copula.ecdf_transform(tr1, test.iloc[:, 0].to_numpy())
        u2 = copula.ecdf_transform(tr2, test.iloc[:, 1].to_numpy())

        g, t = copula.fit_gaussian(p1, p2), copula.fit_t(p1, p2)
        fit = g if family == "gaussian" else t
        h = H_FUNCTIONS[family]
        h1, h2 = h(u1, u2, fit), h(u2, u1, fit)

        tp, sl = s.get("take_profit"), s["stop_loss"]
        n_paths = None
        if exit_rule == "paper_quantile":
            q = s["paper_quantile"]
            train_signal = classify_regions(pd.Series(h(p1, p2, fit)), pd.Series(h(p2, p1, fit)), alpha).to_numpy()
            tp, sl, n_paths = paper_quantile_thresholds(
                px.iloc[train_pos, 0].to_numpy(), px.iloc[train_pos, 1].to_numpy(), train_signal,
                s["max_holding_days"], q["take_profit_quantile"], q["stop_loss_quantile"],
            )
            if n_paths == 0:
                tp, sl = s.get("take_profit"), s["stop_loss"]

        fits.append({
            "month": str(month),
            "gaussian_rho": g["rho"], "gaussian_aic": g["aic"],
            "t_rho": t["rho"], "t_nu": t["nu"], "t_aic": t["aic"],
            "t_tail_dep": copula.tail_dependence_t(t["rho"], t["nu"]),
            "t_converged": t["converged"], "t_nu_at_bound": t["nu_at_bound"], "t_rho_at_bound": t["rho_at_bound"],
            "gaussian_oos_loglik": copula.logpdf_gaussian(u1, u2, g["rho"]).sum(),
            "t_oos_loglik": copula.logpdf_t(u1, u2, t["rho"], t["nu"]).sum(),
            "take_profit": tp, "stop_loss": sl, "threshold_paths": n_paths,
        })

        use_band = exit_rule == "probability" and band is not None
        frames.append(pd.DataFrame({
            "u1": u1, "u2": u2, "h1": h1, "h2": h2,
            "signal": classify_regions(pd.Series(h1), pd.Series(h2), alpha).to_numpy(),
            "close_long": use_band & (h1 >= band) & (h2 <= band),
            "close_short": use_band & (h1 <= band) & (h2 >= band),
            "take_profit": np.nan if tp is None else tp,
            "stop_loss": sl,
        }, index=test.index))
    return pd.concat(frames), pd.DataFrame(fits)


def zscore_frame(prices: pd.DataFrame, s: dict, z_cfg: dict) -> pd.DataFrame:
    """Baseline: spread = log p1 - beta * log p2 with beta, mean, std from the training window."""
    logp = np.log(prices).iloc[1:]  # same trading days as the copula strategy
    frames = []
    for _, train_pos, in_month in _monthly_windows(logp.index, s["estimation_window_days"]):
        train, test = logp.iloc[train_pos], logp[in_month]
        beta, _ = np.polyfit(train.iloc[:, 1], train.iloc[:, 0], 1)
        spread_train = train.iloc[:, 0] - beta * train.iloc[:, 1]
        z = ((test.iloc[:, 0] - beta * test.iloc[:, 1]) - spread_train.mean()) / spread_train.std(ddof=1)
        frames.append(pd.DataFrame({
            "z": z, "beta": beta,
            "signal": np.where(z < -z_cfg["entry"], 1, np.where(z > z_cfg["entry"], -1, 0)),
            "close_long": z >= z_cfg["exit"],
            "close_short": z <= z_cfg["exit"],
            "take_profit": np.nan if s.get("take_profit") is None else s["take_profit"],
            "stop_loss": s["stop_loss"],
        }, index=test.index))
    return pd.concat(frames)


def _exit_reason(pos, trade_ret, held, closes, tp, sl, max_hold) -> str | None:
    if trade_ret <= sl:
        return "stop_loss"
    if not np.isnan(tp) and trade_ret >= tp:
        return "take_profit"
    if closes:
        return "reversion"
    if held >= max_hold:
        return "max_holding"
    return None


def simulate(prices: pd.DataFrame, frame: pd.DataFrame, s: dict, cost_bps: float,
             gross_exposure: float = 2.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily state machine over the trading days in `frame`; returns are on $1 of NAV.

    `frame` needs: signal (+1/-1/0), close_long, close_short (bool), take_profit, stop_loss.
    Exit levels are fixed at the values of the entry day.
    """
    px = prices.loc[frame.index]
    p1, p2 = px.iloc[:, 0].to_numpy(), px.iloc[:, 1].to_numpy()
    signal = frame["signal"].to_numpy()
    close_long, close_short = frame["close_long"].to_numpy(), frame["close_short"].to_numpy()
    tps, sls = frame["take_profit"].to_numpy(dtype=float), frame["stop_loss"].to_numpy(dtype=float)
    leg = gross_exposure / 2
    lag, half_cost, max_hold = s["execution_lag_days"], cost_bps / 1e4 / 2 * leg, s["max_holding_days"]
    dates = frame.index
    n = len(dates)

    ret, held_pos = np.zeros(n), np.zeros(n, dtype=int)
    trades = []
    pos, entry_idx, e1, e2, trade_val, tp, sl = 0, None, None, None, 0.0, np.nan, np.nan
    pending = None  # (exec_idx, action, direction, decided_idx, reason)

    def execute(i, order):
        nonlocal pos, entry_idx, e1, e2, trade_val, tp, sl
        _, action, direction, decided_idx, reason = order
        ret[i] -= half_cost
        if action == "enter":
            pos, entry_idx, e1, e2, trade_val = direction, i, p1[i], p2[i], 0.0
            tp, sl = tps[i], sls[i]
            trades.append({"signal_date": dates[decided_idx], "entry_date": dates[i], "direction": direction,
                           "take_profit": tp, "stop_loss": sl})
        else:
            trades[-1].update({
                "exit_date": dates[i], "exit_reason": reason, "holding_days": i - entry_idx,
                "gross_return": trade_val, "net_return": trade_val - 2 * half_cost, "leg_notional": leg,
            })
            pos = 0

    for i in range(n):
        if pos != 0:
            value = leg * pos * (p1[i] / e1 - p2[i] / e2)
            ret[i] += value - trade_val
            trade_val = value

        if pending is not None and pending[0] == i:
            execute(i, pending)
            pending = None

        if pending is None:
            if pos == 0 and signal[i] != 0:
                pending = (i + lag, "enter", int(signal[i]), i, None)
            elif pos != 0:
                closes = close_long[i] if pos > 0 else close_short[i]
                reason = _exit_reason(pos, trade_val / leg, i - entry_idx, closes, tp, sl, max_hold)
                if reason is not None:
                    pending = (i + lag, "exit", 0, i, reason)
            if pending is not None and pending[0] == i:
                execute(i, pending)
                pending = None

        if i == n - 1 and pos != 0:
            execute(i, (i, "exit", 0, i, "end_of_data"))
        held_pos[i] = pos

    daily = frame.assign(position=held_pos, ret=ret)
    return daily, pd.DataFrame(trades)


def run_backtest(
    prices: pd.DataFrame,
    cfg: dict,
    family: str | None = None,
    cost_bps: float | None = None,
    model: str = "copula",
    exit_rule: str | None = None,
    gross_exposure: float | None = None,
) -> dict:
    """prices: two columns (leg 1, leg 2) of adjusted closes indexed by date.

    model: "copula" or "zscore". exit_rule (copula only): "probability" or "paper_quantile".
    """
    s = {**cfg["strategy"], "paper_quantile": cfg.get("paper_quantile")}
    cost_bps = cfg["costs"]["round_trip_bps"] if cost_bps is None else cost_bps
    gross_exposure = cfg["capital"]["gross_exposure"] if gross_exposure is None else gross_exposure
    fits = None
    if model == "zscore":
        frame = zscore_frame(prices, s, cfg["zscore"])
    else:
        frame, fits = copula_frame(prices, s, family or s["family"], exit_rule or s["exit_rule"])
    daily, trades = simulate(prices, frame, s, cost_bps, gross_exposure)
    return {"daily": daily, "trades": trades, "fits": fits}
