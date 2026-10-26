# Tableau export: data dictionary

Produced by `python -m scripts.export_tableau` into `exports/tableau/`. All returns are simple
returns in decimals (0.01 = 1%). Strategy returns are on $1 of NAV; `gross_exposure` 2.0 means
$1 long + $1 short (the paper's convention), 1.0 means $0.5 + $0.5. Benchmarks (GLD, SPY,
50/50 GLD+SLV) are $1 long-only and, because the data is published, appear **only at monthly
grain** (and as aggregates in `runs`, `subperiods`, `significance`).

Join key for the first six tables: `run_id` = `pair|model|family|exit_rule|cost_bps[|gross1]`
for strategies, `pair|buy_hold|<benchmark>` for benchmarks.

## runs.csv — one row per run

| Field | Meaning |
|---|---|
| run_id, pair | Run key; pair as `LEG1-LEG2` |
| spec | Display name: Copula t (main), Copula Gaussian, Copula t, paper exit, z-score baseline, GLD, SPY, 50/50 GLD+SLV |
| model, family, exit_rule | copula / zscore / buy_hold; t / gaussian; probability / paper_quantile |
| cost_bps | Round-trip cost per pair trade, quoted at $1 per leg |
| gross_exposure | 2.0 or 1.0 (strategies); 1.0 for benchmarks |
| is_main_spec | True for GLD-SLV, Copula t (main), 50 bps, 200% |
| is_benchmark | True for buy-and-hold rows |
| ann_return, ann_vol | Annualized (252 days), from daily returns |
| sharpe, sortino | Annualized, net of the daily Fama/French risk-free rate |
| max_drawdown | Largest peak-to-trough loss from daily NAV, starting NAV counted (positive number) |
| calmar | ann_return / max_drawdown |
| var_95, cvar_95, var_99, cvar_99 | Daily historical VaR / CVaR as return quantiles (negative = loss) |
| n_trades, avg_holding_days, win_rate | Trade statistics (strategies only) |
| breakeven_bps | Round-trip cost at which the average trade nets zero; negative = loses before costs |
| days | Number of daily returns |

## strategy_daily.csv — run × trading day (strategies only)

| Field | Meaning |
|---|---|
| run_id, date | Key |
| ret | Daily return on NAV, net of the run's cost |
| wealth | Growth of $1 (cumulative product of 1 + ret) |
| drawdown | wealth / running peak − 1, starting NAV of 1 counted as a peak (≤ 0) |

## monthly_returns.csv — run × month (strategies and benchmarks)

| Field | Meaning |
|---|---|
| run_id, month | Key; month = first day of the month |
| ret | Monthly return, compounded from daily returns |
| wealth, drawdown | As above, at monthly grain (monthly drawdown understates the daily maximum) |
| pair, spec, cost_bps, gross_exposure, is_benchmark | Copied from runs for convenience |

## trades.csv — one row per trade

| Field | Meaning |
|---|---|
| run_id, trade_no | Key |
| signal_date, entry_date, exit_date | Signal at close, executed at the next close (one-day lag) |
| direction | +1 = long leg 1 / short leg 2; −1 = the reverse |
| exit_reason | reversion, take_profit, stop_loss, max_holding, end_of_data |
| holding_days | Trading days between entry and exit |
| gross_return, net_return | Trade return on NAV before / after cost |
| take_profit, stop_loss | Exit levels in effect (per $1 of leg notional) |
| leg_notional | gross_exposure / 2 |

## subperiods.csv — run × period

| Field | Meaning |
|---|---|
| period, period_label, period_order | gfc_2008, silver_2011, covid_2020, rates_2022, other; label and sort order |
| days | Trading days in the period |
| total_return | Compounded return over the period (not annualized) |
| sharpe, max_drawdown | Within the period (Sharpe on raw returns) |
| run_id, pair, spec, model, cost_bps, gross_exposure, is_benchmark | Run attributes |

## significance.csv — one row per run

| Field | Meaning |
|---|---|
| ann_mean | Arithmetic mean daily return × 252 |
| t_nw, p_nw | Newey-West t-statistic and p-value for mean = 0 |
| sharpe_raw, sharpe_ci_low, sharpe_ci_high | Sharpe on raw returns and its 95% stationary block-bootstrap interval (block 20 days, 2,000 resamples) |
| run attributes | As in runs |

## copula_fits.csv — pair × month (main-spec t-copula and Gaussian fits)

| Field | Meaning |
|---|---|
| pair, month | Key; month = first day of the month the fit is applied to (fit uses the preceding 252 trading days) |
| gaussian_rho, gaussian_aic | Gaussian copula fit |
| t_rho, t_nu, t_aic | t-copula fit; ν bounded to [2.01, 200] |
| t_tail_dep | Implied tail dependence λ of the t-copula |
| gaussian_oos_loglik, t_oos_loglik | Log-likelihood of the following month's data |
| t_converged, t_nu_at_bound, t_rho_at_bound | Optimizer diagnostics (1 = true) |

## tail_calibration.csv — pair × version × measure

| version | measures | Meaning |
|---|---|---|
| Raw | observed_lower, observed_upper, pred_gaussian, pred_t | Share of out-of-sample days with both legs in their 5% tail, observed and predicted |
| Marginal drift removed | same | After re-ranking test-period values so each leg is exactly uniform (diagnostic only) |
| Marginal exceedance | u1_lower, u1_upper, u2_lower, u2_upper | Share of days each leg alone is in its 5% tail (nominal 5%) |
