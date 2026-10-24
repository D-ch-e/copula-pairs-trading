# Copula-Based Pairs Trading: Gold vs Silver

Backtest of a copula-based statistical arbitrage strategy on GLD/SLV, adapting the
framework of Krauss & Stübinger (2015), *Nonlinear dependence modeling with bivariate
copulas: Statistical arbitrage pairs trading on the S&P 100* (IWQW Discussion Paper 15/2015).

> Research project for educational purposes only. Not investment advice.
> Historical backtests do not guarantee future results.

## Paper vs. this project

| Paper (cited) | My implementation | My extension |
|---|---|---|
| t-copula, semiparametric fit, conditional-probability signals (α = 5%) | Same signal rule applied to pre-selected pairs (GLD/SLV + robustness pairs) | Gaussian vs t-copula comparison (AIC, out-of-sample log-likelihood) |
| S&P 100, 1990–2014, copula-based pair selection | Free data (yfinance), 2006–2025, no pair selection | z-score spread baseline |
| 50 bps round-trip cost, one-day waiting rule | Same, as main specification | Cost sensitivity and break-even cost |

## Setup

The virtualenv lives outside the project: this folder sits on an iCloud-synced Desktop,
and iCloud can evict venv files, which makes Python hang at startup.

```bash
python3.12 -m venv ~/.venvs/copula-pairs-trading
source ~/.venvs/copula-pairs-trading/bin/activate
pip install -r requirements.txt
python -m scripts.init_db
python -m scripts.download_data
python -m scripts.run_experiments   # all specs x pairs x costs -> SQLite result tables
pytest
```

## Layout

- `config.yaml` — all parameters (main specification)
- `sql/schema.sql` — SQLite schema
- `cpt/` — library code (data, db, copula, signals, backtest, metrics)
- `scripts/` — runnable entry points
- `notebooks/` — analysis
- `reports/figures/` — output charts

## Limitations

To be documented: single-pair statistical power, ex-ante pair choice, close-to-close
execution, short-selling assumptions, data snooping across parameter variants.
