-- Dates stored as ISO text 'YYYY-MM-DD'

CREATE TABLE IF NOT EXISTS prices (
    ticker        TEXT NOT NULL,
    date          TEXT NOT NULL,
    close         REAL,
    adj_close     REAL,
    volume        INTEGER,
    downloaded_at TEXT NOT NULL,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS risk_free (
    date     TEXT PRIMARY KEY,
    rf_daily REAL NOT NULL          -- decimal, already divided by 100
);

-- Backtest results. run_id = pair|model|family|exit_rule|cost_bps

CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id     TEXT PRIMARY KEY,
    pair       TEXT NOT NULL,
    model      TEXT NOT NULL,     -- copula, zscore, buy_hold
    family     TEXT,              -- t, gaussian (copula only)
    exit_rule  TEXT,              -- probability, paper_quantile (copula only)
    cost_bps   REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_metrics (
    run_id TEXT NOT NULL,
    metric TEXT NOT NULL,
    value  REAL,
    PRIMARY KEY (run_id, metric)
);

CREATE TABLE IF NOT EXISTS strategy_returns (
    run_id   TEXT NOT NULL,
    date     TEXT NOT NULL,
    ret      REAL NOT NULL,
    position INTEGER,
    PRIMARY KEY (run_id, date)
);

CREATE TABLE IF NOT EXISTS trades (
    run_id       TEXT NOT NULL,
    trade_no     INTEGER NOT NULL,
    signal_date  TEXT,
    entry_date   TEXT,
    exit_date    TEXT,
    direction    INTEGER,
    exit_reason  TEXT,
    holding_days INTEGER,
    gross_return REAL,
    net_return   REAL,
    take_profit  REAL,
    stop_loss    REAL,
    PRIMARY KEY (run_id, trade_no)
);

CREATE TABLE IF NOT EXISTS copula_fits (
    pair                TEXT NOT NULL,
    month               TEXT NOT NULL,
    gaussian_rho        REAL,
    gaussian_aic        REAL,
    t_rho               REAL,
    t_nu                REAL,
    t_aic               REAL,
    t_tail_dep          REAL,
    gaussian_oos_loglik REAL,
    t_oos_loglik        REAL,
    PRIMARY KEY (pair, month)
);
