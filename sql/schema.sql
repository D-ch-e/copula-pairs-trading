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
