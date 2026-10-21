from cpt.config import db_path, load_config
from cpt.data import download_prices, download_risk_free
from cpt.db import connect, init_db, upsert


def main() -> None:
    cfg = load_config()
    data_cfg = cfg["data"]
    with connect(db_path(cfg)) as conn:
        init_db(conn)
        for ticker in data_cfg["tickers"]:
            df = download_prices(ticker, str(data_cfg["start"]), str(data_cfg["end"]))
            upsert(conn, "prices", df)
            print(f"{ticker}: {len(df)} rows ({df['date'].min()} -> {df['date'].max()})")

        rf = download_risk_free(data_cfg["french_daily_url"])
        upsert(conn, "risk_free", rf)
        print(f"risk_free: {len(rf)} rows ({rf['date'].min()} -> {rf['date'].max()})")


if __name__ == "__main__":
    main()
