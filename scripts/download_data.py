"""Download prices and the risk-free rate into SQLite, then check them against data/manifest.json.

The manifest is written only if it does not exist yet or with --update-manifest,
so a fresh download is compared with the data behind the published results.
"""
import json
import sys

from cpt.config import db_path, load_config
from cpt.data import download_prices, download_risk_free
from cpt.db import connect, init_db, upsert
from cpt.manifest import build_manifest, compare_manifest, write_manifest
from scripts.verify_data import MANIFEST_PATH


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

        manifest = build_manifest(conn)

    if "--update-manifest" in sys.argv or not MANIFEST_PATH.exists():
        write_manifest(manifest, MANIFEST_PATH)
        print(f"Wrote {MANIFEST_PATH}")
        return
    problems = compare_manifest(json.loads(MANIFEST_PATH.read_text()), manifest)
    print("Matches data/manifest.json." if not problems else
          "Differs from data/manifest.json (run with --update-manifest to accept):\n  " + "\n  ".join(problems))


if __name__ == "__main__":
    main()
