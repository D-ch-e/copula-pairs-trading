"""Check the local database against data/manifest.json (the data behind the published results)."""
import json
import sys

from cpt.config import PROJECT_ROOT, db_path, load_config
from cpt.db import connect
from cpt.manifest import build_manifest, compare_manifest

MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"


def main() -> int:
    expected = json.loads(MANIFEST_PATH.read_text())
    with connect(db_path(load_config())) as conn:
        problems = compare_manifest(expected, build_manifest(conn))
    if problems:
        print("Local data differs from the manifest; results may not reproduce exactly:")
        print("\n".join(f"  {p}" for p in problems))
        return 1
    print(f"Local data matches the manifest ({len(expected)} series).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
