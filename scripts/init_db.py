from cpt.config import db_path, load_config
from cpt.db import connect, init_db


def main() -> None:
    path = db_path(load_config())
    with connect(path) as conn:
        init_db(conn)
    print(f"Initialized {path}")


if __name__ == "__main__":
    main()
