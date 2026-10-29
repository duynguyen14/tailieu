from __future__ import annotations

from app.core.config import get_settings
from app.db.database import Database


def main() -> None:
    settings = get_settings()
    database = Database(settings)
    try:
        database.create_tables()
    finally:
        database.close()
    print("Annotation database tables are ready.")


if __name__ == "__main__":
    main()
