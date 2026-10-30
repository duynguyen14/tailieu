from __future__ import annotations

import re
from pathlib import Path

from app.core.config import PROJECT_ROOT, get_settings
from app.db.database import Database


SQL_SCRIPT_PATH = PROJECT_ROOT / "tools" / "sql" / "create_table_annotation_schema.sql"
GO_SEPARATOR = re.compile(r"^\s*GO\s*;?\s*$", flags=re.IGNORECASE | re.MULTILINE)


def split_sql_server_batches(script: str) -> list[str]:
    """Split a SQL Server script on standalone GO batch separators."""

    return [batch.strip() for batch in GO_SEPARATOR.split(script) if batch.strip()]


def load_script(path: Path = SQL_SCRIPT_PATH) -> str:
    if not path.is_file():
        raise RuntimeError(f"Table annotation SQL script was not found: {path}")
    return path.read_text(encoding="utf-8-sig")


def main() -> None:
    database = Database(get_settings())
    try:
        if database.engine.dialect.name != "mssql":
            raise RuntimeError(
                "Table annotation schema migration supports Microsoft SQL Server only."
            )

        batches = split_sql_server_batches(load_script())
        with database.engine.begin() as connection:
            for batch in batches:
                connection.exec_driver_sql(batch)
    finally:
        database.close()

    print("Table annotation database schema is ready.")


if __name__ == "__main__":
    main()
