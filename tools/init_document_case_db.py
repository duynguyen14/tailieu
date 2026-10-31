from __future__ import annotations

from app.core.config import PROJECT_ROOT, get_settings
from app.db.database import Database
from tools.init_ocr_grounding_db import split_sql_server_batches


SQL_SCRIPT_PATH = PROJECT_ROOT / "tools" / "sql" / "create_document_case_schema.sql"


def main() -> None:
    database = Database(get_settings())
    try:
        if database.engine.dialect.name != "mssql":
            raise RuntimeError("Document case migration supports Microsoft SQL Server only.")
        script = SQL_SCRIPT_PATH.read_text(encoding="utf-8-sig")
        with database.engine.begin() as connection:
            for batch in split_sql_server_batches(script):
                connection.exec_driver_sql(batch)
    finally:
        database.close()
    print("Document case database schema is ready.")


if __name__ == "__main__":
    main()
