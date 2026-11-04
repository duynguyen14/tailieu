from __future__ import annotations

from app.core.config import PROJECT_ROOT, get_settings
from app.db.database import Database
from tools.init_ocr_grounding_db import split_sql_server_batches


SQL_SCRIPT_PATH = PROJECT_ROOT / "tools" / "sql" / "create_teacher_extraction_schema.sql"


def main() -> None:
    settings = get_settings()
    if settings.db_name.lower() != "anotation_2":
        raise RuntimeError(
            "Teacher extraction schema must be initialized with DB_NAME=anotation_2."
        )

    database = Database(settings)
    try:
        if database.engine.dialect.name != "mssql":
            raise RuntimeError(
                "Teacher extraction schema migration supports Microsoft SQL Server only."
            )
        script = SQL_SCRIPT_PATH.read_text(encoding="utf-8-sig")
        with database.engine.begin() as connection:
            for batch in split_sql_server_batches(script):
                connection.exec_driver_sql(batch)
    finally:
        database.close()

    print(f"Teacher extraction database schema is ready in [{settings.db_name}].")


if __name__ == "__main__":
    main()
