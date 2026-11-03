from __future__ import annotations

import argparse
import re

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import URL

from app.core.config import Settings, get_settings
from app.db.database import Database

DATABASE_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a new empty SQL Server annotation database and initialize the "
            "latest ORM schema. Existing databases are never dropped or cleared."
        )
    )
    parser.add_argument(
        "--database-name",
        required=True,
        help="New SQL Server database name, for example anotation_2.",
    )
    return parser.parse_args()


def _master_url(settings: Settings) -> URL:
    query = {
        "driver": settings.db_driver,
        "Encrypt": "yes" if settings.db_encrypt else "no",
        "TrustServerCertificate": (
            "yes" if settings.db_trust_server_certificate else "no"
        ),
    }
    username = settings.db_user
    password = (
        settings.db_password.get_secret_value() if settings.db_password else None
    )
    if settings.db_trusted_connection:
        username = None
        password = None
        query["Trusted_Connection"] = "yes"
    return URL.create(
        drivername="mssql+pyodbc",
        username=username,
        password=password,
        host=settings.db_server,
        port=settings.db_port,
        database="master",
        query=query,
    )


def create_database_if_missing(settings: Settings, database_name: str) -> bool:
    if not DATABASE_NAME_PATTERN.fullmatch(database_name):
        raise ValueError(
            "Database name must start with a letter and contain only letters, "
            "numbers or underscores."
        )

    engine = create_engine(_master_url(settings), pool_pre_ping=True)
    try:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
            exists = connection.scalar(
                text("SELECT CASE WHEN DB_ID(:name) IS NULL THEN 0 ELSE 1 END"),
                {"name": database_name},
            )
            if exists:
                return False
            connection.exec_driver_sql(f"CREATE DATABASE [{database_name}]")
            return True
    finally:
        engine.dispose()


def initialize_schema(settings: Settings, database_name: str) -> list[str]:
    target_settings = settings.model_copy(update={"db_name": database_name})
    database = Database(target_settings)
    try:
        if database.engine.dialect.name != "mssql":
            raise RuntimeError("This tool supports Microsoft SQL Server only.")
        database.create_tables()
        return sorted(inspect(database.engine).get_table_names(schema="dbo"))
    finally:
        database.close()


def main() -> None:
    args = parse_args()
    database_name = args.database_name.strip()
    settings = get_settings()
    created = create_database_if_missing(settings, database_name)
    tables = initialize_schema(settings, database_name)
    action = "created" if created else "already existed; schema verified"
    print(f"Database [{database_name}] {action}.")
    print(f"Initialized {len(tables)} table(s): {', '.join(tables)}")
    print("No existing database was dropped, cleared or copied.")


if __name__ == "__main__":
    main()
