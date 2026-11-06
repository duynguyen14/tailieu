from __future__ import annotations

from sqlalchemy import inspect, text

from app.core.config import get_settings
from app.db.database import Database
from app.db.models import (
    KieAnnotationReview,
    KieAnnotationRun,
    KieEntityAnnotation,
    KieFieldProposal,
    KiePageAnnotation,
    KieRelationAnnotation,
)


MSSQL_UNICODE_JSON_COLUMNS = {
    "kie_page_annotations": {
        "preprocess_metadata_json": "NULL",
    },
    "kie_field_proposals": {
        "chosen_json": "NULL",
        "candidates_json": "NOT NULL",
    },
    "kie_entity_annotations": {
        "spans_json": "NOT NULL",
    },
    "kie_annotation_reviews": {
        "payload_json": "NOT NULL",
    },
}


def _ensure_mssql_unicode_json_columns(database: Database) -> None:
    """Convert multilingual JSON storage from VARCHAR to NVARCHAR(MAX)."""

    with database.engine.begin() as connection:
        for table_name, columns in MSSQL_UNICODE_JSON_COLUMNS.items():
            for column_name, nullability in columns.items():
                data_type = connection.scalar(
                    text(
                        "SELECT DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS "
                        "WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = :table_name "
                        "AND COLUMN_NAME = :column_name"
                    ),
                    {"table_name": table_name, "column_name": column_name},
                )
                if data_type is None:
                    continue
                if str(data_type).lower() == "nvarchar":
                    continue
                connection.execute(
                    text(
                        f"ALTER TABLE dbo.{table_name} "
                        f"ALTER COLUMN {column_name} NVARCHAR(MAX) {nullability}"
                    )
                )
                print(f"Converted {table_name}.{column_name} to NVARCHAR(MAX).")


def main() -> None:
    settings = get_settings()
    if settings.db_name.strip().lower() != "anotation_2":
        raise SystemExit("KIE tables must be created with DB_NAME=anotation_2.")
    database = Database(settings)
    try:
        for table in (
            KieAnnotationRun.__table__,
            KiePageAnnotation.__table__,
            KieFieldProposal.__table__,
            KieEntityAnnotation.__table__,
            KieRelationAnnotation.__table__,
            KieAnnotationReview.__table__,
        ):
            table.create(database.engine, checkfirst=True)
        columns = {
            column["name"]
            for column in inspect(database.engine).get_columns("kie_entity_annotations")
        }
        if "spans_json" not in columns:
            with database.engine.begin() as connection:
                if database.engine.dialect.name == "mssql":
                    connection.execute(text(
                        "ALTER TABLE dbo.kie_entity_annotations "
                        "ADD spans_json NVARCHAR(MAX) NOT NULL "
                        "CONSTRAINT df_kie_entity_spans_json DEFAULT N'[]'"
                    ))
                else:
                    connection.execute(text(
                        "ALTER TABLE kie_entity_annotations "
                        "ADD COLUMN spans_json TEXT NOT NULL DEFAULT '[]'"
                    ))
            print("Added kie_entity_annotations.spans_json.")
        if database.engine.dialect.name == "mssql":
            _ensure_mssql_unicode_json_columns(database)
    finally:
        database.close()
    print("KIE annotation tables are ready.")


if __name__ == "__main__":
    main()
