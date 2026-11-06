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
    finally:
        database.close()
    print("KIE annotation tables are ready.")


if __name__ == "__main__":
    main()
