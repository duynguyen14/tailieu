from __future__ import annotations

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
    finally:
        database.close()
    print("KIE annotation tables are ready.")


if __name__ == "__main__":
    main()
