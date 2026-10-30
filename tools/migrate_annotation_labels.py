from __future__ import annotations

from sqlalchemy import text

from app.core.config import get_settings
from app.db.database import Database
from app.schemas.annotation import AnnotationLabel

CONSTRAINT_NAME = "CK_annotation_pages_label"


def main() -> None:
    database = Database(get_settings())
    try:
        if database.engine.dialect.name != "mssql":
            raise RuntimeError("This migration is intended for Microsoft SQL Server only.")

        allowed_values = [label.value for label in AnnotationLabel]
        allowed_sql = ", ".join(f"N'{value}'" for value in allowed_values)
        with database.engine.begin() as connection:
            invalid_labels = (
                connection.execute(
                    text(
                        f"SELECT DISTINCT label FROM dbo.annotation_pages "
                        f"WHERE label IS NOT NULL AND label NOT IN ({allowed_sql})"
                    )
                )
                .scalars()
                .all()
            )
            if invalid_labels:
                raise RuntimeError(
                    "Cannot update the label constraint because unsupported labels exist: "
                    + ", ".join(sorted(invalid_labels))
                )

            constraint_exists = connection.execute(
                text(
                    "SELECT COUNT(*) FROM sys.check_constraints "
                    "WHERE name = :constraint_name "
                    "AND parent_object_id = OBJECT_ID(N'dbo.annotation_pages')"
                ),
                {"constraint_name": CONSTRAINT_NAME},
            ).scalar_one()
            if constraint_exists:
                connection.execute(
                    text(f"ALTER TABLE dbo.annotation_pages DROP CONSTRAINT {CONSTRAINT_NAME}")
                )
            connection.execute(
                text(
                    f"ALTER TABLE dbo.annotation_pages WITH CHECK "
                    f"ADD CONSTRAINT {CONSTRAINT_NAME} CHECK "
                    f"(label IS NULL OR label IN ({allowed_sql}))"
                )
            )
            connection.execute(
                text(f"ALTER TABLE dbo.annotation_pages CHECK CONSTRAINT {CONSTRAINT_NAME}")
            )
    finally:
        database.close()

    print("Annotation label constraint now accepts the expanded label set.")


if __name__ == "__main__":
    main()
