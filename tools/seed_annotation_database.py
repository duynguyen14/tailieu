from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

from sqlalchemy import text

from app.core.config import get_settings
from app.db.database import Database

DATABASE_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")


@dataclass(frozen=True)
class SeedSummary:
    source_database: str
    target_database: str
    document_count: int
    page_count: int
    labeled_page_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Seed a fresh annotation database from document/page metadata in an "
            "existing database. Source rows are read-only; OCR and review history "
            "are never copied."
        )
    )
    parser.add_argument(
        "--source-database",
        default="document_annotation",
        help="Existing SQL Server annotation database.",
    )
    parser.add_argument(
        "--target-database",
        default=None,
        help="Fresh target database. Defaults to DB_NAME from .env.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform the copy. Without this flag, only validate and report counts.",
    )
    return parser.parse_args()


def _validate_database_name(value: str, *, argument: str) -> str:
    normalized = value.strip()
    if not DATABASE_NAME_PATTERN.fullmatch(normalized):
        raise ValueError(
            f"{argument} must start with a letter and contain only letters, "
            "numbers or underscores."
        )
    return normalized


def _table_name(database_name: str, table_name: str) -> str:
    return f"[{database_name}].[dbo].[{table_name}]"


def seed_database(
    *,
    source_database: str,
    target_database: str,
    execute: bool,
) -> SeedSummary:
    settings = get_settings()
    source_database = _validate_database_name(
        source_database, argument="--source-database"
    )
    target_database = _validate_database_name(
        target_database, argument="--target-database"
    )
    if source_database.casefold() == target_database.casefold():
        raise ValueError("Source and target databases must be different.")

    source_documents = _table_name(source_database, "annotation_documents")
    source_pages = _table_name(source_database, "annotation_pages")
    target_documents = _table_name(target_database, "annotation_documents")
    target_pages = _table_name(target_database, "annotation_pages")

    database = Database(settings.model_copy(update={"db_name": target_database}))
    try:
        with database.engine.connect() as connection:
            source_exists = connection.scalar(
                text("SELECT CASE WHEN DB_ID(:name) IS NULL THEN 0 ELSE 1 END"),
                {"name": source_database},
            )
            target_exists = connection.scalar(
                text("SELECT CASE WHEN DB_ID(:name) IS NULL THEN 0 ELSE 1 END"),
                {"name": target_database},
            )
            if not source_exists:
                raise RuntimeError(f"Source database [{source_database}] does not exist.")
            if not target_exists:
                raise RuntimeError(f"Target database [{target_database}] does not exist.")

            source_document_count = int(
                connection.exec_driver_sql(
                    f"SELECT COUNT_BIG(*) FROM {source_documents}"
                ).scalar_one()
            )
            source_page_count = int(
                connection.exec_driver_sql(
                    f"SELECT COUNT_BIG(*) FROM {source_pages}"
                ).scalar_one()
            )
            labeled_page_count = int(
                connection.exec_driver_sql(
                    f"SELECT COUNT_BIG(*) FROM {source_pages} WHERE label IS NOT NULL"
                ).scalar_one()
            )
            orphan_page_count = int(
                connection.exec_driver_sql(
                    f"""
                    SELECT COUNT_BIG(*)
                    FROM {source_pages} AS p
                    LEFT JOIN {source_documents} AS d ON d.id = p.document_id
                    WHERE d.id IS NULL
                    """
                ).scalar_one()
            )
            target_document_count = int(
                connection.exec_driver_sql(
                    f"SELECT COUNT_BIG(*) FROM {target_documents}"
                ).scalar_one()
            )
            target_page_count = int(
                connection.exec_driver_sql(
                    f"SELECT COUNT_BIG(*) FROM {target_pages}"
                ).scalar_one()
            )

        if source_document_count == 0 or source_page_count == 0:
            raise RuntimeError("Source annotation document/page tables are empty.")
        if orphan_page_count:
            raise RuntimeError(
                f"Source contains {orphan_page_count} page(s) without a document."
            )
        if target_document_count or target_page_count:
            raise RuntimeError(
                "Target annotation_documents and annotation_pages must both be empty. "
                f"Current counts: documents={target_document_count}, "
                f"pages={target_page_count}."
            )

        summary = SeedSummary(
            source_database=source_database,
            target_database=target_database,
            document_count=source_document_count,
            page_count=source_page_count,
            labeled_page_count=labeled_page_count,
        )
        if not execute:
            return summary

        with database.engine.begin() as connection:
            connection.exec_driver_sql(
                f"""
                INSERT INTO {target_documents}
                (
                    id,
                    original_filename,
                    pdf_relative_path,
                    sha256,
                    page_count,
                    review_status,
                    created_at,
                    updated_at
                )
                SELECT
                    id,
                    original_filename,
                    pdf_relative_path,
                    sha256,
                    page_count,
                    N'UNLABELED',
                    created_at,
                    SYSUTCDATETIME()
                FROM {source_documents};
                """
            )
            connection.exec_driver_sql(
                f"""
                INSERT INTO {target_pages}
                (
                    id,
                    document_id,
                    page_index,
                    page_number,
                    image_relative_path,
                    mime_type,
                    width,
                    height,
                    label,
                    review_status,
                    reviewed_by,
                    review_note,
                    reviewed_at,
                    created_at,
                    updated_at,
                    model_predicted_label,
                    model_prediction_confidence,
                    model_prediction_topk_json,
                    model_prediction_version,
                    model_predicted_at
                )
                SELECT
                    id,
                    document_id,
                    page_index,
                    page_number,
                    image_relative_path,
                    mime_type,
                    width,
                    height,
                    label,
                    N'UNLABELED',
                    NULL,
                    NULL,
                    NULL,
                    created_at,
                    SYSUTCDATETIME(),
                    NULL,
                    NULL,
                    NULL,
                    NULL,
                    NULL
                FROM {source_pages};
                """
            )

            copied_document_count = int(
                connection.exec_driver_sql(
                    f"SELECT COUNT_BIG(*) FROM {target_documents}"
                ).scalar_one()
            )
            copied_page_count = int(
                connection.exec_driver_sql(
                    f"SELECT COUNT_BIG(*) FROM {target_pages}"
                ).scalar_one()
            )
            pending_page_count = int(
                connection.exec_driver_sql(
                    f"""
                    SELECT COUNT_BIG(*)
                    FROM {target_pages}
                    WHERE review_status = N'UNLABELED'
                      AND reviewed_by IS NULL
                      AND reviewed_at IS NULL
                    """
                ).scalar_one()
            )
            copied_labeled_page_count = int(
                connection.exec_driver_sql(
                    f"SELECT COUNT_BIG(*) FROM {target_pages} WHERE label IS NOT NULL"
                ).scalar_one()
            )
            if (
                copied_document_count != source_document_count
                or copied_page_count != source_page_count
                or pending_page_count != source_page_count
                or copied_labeled_page_count != labeled_page_count
            ):
                raise RuntimeError(
                    "Post-copy verification failed; the transaction will be rolled back."
                )
        return summary
    finally:
        database.close()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    target_database = args.target_database or settings.db_name
    summary = seed_database(
        source_database=args.source_database,
        target_database=target_database,
        execute=args.execute,
    )
    mode = "EXECUTED" if args.execute else "DRY RUN"
    print(f"Seed annotation database: {mode}")
    print(f"  Source database : {summary.source_database}")
    print(f"  Target database : {summary.target_database}")
    print(f"  Documents       : {summary.document_count}")
    print(f"  Pages           : {summary.page_count}")
    print(f"  Pages with label: {summary.labeled_page_count}")
    if not args.execute:
        print("No rows were copied. Run again with --execute after verifying counts.")
    else:
        print("All copied pages are UNLABELED and ready for human review.")
        print("Source database and shared storage files were not modified.")


if __name__ == "__main__":
    main()
