from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import text

from app.core.config import get_settings
from app.db.database import Database


DATABASE_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")
SOURCE_PAGE_LABELS = ("DIPLOMATIC_NOTE", "APPLICANT_TABLE")
TARGET_PAGE_LABEL = "DIPLOMATIC_NOTE"


@dataclass(frozen=True)
class MigrationSummary:
    source_database: str
    target_database: str
    target_reviewed_notes: int
    source_label_matches: int
    source_note_matches: int
    source_table_matches: int
    source_approved_matches: int
    target_pages_with_existing_ocr: int
    eligible_pages: int
    ocr_runs: int
    ocr_tokens: int
    ocr_layout_regions: int
    ocr_reviews: int
    ocr_review_tokens: int
    ocr_review_regions: int
    executed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse active human-approved OCR snapshots from the legacy annotation "
            "database for reviewed DIPLOMATIC_NOTE pages in the new database. "
            "Without --execute this command is read-only."
        )
    )
    parser.add_argument(
        "--source-database",
        default="document_annotation",
        help="Legacy SQL Server database (default: document_annotation).",
    )
    parser.add_argument(
        "--target-database",
        default=None,
        help="New SQL Server database (default: DB_NAME from .env).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Insert eligible OCR snapshots. Omit this flag for a dry-run report.",
    )
    return parser.parse_args()


def validate_database_name(value: str, *, argument: str) -> str:
    normalized = value.strip()
    if not DATABASE_NAME_PATTERN.fullmatch(normalized):
        raise ValueError(
            f"{argument} must start with a letter and contain only letters, "
            "numbers or underscores."
        )
    return normalized


def table(database_name: str, table_name: str) -> str:
    return f"[{database_name}].[dbo].[{table_name}]"


def scalar_count(connection, sql: str) -> int:
    return int(connection.exec_driver_sql(sql).scalar_one())


def validate_shared_images(connection, *, storage_root: Path) -> None:
    rows = connection.exec_driver_sql(
        """
        SELECT page_id, display_image_relative_path
        FROM #selected_ocr
        ORDER BY page_id
        """
    ).mappings()

    errors: list[str] = []
    for row in rows:
        relative_value = str(row["display_image_relative_path"] or "").strip()
        if not relative_value:
            errors.append(f"{row['page_id']}: no display image path")
            continue

        image_path = (storage_root / relative_value).resolve()
        if not image_path.is_relative_to(storage_root):
            errors.append(f"{row['page_id']}: image path escapes storage ({relative_value})")
        elif not image_path.is_file():
            errors.append(f"{row['page_id']}: image is missing ({relative_value})")

    if errors:
        preview = "\n".join(f"- {item}" for item in errors[:20])
        remainder = len(errors) - 20
        suffix = f"\n- ... and {remainder} more" if remainder > 0 else ""
        raise RuntimeError(
            "OCR migration stopped because reviewed bbox coordinates require the "
            f"same original/processed images:\n{preview}{suffix}"
        )


def create_selected_snapshot_table(
    connection,
    *,
    source_database: str,
    target_database: str,
) -> None:
    source_pages = table(source_database, "annotation_pages")
    source_runs = table(source_database, "ocr_runs")
    source_reviews = table(source_database, "ocr_reviews")
    target_pages = table(target_database, "annotation_pages")
    target_runs = table(target_database, "ocr_runs")

    connection.exec_driver_sql(
        f"""
        IF OBJECT_ID('tempdb..#selected_ocr') IS NOT NULL
            DROP TABLE #selected_ocr;

        ;WITH ranked AS (
            SELECT
                tp.id AS page_id,
                sp.label AS source_page_label,
                sr.id AS ocr_run_id,
                rv.id AS ocr_review_id,
                COALESCE(sr.processed_image_relative_path, tp.image_relative_path)
                    AS display_image_relative_path,
                ROW_NUMBER() OVER (
                    PARTITION BY tp.id
                    ORDER BY rv.reviewed_at DESC, rv.created_at DESC, rv.id DESC
                ) AS row_number
            FROM {target_pages} AS tp
            INNER JOIN {source_pages} AS sp
                ON sp.id = tp.id
            INNER JOIN {source_runs} AS sr
                ON sr.page_id = sp.id
               AND sr.status = N'COMPLETED'
               AND sr.is_active = 1
            INNER JOIN {source_reviews} AS rv
                ON rv.ocr_run_id = sr.id
               AND rv.status = N'REVIEWED'
               AND rv.is_active = 1
               AND rv.include_in_llm = 1
            WHERE tp.label = N'{TARGET_PAGE_LABEL}'
              AND tp.review_status = N'REVIEWED'
              AND tp.reviewed_by IS NOT NULL
              AND tp.reviewed_at IS NOT NULL
              AND sp.label IN (N'{SOURCE_PAGE_LABELS[0]}', N'{SOURCE_PAGE_LABELS[1]}')
              AND sp.review_status = N'REVIEWED'
              AND sp.reviewed_by IS NOT NULL
              AND sp.reviewed_at IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1
                  FROM {target_runs} AS existing_run
                  WHERE existing_run.page_id = tp.id
              )
        )
        SELECT
            page_id,
            source_page_label,
            ocr_run_id,
            ocr_review_id,
            display_image_relative_path
        INTO #selected_ocr
        FROM ranked
        WHERE row_number = 1;

        CREATE UNIQUE CLUSTERED INDEX IX_selected_ocr_page
            ON #selected_ocr(page_id);
        CREATE UNIQUE INDEX IX_selected_ocr_run
            ON #selected_ocr(ocr_run_id);
        CREATE UNIQUE INDEX IX_selected_ocr_review
            ON #selected_ocr(ocr_review_id);
        """
    )


def collect_summary(
    connection,
    *,
    source_database: str,
    target_database: str,
    execute: bool,
) -> MigrationSummary:
    source_pages = table(source_database, "annotation_pages")
    source_runs = table(source_database, "ocr_runs")
    source_reviews = table(source_database, "ocr_reviews")
    source_tokens = table(source_database, "ocr_tokens")
    source_regions = table(source_database, "ocr_layout_regions")
    source_review_tokens = table(source_database, "ocr_review_tokens")
    source_review_regions = table(source_database, "ocr_review_regions")
    target_pages = table(target_database, "annotation_pages")
    target_runs = table(target_database, "ocr_runs")

    target_filter = f"""
        tp.label = N'{TARGET_PAGE_LABEL}'
        AND tp.review_status = N'REVIEWED'
        AND tp.reviewed_by IS NOT NULL
        AND tp.reviewed_at IS NOT NULL
    """
    source_filter = f"""
        sp.label IN (N'{SOURCE_PAGE_LABELS[0]}', N'{SOURCE_PAGE_LABELS[1]}')
        AND sp.review_status = N'REVIEWED'
        AND sp.reviewed_by IS NOT NULL
        AND sp.reviewed_at IS NOT NULL
    """

    target_reviewed_notes = scalar_count(
        connection,
        f"SELECT COUNT_BIG(*) FROM {target_pages} AS tp WHERE {target_filter}",
    )
    source_label_matches = scalar_count(
        connection,
        f"""
        SELECT COUNT_BIG(*)
        FROM {target_pages} AS tp
        INNER JOIN {source_pages} AS sp ON sp.id = tp.id
        WHERE {target_filter} AND {source_filter}
        """,
    )
    source_note_matches = scalar_count(
        connection,
        f"""
        SELECT COUNT_BIG(*)
        FROM {target_pages} AS tp
        INNER JOIN {source_pages} AS sp ON sp.id = tp.id
        WHERE {target_filter}
          AND {source_filter}
          AND sp.label = N'{SOURCE_PAGE_LABELS[0]}'
        """,
    )
    source_table_matches = scalar_count(
        connection,
        f"""
        SELECT COUNT_BIG(*)
        FROM {target_pages} AS tp
        INNER JOIN {source_pages} AS sp ON sp.id = tp.id
        WHERE {target_filter}
          AND {source_filter}
          AND sp.label = N'{SOURCE_PAGE_LABELS[1]}'
        """,
    )
    source_approved_matches = scalar_count(
        connection,
        f"""
        SELECT COUNT_BIG(DISTINCT tp.id)
        FROM {target_pages} AS tp
        INNER JOIN {source_pages} AS sp ON sp.id = tp.id
        INNER JOIN {source_runs} AS sr
            ON sr.page_id = sp.id
           AND sr.status = N'COMPLETED'
           AND sr.is_active = 1
        INNER JOIN {source_reviews} AS rv
            ON rv.ocr_run_id = sr.id
           AND rv.status = N'REVIEWED'
           AND rv.is_active = 1
           AND rv.include_in_llm = 1
        WHERE {target_filter} AND {source_filter}
        """,
    )
    target_pages_with_existing_ocr = scalar_count(
        connection,
        f"""
        SELECT COUNT_BIG(DISTINCT tp.id)
        FROM {target_pages} AS tp
        INNER JOIN {source_pages} AS sp ON sp.id = tp.id
        INNER JOIN {source_runs} AS sr
            ON sr.page_id = sp.id
           AND sr.status = N'COMPLETED'
           AND sr.is_active = 1
        INNER JOIN {source_reviews} AS rv
            ON rv.ocr_run_id = sr.id
           AND rv.status = N'REVIEWED'
           AND rv.is_active = 1
           AND rv.include_in_llm = 1
        INNER JOIN {target_runs} AS tr ON tr.page_id = tp.id
        WHERE {target_filter} AND {source_filter}
        """,
    )

    return MigrationSummary(
        source_database=source_database,
        target_database=target_database,
        target_reviewed_notes=target_reviewed_notes,
        source_label_matches=source_label_matches,
        source_note_matches=source_note_matches,
        source_table_matches=source_table_matches,
        source_approved_matches=source_approved_matches,
        target_pages_with_existing_ocr=target_pages_with_existing_ocr,
        eligible_pages=scalar_count(connection, "SELECT COUNT_BIG(*) FROM #selected_ocr"),
        ocr_runs=scalar_count(connection, "SELECT COUNT_BIG(*) FROM #selected_ocr"),
        ocr_tokens=scalar_count(
            connection,
            f"""
            SELECT COUNT_BIG(*)
            FROM {source_tokens} AS item
            INNER JOIN #selected_ocr AS selected
                ON selected.ocr_run_id = item.ocr_run_id
            """,
        ),
        ocr_layout_regions=scalar_count(
            connection,
            f"""
            SELECT COUNT_BIG(*)
            FROM {source_regions} AS item
            INNER JOIN #selected_ocr AS selected
                ON selected.ocr_run_id = item.ocr_run_id
            """,
        ),
        ocr_reviews=scalar_count(connection, "SELECT COUNT_BIG(*) FROM #selected_ocr"),
        ocr_review_tokens=scalar_count(
            connection,
            f"""
            SELECT COUNT_BIG(*)
            FROM {source_review_tokens} AS item
            INNER JOIN #selected_ocr AS selected
                ON selected.ocr_review_id = item.review_id
            """,
        ),
        ocr_review_regions=scalar_count(
            connection,
            f"""
            SELECT COUNT_BIG(*)
            FROM {source_review_regions} AS item
            INNER JOIN #selected_ocr AS selected
                ON selected.ocr_review_id = item.review_id
            """,
        ),
        executed=execute,
    )


def copy_selected_snapshots(
    connection,
    *,
    source_database: str,
    target_database: str,
) -> None:
    source_runs = table(source_database, "ocr_runs")
    source_tokens = table(source_database, "ocr_tokens")
    source_regions = table(source_database, "ocr_layout_regions")
    source_reviews = table(source_database, "ocr_reviews")
    source_review_tokens = table(source_database, "ocr_review_tokens")
    source_review_regions = table(source_database, "ocr_review_regions")

    target_runs = table(target_database, "ocr_runs")
    target_tokens = table(target_database, "ocr_tokens")
    target_regions = table(target_database, "ocr_layout_regions")
    target_reviews = table(target_database, "ocr_reviews")
    target_review_tokens = table(target_database, "ocr_review_tokens")
    target_review_regions = table(target_database, "ocr_review_regions")

    connection.exec_driver_sql("SET XACT_ABORT ON;")

    connection.exec_driver_sql(
        f"""
        INSERT INTO {target_runs} (
            id, page_id, status, is_active, model_name, model_version, device,
            processing_time_ms, error_message, orientation_angle,
            orientation_confidence, processed_image_relative_path,
            processed_width, processed_height, preprocess_metadata_json,
            preprocess_version, created_at, updated_at
        )
        SELECT
            source.id, source.page_id, source.status, source.is_active,
            source.model_name, source.model_version, source.device,
            source.processing_time_ms, source.error_message,
            source.orientation_angle, source.orientation_confidence,
            source.processed_image_relative_path, source.processed_width,
            source.processed_height, source.preprocess_metadata_json,
            source.preprocess_version, source.created_at, source.updated_at
        FROM {source_runs} AS source
        INNER JOIN #selected_ocr AS selected
            ON selected.ocr_run_id = source.id;
        """
    )

    connection.exec_driver_sql(
        f"""
        INSERT INTO {target_tokens} (
            id, ocr_run_id, token_index, token_id, text, confidence,
            x1, y1, x2, y2, polygon_json, include_in_extraction,
            ignored_reason, created_at
        )
        SELECT
            source.id, source.ocr_run_id, source.token_index, source.token_id,
            source.text, source.confidence, source.x1, source.y1, source.x2,
            source.y2, source.polygon_json, source.include_in_extraction,
            source.ignored_reason, source.created_at
        FROM {source_tokens} AS source
        INNER JOIN #selected_ocr AS selected
            ON selected.ocr_run_id = source.ocr_run_id;
        """
    )

    connection.exec_driver_sql(
        f"""
        INSERT INTO {target_regions} (
            id, ocr_run_id, region_index, region_type, confidence,
            x1, y1, x2, y2, created_at
        )
        SELECT
            source.id, source.ocr_run_id, source.region_index,
            source.region_type, source.confidence, source.x1, source.y1,
            source.x2, source.y2, source.created_at
        FROM {source_regions} AS source
        INNER JOIN #selected_ocr AS selected
            ON selected.ocr_run_id = source.ocr_run_id;
        """
    )

    # The active approved snapshot becomes the root review in the clean database.
    # Historical parents are deliberately not copied.
    connection.exec_driver_sql(
        f"""
        INSERT INTO {target_reviews} (
            id, ocr_run_id, parent_review_id, status, is_active,
            include_in_llm, reviewed_by, review_note, reviewed_at,
            created_at, updated_at
        )
        SELECT
            source.id, source.ocr_run_id, NULL, source.status,
            source.is_active, source.include_in_llm, source.reviewed_by,
            source.review_note, source.reviewed_at, source.created_at,
            source.updated_at
        FROM {source_reviews} AS source
        INNER JOIN #selected_ocr AS selected
            ON selected.ocr_review_id = source.id;
        """
    )

    connection.exec_driver_sql(
        f"""
        INSERT INTO {target_review_tokens} (
            id, review_id, source_token_id, token_order, reviewed_text,
            x1, y1, x2, y2, include_in_extraction, ignored_reason, created_at
        )
        SELECT
            source.id, source.review_id, source.source_token_id,
            source.token_order, source.reviewed_text, source.x1, source.y1,
            source.x2, source.y2, source.include_in_extraction,
            source.ignored_reason, source.created_at
        FROM {source_review_tokens} AS source
        INNER JOIN #selected_ocr AS selected
            ON selected.ocr_review_id = source.review_id;
        """
    )

    connection.exec_driver_sql(
        f"""
        INSERT INTO {target_review_regions} (
            id, review_id, source_region_id, region_order, region_type,
            x1, y1, x2, y2, created_at
        )
        SELECT
            source.id, source.review_id, source.source_region_id,
            source.region_order, source.region_type, source.x1, source.y1,
            source.x2, source.y2, source.created_at
        FROM {source_review_regions} AS source
        INNER JOIN #selected_ocr AS selected
            ON selected.ocr_review_id = source.review_id;
        """
    )


def print_summary(summary: MigrationSummary) -> None:
    mode = "EXECUTED" if summary.executed else "DRY RUN"
    print(f"OCR review migration: {mode}")
    print(f"Source database: [{summary.source_database}]")
    print(f"Target database: [{summary.target_database}]")
    print(f"Target reviewed DIPLOMATIC_NOTE pages: {summary.target_reviewed_notes}")
    print(f"Matched old NOTE/TABLE labels: {summary.source_label_matches}")
    print(f"  old DIPLOMATIC_NOTE: {summary.source_note_matches}")
    print(f"  old APPLICANT_TABLE: {summary.source_table_matches}")
    print(f"Matched active approved OCR reviews: {summary.source_approved_matches}")
    print(f"Skipped because target already has OCR: {summary.target_pages_with_existing_ocr}")
    print(f"Eligible pages: {summary.eligible_pages}")
    print("Rows selected:")
    print(f"  ocr_runs: {summary.ocr_runs}")
    print(f"  ocr_tokens: {summary.ocr_tokens}")
    print(f"  ocr_layout_regions: {summary.ocr_layout_regions}")
    print(f"  ocr_reviews: {summary.ocr_reviews}")
    print(f"  ocr_review_tokens: {summary.ocr_review_tokens}")
    print(f"  ocr_review_regions: {summary.ocr_review_regions}")
    if not summary.executed:
        print("No database rows were changed. Add --execute to perform the migration.")


def migrate(*, source_database: str, target_database: str, execute: bool) -> MigrationSummary:
    settings = get_settings()
    source_database = validate_database_name(
        source_database, argument="--source-database"
    )
    target_database = validate_database_name(
        target_database, argument="--target-database"
    )
    if source_database.casefold() == target_database.casefold():
        raise ValueError("Source and target databases must be different.")

    database = Database(settings.model_copy(update={"db_name": target_database}))
    try:
        transaction = database.engine.begin()
        with transaction as connection:
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

            create_selected_snapshot_table(
                connection,
                source_database=source_database,
                target_database=target_database,
            )
            validate_shared_images(
                connection,
                storage_root=settings.annotation_storage_path.resolve(),
            )
            summary = collect_summary(
                connection,
                source_database=source_database,
                target_database=target_database,
                execute=execute,
            )
            if execute and summary.eligible_pages:
                copy_selected_snapshots(
                    connection,
                    source_database=source_database,
                    target_database=target_database,
                )
        return summary
    finally:
        database.close()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    target_database = args.target_database or settings.db_name
    summary = migrate(
        source_database=args.source_database,
        target_database=target_database,
        execute=args.execute,
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
