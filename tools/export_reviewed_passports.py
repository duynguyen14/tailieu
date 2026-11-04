from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from app.core.config import PROJECT_ROOT, get_settings
from app.db.database import Database
from app.db.models import AnnotationDocument, AnnotationPage
from app.schemas.annotation import AnnotationLabel, ReviewStatus


DEFAULT_SOURCE_DATABASE = "document_annotation"
DEFAULT_OUTPUT_DIR = Path("data/exports/passport_front")
TARGET_LABEL = AnnotationLabel.PASSPORT_FRONT.value


@dataclass(frozen=True)
class PassportPage:
    document_id: str
    page_id: str
    original_filename: str
    page_number: int
    image_relative_path: str
    mime_type: str
    width: int
    height: int
    label: str
    review_status: str
    reviewed_by: str
    reviewed_at: str


@dataclass(frozen=True)
class ExportedPassportPage:
    exported_filename: str
    document_id: str
    page_id: str
    original_filename: str
    page_number: int
    label: str
    source_relative_path: str
    reviewed_by: str
    reviewed_at: str
    mime_type: str
    width: int
    height: int
    sha256: str


MANIFEST_FIELDS = [field.name for field in ExportedPassportPage.__dataclass_fields__.values()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy human-reviewed PASSPORT_FRONT page images from the legacy "
            "annotation database into one export directory. The source database "
            "and source images are read-only."
        )
    )
    parser.add_argument(
        "--source-database",
        default=DEFAULT_SOURCE_DATABASE,
        help=f"Legacy SQL Server database (default: {DEFAULT_SOURCE_DATABASE}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Empty destination directory (default: {DEFAULT_OUTPUT_DIR.as_posix()}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate DB rows and source files without creating or copying anything.",
    )
    return parser.parse_args()


def resolve_output_dir(value: Path) -> Path:
    if value.is_absolute():
        return value.resolve()
    return (PROJECT_ROOT / value).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_passport_pages(database: Database) -> list[PassportPage]:
    with database.session() as session:
        rows = session.execute(
            select(AnnotationDocument, AnnotationPage)
            .join(AnnotationPage, AnnotationPage.document_id == AnnotationDocument.id)
            .where(
                AnnotationPage.label == TARGET_LABEL,
                AnnotationPage.review_status == ReviewStatus.REVIEWED.value,
                AnnotationPage.reviewed_by.is_not(None),
                AnnotationPage.reviewed_at.is_not(None),
            )
            .order_by(
                AnnotationDocument.original_filename.asc(),
                AnnotationDocument.id.asc(),
                AnnotationPage.page_number.asc(),
            )
        ).all()

        return [
            PassportPage(
                document_id=document.id,
                page_id=page.id,
                original_filename=document.original_filename,
                page_number=page.page_number,
                image_relative_path=page.image_relative_path,
                mime_type=page.mime_type,
                width=page.width,
                height=page.height,
                label=str(page.label),
                review_status=page.review_status,
                reviewed_by=str(page.reviewed_by),
                reviewed_at=page.reviewed_at.isoformat(),
            )
            for document, page in rows
        ]


def validate_source_files(
    records: list[PassportPage], storage_root: Path
) -> list[tuple[PassportPage, Path]]:
    validated: list[tuple[PassportPage, Path]] = []
    errors: list[str] = []

    for record in records:
        source_path = (storage_root / record.image_relative_path).resolve()
        if not source_path.is_relative_to(storage_root):
            errors.append(
                f"{record.page_id}: path escapes ANNOTATION_STORAGE_DIR "
                f"({record.image_relative_path})"
            )
            continue
        if not source_path.is_file():
            errors.append(
                f"{record.page_id}: source image does not exist "
                f"({record.image_relative_path})"
            )
            continue
        validated.append((record, source_path))

    if errors:
        preview = "\n".join(f"- {item}" for item in errors[:20])
        remainder = len(errors) - 20
        suffix = f"\n- ... and {remainder} more" if remainder > 0 else ""
        raise RuntimeError(f"Passport export validation failed:\n{preview}{suffix}")
    return validated


def ensure_empty_destination(output_dir: Path) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise RuntimeError(f"Output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {output_dir}. "
            "Choose a new directory or empty it explicitly before exporting."
        )


def exported_filename(record: PassportPage, source_path: Path) -> str:
    suffix = source_path.suffix.lower() or ".png"
    return (
        f"{record.document_id}_page_{record.page_number:04d}_"
        f"{record.page_id}{suffix}"
    )


def write_manifest(
    output_dir: Path,
    rows: list[ExportedPassportPage],
    *,
    source_database: str,
) -> None:
    manifest_csv = output_dir / "manifest.csv"
    with manifest_csv.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    manifest_json = output_dir / "manifest.json"
    manifest_json.write_text(
        json.dumps(
            {
                "created_at": datetime.now(UTC).isoformat(),
                "source_database": source_database,
                "label": TARGET_LABEL,
                "exported_count": len(rows),
                "items": [asdict(row) for row in rows],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def export_reviewed_passports(
    *, source_database: str, output_dir: Path, dry_run: bool
) -> int:
    if not source_database:
        raise ValueError("--source-database cannot be empty.")
    settings = get_settings()
    source_settings = settings.model_copy(update={"db_name": source_database})
    storage_root = settings.annotation_storage_path.resolve()

    database = Database(source_settings)
    try:
        records = load_passport_pages(database)
    finally:
        database.close()

    if not records:
        raise RuntimeError(
            f"No human-reviewed {TARGET_LABEL} pages were found in "
            f"database [{source_database}]."
        )

    validated = validate_source_files(records, storage_root)
    if dry_run:
        print(
            f"DRY RUN OK: database=[{source_database}], label={TARGET_LABEL}, "
            f"reviewed_pages={len(validated)}, storage={storage_root}"
        )
        print(f"No files were copied. Planned output: {output_dir}")
        return len(validated)

    ensure_empty_destination(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    exported: list[ExportedPassportPage] = []
    for index, (record, source_path) in enumerate(validated, start=1):
        filename = exported_filename(record, source_path)
        destination = images_dir / filename
        shutil.copy2(source_path, destination)
        exported.append(
            ExportedPassportPage(
                exported_filename=(Path("images") / filename).as_posix(),
                document_id=record.document_id,
                page_id=record.page_id,
                original_filename=record.original_filename,
                page_number=record.page_number,
                label=record.label,
                source_relative_path=record.image_relative_path,
                reviewed_by=record.reviewed_by,
                reviewed_at=record.reviewed_at,
                mime_type=record.mime_type,
                width=record.width,
                height=record.height,
                sha256=sha256_file(source_path),
            )
        )
        print(f"[{index}/{len(validated)}] copied page={record.page_id} -> {filename}")

    write_manifest(output_dir, exported, source_database=source_database)
    print(f"Exported {len(exported)} passport page image(s) to {output_dir}")
    print(f"Manifest: {output_dir / 'manifest.csv'}")
    return len(exported)


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    export_reviewed_passports(
        source_database=args.source_database.strip(),
        output_dir=output_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
