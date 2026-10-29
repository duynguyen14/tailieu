from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from os import scandir
from os.path import abspath
from pathlib import Path

from sqlalchemy import text

from app.core.config import Settings, get_settings
from app.db.database import Database
from app.services.annotation import AnnotationService
from app.services.pdf_renderer import PdfRenderer
from app.services.storage import AnnotationStorage


@dataclass(frozen=True)
class BulkImportSummary:
    discovered: int = 0
    imported: int = 0
    duplicates: int = 0
    failed: int = 0
    pages_created: int = 0


def paths_overlap(left: Path, right: Path) -> bool:
    left = Path(abspath(left))
    right = Path(abspath(right))
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def validate_source_root(source_root: Path, storage_root: Path) -> None:
    if source_root.is_symlink() or source_root.is_junction():
        raise ValueError("BULK_IMPORT_ROOT cannot be a symbolic link or directory junction.")
    if not source_root.exists():
        raise ValueError(f"BULK_IMPORT_ROOT does not exist: {source_root}")
    if not source_root.is_dir():
        raise ValueError(f"BULK_IMPORT_ROOT is not a directory: {source_root}")
    if paths_overlap(source_root, storage_root):
        raise ValueError(
            "BULK_IMPORT_ROOT must not overlap ANNOTATION_STORAGE_DIR; "
            "input PDFs and generated files must be in separate directories."
        )


def discover_pdfs(source_root: Path) -> list[Path]:
    """Find every PDF recursively without following links outside the source tree."""

    pdf_paths: list[Path] = []
    directories = [source_root]
    while directories:
        directory = directories.pop()
        with scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                if path.is_symlink() or path.is_junction():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    directories.append(path)
                elif entry.is_file(follow_symlinks=False) and path.suffix.casefold() == ".pdf":
                    pdf_paths.append(path)

    return sorted(
        pdf_paths,
        key=lambda path: str(path.relative_to(source_root)).casefold(),
    )


def run_bulk_import(
    settings: Settings,
    *,
    emit: Callable[[str], None] = print,
) -> BulkImportSummary:
    source_root = settings.bulk_import_root_path
    storage_root = settings.annotation_storage_path
    validate_source_root(source_root, storage_root)

    pdf_paths = discover_pdfs(source_root)
    emit(f"Source: {source_root}")
    emit(f"Storage: {storage_root}")
    emit(f"PDF files discovered: {len(pdf_paths)}")
    if not pdf_paths:
        return BulkImportSummary()

    database = Database(settings)
    try:
        if settings.auto_create_db:
            database.create_tables()
        with database.engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        service = AnnotationService(
            database=database,
            storage=AnnotationStorage(storage_root),
            renderer=PdfRenderer(
                dpi=settings.pdf_dpi,
                max_pages=settings.max_pages,
                max_render_pixels=settings.max_render_pixels,
            ),
            max_file_size_bytes=settings.max_file_size_bytes,
        )

        imported = 0
        duplicates = 0
        failed = 0
        pages_created = 0
        for position, pdf_path in enumerate(pdf_paths, start=1):
            relative_source = pdf_path.relative_to(source_root)
            try:
                with pdf_path.open("rb") as source:
                    created, document = service.import_pdf(
                        source=source,
                        original_filename=pdf_path.name,
                    )
                if created:
                    imported += 1
                    pages_created += document.page_count
                    emit(
                        f"[{position}/{len(pdf_paths)}] IMPORTED "
                        f"{relative_source} ({document.page_count} pages)"
                    )
                else:
                    duplicates += 1
                    emit(f"[{position}/{len(pdf_paths)}] DUPLICATE {relative_source}")
            except Exception as exc:  # Continue importing independent PDFs.
                failed += 1
                emit(f"[{position}/{len(pdf_paths)}] FAILED {relative_source}: {exc}")

        return BulkImportSummary(
            discovered=len(pdf_paths),
            imported=imported,
            duplicates=duplicates,
            failed=failed,
            pages_created=pages_created,
        )
    finally:
        database.close()


def main() -> None:
    try:
        summary = run_bulk_import(get_settings())
    except Exception as exc:
        print(f"Bulk import could not start: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    print("\nBulk import summary")
    print(f"  PDF discovered : {summary.discovered}")
    print(f"  Imported       : {summary.imported}")
    print(f"  Duplicates     : {summary.duplicates}")
    print(f"  Failed         : {summary.failed}")
    print(f"  Pages created  : {summary.pages_created}")
    if summary.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
