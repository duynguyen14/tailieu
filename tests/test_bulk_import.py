from __future__ import annotations

import tempfile
from pathlib import Path

import pymupdf
import pytest
from sqlalchemy import func, select

from app.core.config import PROJECT_ROOT, Settings
from app.db.database import Database
from app.db.models import AnnotationDocument, AnnotationPage
from tools.bulk_import_pdfs import run_bulk_import, validate_source_root


def _pdf_bytes(text: str, page_count: int) -> bytes:
    document = pymupdf.open()
    for page_number in range(1, page_count + 1):
        page = document.new_page(width=300, height=400)
        page.insert_text((40, 60), f"{text} - page {page_number}")
    result = document.tobytes()
    document.close()
    return result


def _test_settings(root: Path, source: Path) -> Settings:
    database_path = (root / "annotation.db").as_posix()
    return Settings(
        _env_file=None,
        database_url=f"sqlite:///{database_path}",
        auto_create_db=True,
        annotation_storage_dir=(root / "storage").relative_to(PROJECT_ROOT),
        bulk_import_root=source,
        pdf_dpi=100,
        max_render_pixels=5_000_000,
    )


def test_bulk_import_recurses_renders_and_inserts_unreviewed_pages() -> None:
    temp_parent = PROJECT_ROOT / "tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bulk-import-test-", dir=temp_parent) as temp_dir:
        root = Path(temp_dir)
        source = root / "source"
        nested = source / "level-1" / "level-2"
        nested.mkdir(parents=True)
        (source / "first.PDF").write_bytes(_pdf_bytes("first", 1))
        (nested / "second.pdf").write_bytes(_pdf_bytes("second", 2))
        (nested / "ignore.txt").write_text("not a PDF", encoding="utf-8")
        settings = _test_settings(root, source)
        messages: list[str] = []

        summary = run_bulk_import(settings, emit=messages.append)

        assert summary.discovered == 2
        assert summary.imported == 2
        assert summary.duplicates == 0
        assert summary.failed == 0
        assert summary.pages_created == 3

        database = Database(settings)
        try:
            with database.session() as session:
                documents = list(session.scalars(select(AnnotationDocument)).all())
                pages = list(session.scalars(select(AnnotationPage)).all())
                assert len(documents) == 2
                assert len(pages) == 3
                assert all(document.review_status == "UNLABELED" for document in documents)
                assert all(page.review_status == "UNLABELED" for page in pages)
                assert all(page.label is None for page in pages)
                assert all(not Path(page.image_relative_path).is_absolute() for page in pages)
                assert all(
                    settings.annotation_storage_path.joinpath(page.image_relative_path).is_file()
                    for page in pages
                )
        finally:
            database.close()


def test_bulk_import_skips_duplicate_pdf_content() -> None:
    temp_parent = PROJECT_ROOT / "tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bulk-duplicate-test-", dir=temp_parent) as temp_dir:
        root = Path(temp_dir)
        source = root / "source"
        source.mkdir()
        content = _pdf_bytes("same", 1)
        (source / "one.pdf").write_bytes(content)
        (source / "two.pdf").write_bytes(content)
        settings = _test_settings(root, source)

        summary = run_bulk_import(settings, emit=lambda _: None)

        assert summary.discovered == 2
        assert summary.imported == 1
        assert summary.duplicates == 1
        assert summary.pages_created == 1

        database = Database(settings)
        try:
            with database.session() as session:
                document_count = session.scalar(select(func.count(AnnotationDocument.id)))
                assert document_count == 1
        finally:
            database.close()


def test_bulk_import_rejects_overlapping_source_and_storage() -> None:
    root = PROJECT_ROOT / "tmp" / "overlap-check"
    root.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="must not overlap"):
        validate_source_root(root, root / "generated")
