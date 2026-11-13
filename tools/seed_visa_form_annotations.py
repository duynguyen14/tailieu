from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.db.database import Database
from app.services.storage import AnnotationStorage
from app.services.visa_form_annotation import (
    DEFAULT_TEMPLATE_VERSION,
    VISA_FORM_TEMPLATE_FIELDS,
    VisaFormAnnotationConflictError,
    VisaFormAnnotationService,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select VISA_APPLICATION_FORM pages and seed 19 grounded bbox annotations "
            "using OCR anchor detection and relative coordinates. "
            "The command is dry-run unless --execute is supplied."
        )
    )
    parser.add_argument("--execute", action="store_true", help="Write annotations to the DB")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--page-id", default=None)
    parser.add_argument("--template-version", default="visa-form-anchor-v1")
    parser.add_argument(
        "--include-unreviewed-pages",
        action="store_true",
        help="Also select form pages whose page label has not been human-reviewed",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Create a new PENDING template version and deactivate the current version",
    )
    parser.add_argument(
        "--use-ocr",
        dest="use_ocr",
        action="store_true",
        default=True,
        help="Use OCR tokens and anchor matching for dynamic bounding boxes (default: True)",
    )
    parser.add_argument(
        "--no-ocr",
        dest="use_ocr",
        action="store_false",
        help="Disable OCR anchor extraction and use static template coordinates",
    )
    parser.add_argument(
        "--run-ocr-if-missing",
        dest="run_ocr_if_missing",
        action="store_true",
        default=True,
        help="Execute OCR on-the-fly if no completed OCR run is stored in DB (default: True)",
    )
    parser.add_argument(
        "--no-run-ocr-if-missing",
        dest="run_ocr_if_missing",
        action="store_false",
        help="Do not trigger OCR if not already in DB",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    settings = get_settings()
    database = Database(settings)
    service = VisaFormAnnotationService(
        database=database,
        storage=AnnotationStorage(settings.annotation_storage_path),
    )
    try:
        pages = service.eligible_pages(
            limit=args.limit,
            page_id=args.page_id,
            include_unreviewed=args.include_unreviewed_pages,
            replace_existing=args.replace_existing,
        )
        created = 0
        for index, page in enumerate(pages, start=1):
            mode = "SEED" if args.execute else "DRY-RUN"
            try:
                fields = service.extract_page_fields(
                    page.id,
                    use_ocr=args.use_ocr,
                    run_ocr_if_missing=args.run_ocr_if_missing,
                )
                anchors_found = sum(1 for f in fields if f.anchor_found)
                values_extracted = sum(1 for f in fields if f.value_text)
                print(
                    f"[{index}/{len(pages)}] {mode} page={page.id} "
                    f"document={page.document_id} page_number={page.page_number} "
                    f"size={page.width}x{page.height} fields={len(fields)} "
                    f"anchors_detected={anchors_found}/{len(fields)} "
                    f"values_extracted={values_extracted}/{len(fields)}"
                )
                if args.execute:
                    try:
                        service.create_template_annotation(
                            page_id=page.id,
                            template_version=args.template_version,
                            replace_existing=args.replace_existing,
                            use_ocr=args.use_ocr,
                            run_ocr_if_missing=args.run_ocr_if_missing,
                        )
                        created += 1
                    except VisaFormAnnotationConflictError as exc:
                        print(f"    [SKIP] page={page.id} conflict: {exc}")
            except Exception as exc:
                print(f"[{index}/{len(pages)}] [ERROR] page={page.id} failed: {exc}", file=sys.stderr)
    finally:
        database.close()

    print(
        "Visa form bbox seeding finished: "
        f"eligible={len(pages)}, created={created}, dry_run={not args.execute}, "
        f"template={args.template_version!r}, use_ocr={args.use_ocr}."
    )


if __name__ == "__main__":
    main()
