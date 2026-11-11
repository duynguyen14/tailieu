from __future__ import annotations

import argparse

from app.core.config import get_settings
from app.db.database import Database
from app.services.storage import AnnotationStorage
from app.services.visa_form_annotation import (
    DEFAULT_TEMPLATE_VERSION,
    VISA_FORM_TEMPLATE_FIELDS,
    VisaFormAnnotationService,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select VISA_APPLICATION_FORM pages and seed 19 template bbox annotations. "
            "The command is dry-run unless --execute is supplied."
        )
    )
    parser.add_argument("--execute", action="store_true", help="Write annotations to the DB")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--page-id", default=None)
    parser.add_argument("--template-version", default=DEFAULT_TEMPLATE_VERSION)
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
            print(
                f"[{index}/{len(pages)}] {mode} page={page.id} "
                f"document={page.document_id} page_number={page.page_number} "
                f"size={page.width}x{page.height} fields={len(VISA_FORM_TEMPLATE_FIELDS)}"
            )
            if args.execute:
                service.create_template_annotation(
                    page_id=page.id,
                    template_version=args.template_version,
                    replace_existing=args.replace_existing,
                )
                created += 1
    finally:
        database.close()

    print(
        "Visa form bbox seeding finished: "
        f"eligible={len(pages)}, created={created}, dry_run={not args.execute}, "
        f"template={args.template_version!r}."
    )


if __name__ == "__main__":
    main()
