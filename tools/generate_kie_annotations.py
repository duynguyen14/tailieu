from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from decimal import Decimal
from uuid import uuid4

from sqlalchemy import select, true, update

from app.core.config import get_settings
from app.db.database import Database
from app.db.models import (
    KieAnnotationRun,
    KieEntityAnnotation,
    KieFieldProposal,
    KiePageAnnotation,
    KieRelationAnnotation,
    utc_now,
)
from tools.audit_kie_alignment import (
    _audit_document,
    _entity_label,
    _load_case_tokens,
    _normalized_bbox,
    _review_statement,
)

ALGORITHM_VERSION = "kie-alignment-review-v1"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate persistent SER/RE review proposals from approved OCR and teacher JSON.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--minimum-score", type=float, default=0.88)
    parser.add_argument("--ambiguity-margin", type=float, default=0.035)
    parser.add_argument("--max-span-tokens", type=int, default=6)
    parser.add_argument("--top-candidates", type=int, default=5)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _source_hash(review_id, tokens, args):
    raw = json.dumps({
        "review": review_id,
        "ocr_reviews": sorted({token.review_id for token in tokens}),
        "algorithm": ALGORITHM_VERSION,
        "minimum_score": args.minimum_score,
        "ambiguity_margin": args.ambiguity_margin,
        "max_span_tokens": args.max_span_tokens,
        "top_candidates": args.top_candidates,
    }, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def run(args):
    settings = get_settings()
    if settings.db_name.strip().lower() != "anotation_2":
        raise RuntimeError("KIE generation must run with DB_NAME=anotation_2.")
    database = Database(settings)
    summary = {"discovered": 0, "created_runs": 0, "pages": 0, "proposals": 0, "entities": 0, "relations": 0, "skipped": 0, "failed": 0, "execute": args.execute}
    try:
        with database.session() as session:
            reviews = list(session.scalars(_review_statement(args.limit)).unique())
            summary["discovered"] = len(reviews)
            tokens_by_case = _load_case_tokens(session, sorted({review.job.case_id for review in reviews}))
            for position, review in enumerate(reviews, start=1):
                review_id = review.id
                original_filename = review.job.document.original_filename
                try:
                    tokens = tokens_by_case.get(review.job.case_id, [])
                    source_hash = _source_hash(review.id, tokens, args)
                    existing = session.scalar(select(KieAnnotationRun).where(KieAnnotationRun.source_hash == source_hash, KieAnnotationRun.is_active == true()))
                    if existing and not args.force:
                        summary["skipped"] += 1
                        print(f"[{position}/{len(reviews)}] SKIP file='{review.job.document.original_filename}' run={existing.id}")
                        continue
                    document_summary, alignments = _audit_document(
                        review, tokens, minimum_score=args.minimum_score,
                        ambiguity_margin=args.ambiguity_margin,
                        max_span_tokens=args.max_span_tokens,
                        top_candidates=args.top_candidates,
                    )
                    page_tokens = defaultdict(list)
                    for token in tokens:
                        page_tokens[token.page_id].append(token)
                    if not page_tokens:
                        summary["skipped"] += 1
                        continue
                    if not args.execute:
                        candidate_count = sum(len(item["candidates"]) for item in alignments)
                        print(f"[{position}/{len(reviews)}] DRY file='{document_summary['original_filename']}' pages={len(page_tokens)} proposals={len(alignments)} candidates={candidate_count}")
                        summary["pages"] += len(page_tokens)
                        summary["proposals"] += len(alignments)
                        summary["entities"] += candidate_count
                        continue
                    if args.force:
                        session.execute(update(KieAnnotationRun).where(KieAnnotationRun.teacher_review_id == review.id, KieAnnotationRun.is_active == true()).values(is_active=False, updated_at=utc_now()))
                    run_id = str(uuid4())
                    session.add(KieAnnotationRun(
                        id=run_id, teacher_review_id=review.id, case_id=review.job.case_id,
                        document_id=review.job.document_id, algorithm_version=ALGORITHM_VERSION,
                        source_hash=source_hash, status="GENERATED", is_active=True,
                    ))
                    pages = {}
                    for page_id, page_items in page_tokens.items():
                        first = sorted(page_items, key=lambda item: item.token_order)[0]
                        page_annotation = KiePageAnnotation(
                            id=str(uuid4()), run_id=run_id, page_id=page_id,
                            ocr_review_id=first.review_id, page_order=first.page_order,
                            page_number=first.page_number, image_relative_path=first.image_relative_path,
                            image_width=first.page_width, image_height=first.page_height,
                            orientation_angle=first.orientation_angle,
                            preprocess_metadata_json=json.dumps(first.preprocess_metadata, ensure_ascii=False) if first.preprocess_metadata else None,
                            ser_status="PENDING", re_status="BLOCKED",
                        )
                        session.add(page_annotation)
                        pages[page_id] = page_annotation
                    session.flush()
                    # Flush proposals before entities. The entity mapper stores
                    # only the proposal UUID (no ORM relationship), therefore
                    # SQLAlchemy's unit-of-work cannot reliably order these two
                    # bulk INSERTs for SQL Server by itself.
                    proposal_ids: dict[str, str] = {}
                    for alignment in alignments:
                        proposal_id = str(uuid4())
                        proposal_ids[alignment["field_path"]] = proposal_id
                        session.add(KieFieldProposal(
                            id=proposal_id, run_id=run_id, field_path=alignment["field_path"],
                            entity_type=alignment["entity_type"], entity_index=alignment["entity_index"],
                            field_name=alignment["field"], expected_value=None if alignment["value"] is None else str(alignment["value"]),
                            match_status=alignment["status"],
                            chosen_json=json.dumps(alignment["chosen"], ensure_ascii=False) if alignment["chosen"] else None,
                            candidates_json=json.dumps(alignment["candidates"], ensure_ascii=False),
                        ))
                    session.flush()

                    entity_rows = []
                    for alignment in alignments:
                        proposal_id = proposal_ids[alignment["field_path"]]
                        for rank, candidate in enumerate(alignment["candidates"], start=1):
                            page = pages.get(candidate["page_id"])
                            if page is None:
                                continue
                            bbox = tuple(candidate["bbox"])
                            normalized = _normalized_bbox(bbox, page.image_width, page.image_height)
                            entity = KieEntityAnnotation(
                                id=str(uuid4()), page_annotation_id=page.id,
                                field_proposal_id=proposal_id,
                                entity_key=f"{alignment['field_path']}#{rank}",
                                label=_entity_label(alignment["entity_type"], alignment["field"]),
                                value=candidate["text"], field_path=alignment["field_path"],
                                applicant_index=alignment["entity_index"],
                                token_ids_json=json.dumps(candidate["token_ids"]),
                                x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3],
                                nx1=normalized[0], ny1=normalized[1], nx2=normalized[2], ny2=normalized[3],
                                confidence=Decimal(str(candidate["selection_score"])), source="MODEL",
                                candidate_rank=rank, is_active=True,
                            )
                            session.add(entity)
                            entity_rows.append(entity)
                    session.flush()
                    grouped = defaultdict(list)
                    for entity in entity_rows:
                        if entity.applicant_index is not None and entity.candidate_rank == 1:
                            grouped[(entity.page_annotation_id, entity.applicant_index)].append(entity)
                    relation_count = 0
                    for (page_annotation_id, applicant_index), items in grouped.items():
                        anchors = [item for item in items if item.field_path and item.field_path.endswith((".passport_number", ".full_name"))]
                        if not anchors:
                            continue
                        anchor = anchors[0]
                        for item in items:
                            if item.id == anchor.id:
                                continue
                            session.add(KieRelationAnnotation(
                                id=str(uuid4()), page_annotation_id=page_annotation_id,
                                head_entity_id=anchor.id, tail_entity_id=item.id,
                                relation_type="BELONGS_TO_APPLICANT", applicant_index=applicant_index,
                                confidence=min(anchor.confidence, item.confidence), source="MODEL", is_active=True,
                            ))
                            relation_count += 1
                    summary["created_runs"] += 1
                    summary["pages"] += len(pages)
                    summary["proposals"] += len(alignments)
                    summary["entities"] += len(entity_rows)
                    summary["relations"] += relation_count
                    print(f"[{position}/{len(reviews)}] CREATED file='{document_summary['original_filename']}' pages={len(pages)} proposals={len(alignments)} entities={len(entity_rows)} relations={relation_count}")
                except Exception as exc:
                    summary["failed"] += 1
                    print(
                        f"[{position}/{len(reviews)}] FAILED "
                        f"file='{original_filename}' review={review_id}: {exc}"
                    )
                    raise
    finally:
        database.close()
    print("KIE proposal generation finished:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    try:
        run(parse_args())
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"KIE proposal generation could not start: {exc}")
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
