from __future__ import annotations

import json
from pathlib import Path

from pypdf import PdfReader, PdfWriter


ROOT = Path(r"D:\Project Real\THI_THUC\EXTRACT_DOCUMENT")
SOURCE_DIR = ROOT / "CamScanner-260703105539000"
OUTPUT_DIR = ROOT / "output" / "pdf" / "document_classifier_demo"


# These are verified ground-truth labels for the demo. Page numbers are 1-based.
# In production, `label` is supplied by the trained page-classification model.
DEMO_PAGES = [
    {
        "bundle_page": 1,
        "label": "DIPLOMATIC_NOTE",
        "source_file": "CamScanner 3-7-26 10.38.pdf",
        "source_page": 1,
        "reason": "Diplomatic note with note number, narrative body, recipient and seal.",
    },
    {
        "bundle_page": 2,
        "label": "APPLICANT_TABLE",
        "source_file": "CamScanner 3-7-26 10.38.pdf",
        "source_page": 12,
        "reason": "Applicant table with headers, rows and passport-related columns.",
    },
    {
        "bundle_page": 3,
        "label": "VISA_APPLICATION",
        "source_file": "CamScanner 3-7-26 10.38.pdf",
        "source_page": 14,
        "reason": "Vietnam visa application form with labelled key-value fields.",
    },
    {
        "bundle_page": 4,
        "label": "PASSPORT",
        "source_file": "CamScanner 3-7-26 10.40.pdf",
        "source_page": 14,
        "reason": "Passport biographical page with portrait, document fields and MRZ.",
    },
    {
        "bundle_page": 5,
        "label": "ID_CARD",
        "source_file": "CamScanner 3-7-26 10.38.pdf",
        "source_page": 13,
        "reason": "Diplomatic identification card, front and back.",
    },
    {
        "bundle_page": 6,
        "label": "UNKNOWN",
        "source_file": "CamScanner 3-7-26 10.40.pdf",
        "source_page": 22,
        "reason": "Notarial certificate outside the current five supported classes.",
    },
]


def add_source_page(writer: PdfWriter, item: dict) -> None:
    source = SOURCE_DIR / item["source_file"]
    reader = PdfReader(str(source))
    page_index = item["source_page"] - 1
    if page_index < 0 or page_index >= len(reader.pages):
        raise IndexError(f"Invalid page {item['source_page']} for {source}")
    writer.add_page(reader.pages[page_index])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    by_class_dir = OUTPUT_DIR / "by_class"
    by_class_dir.mkdir(parents=True, exist_ok=True)

    bundle = PdfWriter()
    for item in DEMO_PAGES:
        add_source_page(bundle, item)
    bundle_path = OUTPUT_DIR / "demo_mixed_6_classes.pdf"
    with bundle_path.open("wb") as stream:
        bundle.write(stream)

    outputs = []
    for item in DEMO_PAGES:
        writer = PdfWriter()
        add_source_page(writer, item)
        output_name = f"{item['bundle_page']:02d}_{item['label']}.pdf"
        output_path = by_class_dir / output_name
        with output_path.open("wb") as stream:
            writer.write(stream)
        outputs.append(str(output_path))

    manifest = {
        "demo_type": "ground_truth_routing_demo",
        "warning": (
            "Labels are manually verified ground truth. They are not predictions from a "
            "trained classifier because no trained classifier exists in the project yet."
        ),
        "input_bundle": str(bundle_path),
        "pages": DEMO_PAGES,
        "routed_outputs": outputs,
        "production_contract": {
            "classifier_input": "one rendered page image",
            "classifier_output": {
                "label": "one of the configured document classes",
                "confidence": "float from 0 to 1",
            },
            "router_behavior": "append the original PDF page to the PDF for its predicted label",
            "fallback_behavior": "route low-confidence or unsupported pages to UNKNOWN",
        },
    }
    manifest_path = OUTPUT_DIR / "classification_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(bundle_path)
    for output in outputs:
        print(output)
    print(manifest_path)


if __name__ == "__main__":
    main()
