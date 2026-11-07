# Production PDF extraction API

The endpoint `POST /api/v1/document-extractions` accepts one scanned PDF and
runs the complete synchronous pipeline:

1. render PDF pages;
2. detect and normalize page orientation;
3. classify every page with the exported ONNX document classifier;
4. OCR configured business pages with the custom Paddle detection/recognition models;
5. reconstruct page text deterministically from OCR text and bounding boxes;
6. extract compact v8 fields with the merged Gemma model;
7. validate and map the compact response to `CompactDocumentExtraction`.

Seal detection is not part of this runtime. `PaddleLayoutDetector` is never
loaded and OCR tokens are never removed because they overlap a seal.

## Required model artifacts

Export the trained page classifier checkpoint once:

```bat
python -m tools.export_classifier_onnx --checkpoint output\document_classifier_anotation_2_v3\best_model.pt
```

The production host must then contain:

```text
models/page_classifier/model.onnx
models/page_classifier/labels.json
models/ocr/recognition/latin_ppocrv5_mobile_phase2_v2/inference/
models/student_extractor/gemma3_270m_shortkey_v8/merged_model/
```

## Environment

Use the relevant values below in `.env` (adjust paths and GPU ID if needed):

```dotenv
PAGE_CLASSIFIER_MODEL_PATH=models/page_classifier/model.onnx
PAGE_CLASSIFIER_LABELS_PATH=models/page_classifier/labels.json

OCR_RUNTIME_DEVICE=cuda
OCR_RUNTIME_GPU_DEVICE_ID=0
OCR_RUNTIME_ALLOW_CPU_FALLBACK=false
OCR_DETECTION_MODEL_NAME=PP-OCRv6_medium_det
OCR_RECOGNITION_MODEL_NAME=latin_PP-OCRv5_mobile_rec
OCR_DETECTION_MODEL_DIR=
OCR_RECOGNITION_MODEL_DIR=models/ocr/recognition/latin_ppocrv5_mobile_phase2_v2/inference

PRODUCTION_EXTRACTION_ENABLED=true
PRODUCTION_WARMUP_MODELS=true
PRODUCTION_REQUIRE_GPU=true
PRODUCTION_GPU_DEVICE_ID=0
PRODUCTION_STUDENT_MODEL_DIR=models/student_extractor/gemma3_270m_shortkey_v8/merged_model
PRODUCTION_STUDENT_PRECISION=bf16
PRODUCTION_STUDENT_MAX_INPUT_TOKENS=4096
PRODUCTION_STUDENT_MAX_NEW_TOKENS=512
PRODUCTION_OCR_PAGE_TYPES=DIPLOMATIC_NOTE,APPLICANT_TABLE,APPLICANT_LIST
PRODUCTION_OCR_ENABLE_REFINEMENT=false
PRODUCTION_INCLUDE_RECONSTRUCTED_TEXT=false
```

`OCR_RUNTIME_GPU_DEVICE_ID` and `PRODUCTION_GPU_DEVICE_ID` must match. The app
fails at startup rather than silently placing one of the models on CPU.

## Start and verify

Install the host-specific CUDA PyTorch wheel first, then:

```bat
pip install -r requirements-production-extraction.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Use exactly one Uvicorn worker for each configured GPU. Multiple workers each
load their own copy of all model weights and can exhaust VRAM.

Readiness becomes true only after model loading and warm-up finish:

```bat
curl http://127.0.0.1:8000/api/v1/document-extractions/readiness
```

Example request:

```bat
curl -X POST http://127.0.0.1:8000/api/v1/document-extractions -F "file=@scan.pdf;type=application/pdf"
```

The response contains the backend-validated full business object in
`extraction`, the original compact v8 model result in `compact_model_output`,
page-level classifier/OCR diagnostics, and stage timings. The API is
single-flight: while the GPU worker is processing a PDF, another request gets
HTTP 429 instead of competing for VRAM.

