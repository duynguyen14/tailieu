# Split production PDF extraction API

The public endpoint `POST /api/v1/document-extractions` accepts one scanned PDF
and returns backend-validated multi-source business JSON. Production uses
isolated localhost services so Paddle and Torch never share a Python
environment or CUDA/cuDNN packages.

```text
Frontend
   -> public student API :8002
      -> private OCR API 127.0.0.1:8001
         -> page classification (ONNX CPU)
         -> orientation + OCR (Paddle GPU)
         -> PASSPORT_FRONT pages -> MRZ API 127.0.0.1:8007 (in-memory, parallel)
         -> deterministic text reconstruction
      <- reconstructed text + page diagnostics
      -> Qwen student through vLLM 127.0.0.1:8005
      -> catalog mapping + multi-source aggregation
   <- validated multi-source-v1 JSON
```

Seal detection is not loaded by either service.

## Required model artifacts

```text
models/page_classifier/model.onnx
models/page_classifier/labels.json
output/ocr_recognition_latin_ppocrv5_mobile_phase3_v3/inference/
output/student_extractor_qwen3_1_7b_augmented_v9/merged_model/
models/passport_mrz/detector.onnx
models/passport_mrz/recognition/
```

## Environment

Both processes read the same `.env`. A recommended three-GPU host assignment is
Qwen on physical GPU 2, Paddle on physical GPU 3, and the small ONNX classifier
on CPU:

```dotenv
PAGE_CLASSIFIER_MODEL_PATH=models/page_classifier/model.onnx
PAGE_CLASSIFIER_LABELS_PATH=models/page_classifier/labels.json

OCR_RUNTIME_DEVICE=cuda
OCR_RUNTIME_GPU_DEVICE_ID=3
OCR_RUNTIME_ALLOW_CPU_FALLBACK=false
OCR_DETECTION_MODEL_NAME=PP-OCRv6_medium_det
OCR_DETECTION_MODEL_DIR=
OCR_RECOGNITION_MODEL_NAME=latin_PP-OCRv5_mobile_rec
OCR_RECOGNITION_BATCH_SIZE=16
PRODUCTION_CATALOG_MAPPING_ENABLED=true
PRODUCTION_CATALOG_PATH=config/business_catalog.v1.json
OCR_RECOGNITION_MODEL_DIR=output/ocr_recognition_latin_ppocrv5_mobile_phase3_v3/inference

PRODUCTION_EXTRACTION_ENABLED=true
PRODUCTION_SPLIT_RUNTIME=true
PRODUCTION_OCR_SERVICE_ENABLED=true
PRODUCTION_OCR_SERVICE_URL=http://127.0.0.1:8001/internal/v1/ocr
PRODUCTION_OCR_SERVICE_TIMEOUT_SECONDS=300
PRODUCTION_OCR_CLASSIFIER_DEVICE=cpu
PRODUCTION_PASSPORT_SERVICE_ENABLED=true
PRODUCTION_PASSPORT_SERVICE_URL=http://127.0.0.1:8007/internal/v1/passports
PRODUCTION_PASSPORT_SERVICE_TIMEOUT_SECONDS=5
PRODUCTION_PASSPORT_PAGE_TYPES=PASSPORT_FRONT
PRODUCTION_WARMUP_MODELS=true
PRODUCTION_REQUIRE_GPU=true
PRODUCTION_GPU_DEVICE_ID=2
PRODUCTION_STUDENT_MODEL_DIR=output/student_extractor_qwen3_1_7b_augmented_v9/merged_model
PRODUCTION_STUDENT_PRECISION=bf16
PRODUCTION_STUDENT_MAX_INPUT_TOKENS=4096
PRODUCTION_STUDENT_MAX_NEW_TOKENS=1024
PRODUCTION_OCR_PAGE_TYPES=DIPLOMATIC_NOTE,APPLICANT_TABLE,APPLICANT_LIST
PRODUCTION_OCR_ENABLE_REFINEMENT=false
PRODUCTION_INCLUDE_RECONSTRUCTED_TEXT=false

PASSPORT_MRZ_SERVICE_ENABLED=true
PASSPORT_MRZ_DETECTOR_MODEL_PATH=models/passport_mrz/detector.onnx
PASSPORT_MRZ_DETECTOR_DEVICE=cpu
PASSPORT_MRZ_OCR_MODEL_DIR=models/passport_mrz/recognition
PASSPORT_MRZ_OCR_DICT_PATH=models/passport_mrz/recognition/mrz_dict.txt
PASSPORT_MRZ_OCR_DEVICE=cuda
PASSPORT_MRZ_OCR_GPU_DEVICE_ID=2
PASSPORT_MRZ_API_PORT=8007
```

Do not set `CUDA_VISIBLE_DEVICES`: both settings use physical GPU IDs.

## Install isolated dependencies

The existing OCR environment receives only Paddle/ONNX dependencies:

```bash
.venv-ocr/bin/python -m pip install -r requirements-production-ocr-service.txt
```

The existing student environment receives only API/Transformers dependencies;
install its CUDA-enabled Torch wheel separately before this command:

```bash
.venv-student/bin/python -m pip install -r requirements-production-extraction.txt
```

Create a third environment for the private MRZ service. It contains Paddle and
ONNX Runtime, but no Torch/Ultralytics:

```bash
python3 -m venv .venv-passport-mrz
.venv-passport-mrz/bin/python -m pip install -U pip
.venv-passport-mrz/bin/python -m pip install -r requirements-passport-mrz-service.txt
```

No environment contains both `paddlepaddle-gpu` and `torch`, so their pinned
CUDA/cuDNN package versions cannot conflict.

## Start and verify

Start the private MRZ service and private OCR service first. Bind both to
loopback so they are not reachable from other machines:

```bash
.venv-passport-mrz/bin/python -m uvicorn app.passport_mrz_main:app \
  --host 127.0.0.1 --port 8007 --workers 1
```

Verify MRZ preload:

```bash
curl -s http://127.0.0.1:8007/internal/v1/passports/readiness \
  | python3 -m json.tool
```

```bash
.venv-ocr/bin/python -m uvicorn app.production_ocr_main:app \
  --host 127.0.0.1 --port 8001 --workers 1
```

Verify OCR preload:

```bash
curl -s http://127.0.0.1:8001/internal/v1/ocr/readiness | python3 -m json.tool
```

Then start the public student/orchestrator API:

```bash
.venv-student/bin/python -m uvicorn app.production_main:app \
  --host 0.0.0.0 --port 8002 --workers 1
```

The public API reports ready only when both the student runtime and private OCR
service are ready:

```bash
curl -s http://127.0.0.1:8002/api/v1/document-extractions/readiness \
  | python3 -m json.tool
```

Example request:

```bash
curl -X POST http://127.0.0.1:8002/api/v1/document-extractions \
  -F "file=@scan.pdf;type=application/pdf"
```

Both services are single-flight. Concurrent work receives HTTP 429 rather than
competing for GPU memory. The production path is stateless and does not write
the extraction to the annotation database.
