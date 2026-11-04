# GPU OCR + grounded LLM extraction

This branch keeps page classification and changes the information extraction
stage to:

```text
reviewed page classifications
  -> group adjacent note/applicant pages into a document case
  -> PP-OCRv6 medium on GPU
  -> immutable OCR token IDs + source bboxes in SQL Server
  -> Qwen3-4B QLoRA
  -> one case-level JSON + page-prefixed evidence token IDs
  -> backend resolves evidence token IDs to real page bboxes
```

The LLM is never trusted to generate coordinates. Highlight boxes always come
from stored OCR tokens.

## Selected models for RTX 3060 12 GB

```text
OCR detection:   PP-OCRv6_medium_det
OCR recognition: PP-OCRv6_medium_rec
Base LLM:        Qwen/Qwen3-4B
Fine-tuning:     QLoRA 4-bit, sequence length 4096, batch size 1
```

PP-OCRv6 medium is small enough for this GPU and supports Vietnamese, English
and other Latin-script languages. Qwen3-4B in 4-bit QLoRA mode is the selected
quality/memory balance for a 12 GB card. Use Qwen3-1.7B only as a fallback if
another process consumes enough VRAM to cause an out-of-memory error.

## 1. Install GPU runtimes

Use the project virtual environment only. CPU `paddlepaddle` and GPU
`paddlepaddle-gpu` must not coexist.

The CUDA 12.6 package index contains a Windows `cp313` wheel, so this setup is
compatible with the server's Python 3.13 virtual environment. The NVIDIA
driver can be newer than the CUDA runtime used by this wheel.

```cmd
python -m pip uninstall -y paddlepaddle paddlepaddle-gpu
python -m pip install -r requirements-ocr-gpu.txt
python -m pip install -r requirements-llm-gpu.txt
```

PyTorch CUDA should already be installed. Verify both runtimes:

```cmd
python -c "import paddle, torch; print('Paddle CUDA:', paddle.is_compiled_with_cuda()); print('Torch CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

## 2. Environment

```env
OCR_RUNTIME_DEVICE=auto
OCR_RUNTIME_GPU_DEVICE_ID=0
OCR_RUNTIME_ALLOW_CPU_FALLBACK=true
OCR_DETECTION_MODEL_NAME=PP-OCRv6_medium_det
OCR_RECOGNITION_MODEL_NAME=PP-OCRv6_medium_rec
OCR_DETECTION_MODEL_DIR=
OCR_RECOGNITION_MODEL_DIR=
OCR_MODEL_VERSION=pp-ocrv6-medium-v1
OCR_ENABLE_HPI=false
OCR_CPU_THREADS=8
OCR_MIN_CONFIDENCE=0.50
OCR_ENABLE_EXCLUSION_DETECTION=true
OCR_LAYOUT_MODEL_NAME=PP-DocLayout-L
OCR_LAYOUT_MODEL_DIR=
OCR_LAYOUT_THRESHOLD=0.50
OCR_EXCLUSION_REGION_TYPES=seal
OCR_AUTO_IGNORE_OVERLAP=0.70
OCR_ENABLE_ORIENTATION=true
OCR_ORIENTATION_MODEL_NAME=PP-LCNet_x1_0_doc_ori
OCR_ORIENTATION_MODEL_DIR=
OCR_ORIENTATION_MIN_CONFIDENCE=0.80
OCR_PREPROCESS_VERSION=orientation-content-v2
OCR_ENABLE_SELECTIVE_REFINEMENT=true
OCR_REFINEMENT_VERSION=selective-multiscale-v1
OCR_REFINEMENT_SCALES=2.0,3.0
OCR_REFINEMENT_CONFIDENCE_THRESHOLD=0.90
OCR_REFINEMENT_CANDIDATE_MIN_CONFIDENCE=0.70
OCR_REFINEMENT_AGREEMENT_SIMILARITY=0.96
OCR_REFINEMENT_MIN_CONFIDENCE_GAIN=0.05
OCR_REFINEMENT_MAX_REGIONS=32
OCR_REFINEMENT_MAX_SECONDS=15
OCR_REFINEMENT_PADDING_RATIO=0.12
OCR_REFINEMENT_MAX_CROP_SIDE=2400
OCR_REFINEMENT_PROTECT_DIGITS=true
OCR_TARGET_PAGE_TYPES=DIPLOMATIC_NOTE,APPLICANT_TABLE,APPLICANT_LIST

LLM_BASE_MODEL_ID=Qwen/Qwen3-4B
LLM_BASE_MODEL_DIR=models/llm/qwen3-4b
LLM_DEVICE=auto
LLM_GPU_DEVICE_ID=0
LLM_LOAD_IN_4BIT=true
LLM_MAX_SEQUENCE_LENGTH=4096
LLM_SCHEMA_VERSION=document-case-v1
LLM_DATASET_DIR=data/training/grounded_llm
LLM_TRAIN_BATCH_SIZE=1
LLM_GRADIENT_ACCUMULATION_STEPS=8
LLM_LORA_RANK=16
LLM_LORA_ALPHA=32
LLM_LORA_DROPOUT=0.05
```

`auto` selects GPU 0 when the corresponding CUDA runtime is available and
falls back to CPU only when the fallback flag permits it.

The page is still detected once by PP-OCRv6. After orientation, conservative
content crop and optional whole-page upscale, only suspicious Vietnamese,
small or low-confidence regions are cropped from that normalized page and
re-read at two scales in one Paddle batch. Text is replaced only when both
scales agree and the confidence policy passes. A different digit sequence is
never accepted, so document numbers, passport numbers and dates remain on the
original OCR result for human review. The original text, both candidates,
scores, decision and reason are stored in
`ocr_runs.preprocess_metadata_json.selective_refinement`; no new table is
required and source images are never overwritten.

## 3. Initialize second-pass audit, case and OCR tables

```cmd
python -m tools.init_page_label_audit_db
python -m tools.init_document_case_db
python -m tools.init_ocr_grounding_db
```

Tables:

```text
dbo.page_label_audits
dbo.document_cases
dbo.document_case_pages
dbo.case_extraction_annotations
dbo.ocr_runs
dbo.ocr_tokens
```

## 4. Verify target page labels a second time

Open the Angular route `/document-label-audit`. The queue is seeded only from
pages whose first-pass `review_status` is `REVIEWED` and whose human label is
`DIPLOMATIC_NOTE`, `APPLICANT_TABLE`, or `APPLICANT_LIST`. Model predictions do
not enter this queue. Each PDF is saved as one transaction; the original label
is preserved in `page_label_audits` and the audited label becomes the canonical
`annotation_pages.label`.

## 5. Build cases from audited page classifications

Preview grouping without writing anything:

```cmd
python -m tools.build_document_cases --dry-run
```

Create the cases after checking the preview:

```cmd
python -m tools.build_document_cases
```

An unrelated or unreviewed page is a case boundary. A
`DIPLOMATIC_NOTE` starts a case and immediately following
`APPLICANT_TABLE` / `APPLICANT_LIST` pages are attached to it. Applicant pages
without a preceding note are retained with `NEEDS_REVIEW` status. To recompute
grouping later, use `--rebuild`; old cases are deactivated rather than deleted.

`build_document_cases` ignores the first-pass label unless it has a completed,
active second-pass audit.

## 6. Download/verify models

This does not process any document:

```cmd
python -m tools.download_document_ai_models --component all --device auto
```

Paddle stores official OCR weights in its model cache. Qwen is stored under
`models/llm/qwen3-4b`, which is already ignored by Git.

## 7. Run a small OCR batch

### Chọn OCR engine

Mặc định hệ thống vẫn dùng PaddleOCR như trước:

```env
OCR_ENGINE=paddle
```

Để dùng pipeline Tesseract được port trực tiếp vào Python backend (không gọi
ScanManager và không mở thêm service), cài riêng nhóm phụ thuộc:

```cmd
python -m pip install -r requirements-ocr-tesseract.txt
```

Tesseract là chương trình native nên `pip` không cài được chính engine. Trên
Windows Server cần cài Tesseract 5 và chuẩn bị các file model:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
C:\Program Files\Tesseract-OCR\tessdata\vie.traineddata
C:\Program Files\Tesseract-OCR\tessdata\eng.traineddata
C:\Program Files\Tesseract-OCR\tessdata\deu.traineddata
C:\Program Files\Tesseract-OCR\tessdata\fra.traineddata
C:\Program Files\Tesseract-OCR\tessdata_best\vie.traineddata
```

Sau đó cấu hình:

```env
OCR_ENGINE=tesseract
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
TESSERACT_FAST_DATA_DIR=C:\Program Files\Tesseract-OCR\tessdata
TESSERACT_BEST_DATA_DIR=C:\Program Files\Tesseract-OCR\tessdata_best
TESSERACT_LANGUAGES=vie+eng+deu+fra
TESSERACT_ENABLE_VIE_REFINEMENT=true
TESSERACT_ENABLE_TABLE_CELLS=true
TESSERACT_TRY_ALL_ORIENTATIONS=true
```

Backend Python thực hiện toàn bộ orchestration, chuẩn hóa ảnh, thử chiều xoay,
phát hiện ô bảng bằng OpenCV, đọc lại vùng tiếng Việt bằng `tessdata_best` và
ghi token/bbox vào đúng các bảng OCR hiện có. Tesseract chạy CPU; tham số
`--device cuda` không làm Tesseract chuyển sang GPU. Nếu bật
`TESSERACT_USE_PADDLE_AUX_MODELS=true`, riêng model orientation/layout phụ trợ
có thể chạy theo `OCR_RUNTIME_DEVICE`, còn nhận dạng chữ vẫn là Tesseract CPU.

Nên thử hai trang trước khi chạy hàng loạt:

```cmd
python -m tools.run_grounded_ocr --limit 2 --pending-review-only
```

Pipeline Tesseract ưu tiên độ chính xác để tạo dữ liệu: khi không dùng model
orientation phụ trợ, nó có thể OCR thử cả bốn chiều; các vùng yếu còn được đọc
lại bằng model tiếng Việt tốt hơn. Vì `pytesseract` tạo tiến trình native cho
mỗi lượt đọc, tốc độ sẽ không bằng ScanManager giữ Tesseract engine nóng trong
cùng process. Chỉ dùng chế độ này cho batch làm sạch dữ liệu sau khi benchmark
trên server; production vẫn có thể giữ `OCR_ENGINE=paddle`.

```cmd
python -m tools.run_grounded_ocr --limit 10 --device auto
```

Only reviewed pages belonging to an active case are processed. Each OCR line
receives a stable page-local token ID such as `t_0001` and an original-page
bbox. Use `--case-id CASE_UUID` to process one case only.

Run all remaining eligible pages after checking the first batch:

```cmd
python -m tools.run_grounded_ocr --device auto
```

OCR vẫn lưu toàn bộ token gốc. `PP-DocLayout-L` phát hiện vùng `SEAL`; token có
ít nhất 70% diện tích nằm trong vùng đó được lưu với
`include_in_extraction=false` và không được xuất sang LLM. Tab Grounded OCR
Review cho phép khôi phục token hoặc khoanh thêm vùng `SEAL`, `SIGNATURE`,
`LOGO`.

Nếu cả trang OCR không đủ chất lượng, chọn **Bỏ qua trang**. OCR và lịch sử vẫn
được giữ trong DB, nhưng trang có `ocr_reviews.include_in_llm=false` sẽ không
được đưa vào `annotation_seed.jsonl`. Case chỉ bị bỏ hoàn toàn khi không còn
trang nào được duyệt để xuất LLM.

`--force` sẽ chạy lại cả trang đã có quyết định review. Không dùng tùy chọn này
trong luồng làm sạch dữ liệu thông thường:

```bat
python -m tools.run_grounded_ocr --device cuda --force
```

Để áp dụng pipeline orientation mới nhưng tuyệt đối không chạy lại trang đã
approve hoặc đã bỏ qua, dùng:

```bat
python -m tools.run_grounded_ocr --device cuda --pending-review-only
```

Chế độ này chạy cả trang chưa từng OCR và trang đã OCR bằng pipeline cũ nhưng
chưa có quyết định review. Trang đã chạy thành công với đúng
`OCR_PREPROCESS_VERSION` sẽ không chạy lặp lại, nên có thể tiếp tục lệnh sau khi
bị ngắt. Ảnh chuẩn hóa được lưu riêng dưới `ocr_preprocessed/`; PNG nguồn không
bị ghi đè.

## 8. Produce case-level annotation seeds

```cmd
python -m tools.export_llm_annotation_seed
```

Output:

```text
data/training/grounded_llm/annotation_seed.jsonl
```

Seeds are not yet trainable because they intentionally have
`assistant_target=null`. A reviewer or reviewed model-assistance workflow must
create `reviewed_targets.jsonl`. Prefixes such as `p2_` are added during export
and do not change the page-local token IDs stored in SQL Server:

```json
{"case_id":"CASE_UUID","target":{"schema_version":"document-case-v1","case_id":"CASE_UUID","document_id":"DOCUMENT_UUID","note":{},"applicants":[{"full_name":{"value":"ZIYI PAN","evidence":[{"page_number":2,"token_ids":["p2_t_0012"]}]},"passport_number":{"value":"EA3672591","evidence":[{"page_number":2,"token_ids":["p2_t_0018"]}]}}],"warnings":[]}}
```

Every evidence token is validated against all OCR tokens in that case.

## 9. Build SFT splits

```cmd
python -m tools.build_grounded_sft_dataset
```

Outputs are split by source document rather than case/page to prevent leakage:

```text
data/training/grounded_llm/sft/train.jsonl
data/training/grounded_llm/sft/val.jsonl
data/training/grounded_llm/sft/test.jsonl
```

The next phase is a review UI for creating `reviewed_targets.jsonl`, followed by
the QLoRA training script and grounded inference API.
## Teacher tạo nhãn JSON compact

Các bảng teacher chỉ được tạo trong `anotation_2`. Kết quả model lớn luôn đi vào
hàng đợi `PENDING`; chưa được dùng để train student cho tới khi người kiểm định
chuyển bản review sang `APPROVED`.

```powershell
# Tạo schema, chạy lặp lại an toàn.
python -m tools.init_teacher_extraction_db

# Dựng snapshot text từ các hồ sơ có OCR human-approved.
python -m tools.reconstruct_case_text --ready-only --include-incomplete --limit 5 --dry-run --print-text
python -m tools.reconstruct_case_text --ready-only --include-incomplete --limit 5

# Chỉ xem các hồ sơ đủ điều kiện; không ghi DB và không gọi model lớn.
python -m tools.generate_teacher_extractions --limit 5

# Nếu text mới dựng chưa được human review, phải bật rõ ràng cờ này.
python -m tools.generate_teacher_extractions --limit 5 --include-generated

# Tạo job và gọi teacher đồng bộ. Nên thử 1 hồ sơ trước.
python -m tools.generate_teacher_extractions --limit 1 --execute

# Gọi lại các job thất bại mà không thay đổi snapshot đầu vào.
python -m tools.generate_teacher_extractions --limit 5 --execute --retry-failed
```

Có thể chọn đúng một hồ sơ bằng `--case-id`, `--document-id` hoặc
`--filename 000001.pdf`. Mỗi job giữ nguyên snapshot text, schema, prompt và model;
mỗi lần gọi được lưu thành một attempt riêng nên không ghi đè lịch sử.

