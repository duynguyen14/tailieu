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
