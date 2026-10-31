# Table structure baseline and review backend

This phase runs table detection and table structure recognition only. OCR and
cell text matching are intentionally not included yet.

## Data flow

```text
reviewed APPLICANT_TABLE page
  -> PaddleOCR LayoutDetection (table regions)
  -> crop each table
  -> PaddleOCR TableStructureRecognition / SLANet_plus
  -> derive header rows, data rows, columns and cells
  -> store MODEL annotation in SQL Server
  -> review through the API
  -> store a new HUMAN annotation version
```

The original model prediction is never overwritten by a human review.

## 1. Initialize SQL Server schema

```cmd
python -m tools.init_table_annotation_db
```

This creates, when missing:

```text
dbo.table_annotations
dbo.table_regions
dbo.table_structure_elements
```

The migration is idempotent and does not delete existing rows.

## 2. Install table inference dependencies

Install exactly one PaddlePaddle inference engine in the active virtual
environment. The wrapper requirement files install both the selected engine
and the shared table dependencies in one command.

CPU:

```cmd
python -m pip install -r requirements-table-cpu.txt
```

GPU for a compatible NVIDIA driver:

```cmd
python -m pip install -r requirements-table-gpu.txt
```

Do not install both wrapper files in the same virtual environment. The base
`requirements-table.txt` intentionally excludes the Paddle engine so it can be
shared by CPU and GPU deployments.

Verify the engine:

```cmd
python -c "import paddle; print(paddle.__version__); print(paddle.device.get_device())"
```

## 3. Configure `.env`

```env
TABLE_DEVICE=cpu
TABLE_GPU_DEVICE_ID=0
TABLE_ALLOW_CPU_FALLBACK=true

TABLE_LAYOUT_MODEL_NAME=PP-DocLayout-L
TABLE_LAYOUT_MODEL_DIR=
TABLE_STRUCTURE_MODEL_NAME=SLANet_plus
TABLE_STRUCTURE_MODEL_DIR=
TABLE_MODEL_VERSION=paddle-table-baseline-v1

TABLE_LAYOUT_THRESHOLD=0.50
TABLE_CPU_THREADS=8
TABLE_ENABLE_HPI=false
TABLE_ASSUME_FIRST_ROW_HEADER=true
```

Keep `TABLE_ENABLE_HPI=false` for the first compatibility test. It can be
enabled later after installing and validating the optional high-performance
inference backends on the target server.

When the model directories are empty, PaddleOCR downloads and caches the
official pretrained models on first use. Set a model directory to use a local
exported inference model instead. If Hugging Face is unavailable, set:

```env
PADDLE_PDX_MODEL_SOURCE=BOS
```

## 4. Run a small baseline

```cmd
python -m tools.infer_table_structure --limit 10
```

Useful variants:

```cmd
python -m tools.infer_table_structure --limit 10 --device cpu
python -m tools.infer_table_structure --page-id PAGE_UUID
python -m tools.infer_table_structure --page-id PAGE_UUID --force
```

`--force` can replace an active MODEL prediction with a new MODEL version. It
never replaces an active HUMAN-reviewed annotation.

Only pages satisfying both conditions are selected:

```text
annotation_pages.review_status = REVIEWED
annotation_pages.label = APPLICANT_TABLE
```

## 5. Review API

```text
GET /api/v1/table-annotation/review/next
GET /api/v1/table-annotation/statistics
GET /api/v1/table-annotation/{annotation_id}
PUT /api/v1/table-annotation/{annotation_id}/review
```

The detail response contains the original page image as base64, page width and
height, and table/row/column/cell bounding boxes in original page pixels.

Example review request:

```json
{
  "reviewed_by": "thai",
  "review_note": "Adjusted the first table boundary",
  "tables": [
    {
      "table_index": 0,
      "bbox": [100, 420, 1540, 2050],
      "confidence": 0.97,
      "elements": [
        {
          "element_type": "HEADER_ROW",
          "row_index": 0,
          "bbox": [100, 420, 1540, 500]
        },
        {
          "element_type": "DATA_ROW",
          "row_index": 1,
          "bbox": [100, 500, 1540, 590]
        },
        {
          "element_type": "COLUMN",
          "column_index": 0,
          "bbox": [100, 420, 500, 2050]
        },
        {
          "element_type": "CELL",
          "row_index": 1,
          "column_index": 0,
          "bbox": [100, 500, 500, 590]
        }
      ]
    }
  ]
}
```

Submitting the request creates a new active HUMAN version and marks the prior
MODEL version inactive. Bounding boxes are validated against the original page
dimensions and their parent table region.
