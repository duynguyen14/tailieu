# Document classifier training

Các công cụ trong tài liệu này chạy hoàn toàn tại máy nội bộ. Ảnh, nhãn và checkpoint
không được upload bởi các script.

## Thành phần

- `tools/export_reviewed_dataset.py`: đọc các page `REVIEWED` từ SQL Server, kiểm tra
  ảnh, loại ảnh trùng hoàn toàn và xuất dataset.
- `tools/train_document_classifier.py`: fine-tune MobileNetV3-Small bằng CPU hoặc CUDA.
- `tools/export_classifier_onnx.py`: chuyển checkpoint tốt nhất thành ONNX và tạo
  `labels.json` dùng cho inference.
- `requirements-training.txt`: dependency tách riêng khỏi API runtime.

Không script nào tự chạy khi API khởi động.

## Cài môi trường training

Tạo virtual environment Python 3.11 hoặc 3.13 riêng trên máy training. Với GPU, cài
bản PyTorch CUDA chính thức phù hợp driver của server, sau đó cài các dependency còn
lại trong `requirements-training.txt`.

## Cấu hình

Sao chép các biến `DATASET_*`, `TRAIN_*` và `CLASSIFIER_*` từ `.env.example` vào
`.env` của server. Kết nối SQL Server và `ANNOTATION_STORAGE_DIR` vẫn dùng cấu hình
backend hiện tại.

`DATASET_COPY_MODE=copy` tạo dataset độc lập và dễ mang sang ổ khác. Chế độ
`hardlink` tiết kiệm dung lượng nhưng chỉ dùng được khi nguồn và đích cùng volume.

## Trình tự chạy trên server

```powershell
python -m tools.export_reviewed_dataset
python -m tools.train_document_classifier
python -m tools.export_classifier_onnx
```

Có thể ghi đè đường dẫn bằng tham số mà không sửa `.env`:

```powershell
python -m tools.export_reviewed_dataset --output-dir D:\datasets\document_classifier_v1
python -m tools.train_document_classifier --dataset-dir D:\datasets\document_classifier_v1
python -m tools.export_classifier_onnx --checkpoint output\document_classifier\best_model.pt
```

Script export từ chối ghi vào thư mục đã có dữ liệu để tránh ghi đè dataset cũ.
Muốn tạo phiên bản mới, hãy dùng một thư mục output mới.

## Đầu ra

Dataset:

```text
data/training/document_classifier/
  images/<LABEL>/*.png
  manifest.csv
  train.csv
  val.csv
  test.csv
  labels.json
  statistics.json
```

Training:

```text
output/document_classifier/
  best_model.pt
  history.json
  test_metrics.json
```

Deployment:

```text
models/page_classifier/
  model.onnx
  labels.json
```

Việc chia train/validation/test được thực hiện theo `document_id`; các trang của cùng
một PDF luôn nằm trong cùng một split để tránh rò rỉ dữ liệu giữa train và test.
