# Document classifier training

Các công cụ trong tài liệu này chạy hoàn toàn tại máy nội bộ. Ảnh, nhãn và checkpoint
không được upload bởi các script.

## Thành phần

- `tools/export_reviewed_dataset.py`: đọc các page `REVIEWED` từ SQL Server, kiểm tra
  ảnh, loại ảnh trùng hoàn toàn và xuất dataset.
- `tools/train_document_classifier.py`: fine-tune MobileNetV3-Small bằng CPU hoặc CUDA.
  Terminal hiển thị progress bar theo batch, loss, learning rate, ETA và VRAM CUDA.
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

## Chạy model trên các page chưa review

Sau khi training hoàn tất, export checkpoint tốt nhất:

```powershell
python -m tools.export_classifier_onnx
```

Khởi tạo các cột prediction trong SQL Server (chạy nhiều lần an toàn):

```powershell
python -m tools.migrate_page_predictions
```

Cài ONNX Runtime CPU:

```powershell
python -m pip install -r requirements-inference.txt
```

Nếu dùng CUDA, cài `onnxruntime-gpu` thay cho `onnxruntime`; không cài đồng thời cả
hai package trong cùng virtual environment.

Chạy thử 100 page chưa review:

```powershell
python -m tools.predict_unreviewed_pages --limit 100 --device cpu
```

Sau khi kiểm tra frontend/API, chạy toàn bộ số page còn lại:

```powershell
python -m tools.predict_unreviewed_pages --device cpu
```

Tool chỉ chọn `annotation_pages.review_status = UNLABELED`. Nó cập nhật `label` để
frontend chọn sẵn và lưu thêm nhãn/confidence/top-3/model version vào các cột
`model_prediction_*`, nhưng không thay đổi `review_status`, `reviewed_by` hoặc
`reviewed_at`.

Page đã được dự đoán bởi cùng `PAGE_CLASSIFIER_MODEL_VERSION` sẽ được bỏ qua. Dùng
`--force` nếu cần chạy lại model trên các page vẫn còn `UNLABELED`:

```powershell
python -m tools.predict_unreviewed_pages --device cpu --force
```

Khi reviewer bấm `Lưu & Next`, API hiện tại chuyển page sang `REVIEWED` và giữ lại
`model_predicted_label`. Vì vậy có thể so sánh nhãn cuối cùng với prediction ban đầu
để tính tỷ lệ model được xác nhận hoặc bị sửa. Dataset exporter vẫn chỉ lấy các page
`REVIEWED`, nên prediction chưa được human xác nhận không đi vào dữ liệu train.
