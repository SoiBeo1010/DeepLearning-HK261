# DeepLearning-HK261

Nhận diện cảm xúc khuôn mặt (FER2013) bằng **ResNet18 + Compact Convolutional
Transformer (CCT)**. Dự án có notebook huấn luyện, module Python dùng chung và
ứng dụng nhận diện qua webcam hoặc ảnh.

Xem [tài liệu kiến trúc model](docs/model-architecture.md) để biết luồng dữ liệu,
cấu trúc từng khối, số tham số, cấu hình huấn luyện và các giới hạn hiện tại.

## Cấu trúc

```text
DL/
├── main.py                    # CLI webcam / ảnh
├── src/fer/                   # Model, inference và pipeline huấn luyện CCT
├── notebooks/
│   ├── cct/                   # CCT + ResNet18, bản local và Kaggle
│   ├── custom-cnn/            # CNN tự thiết kế, bản Kaggle
│   └── pair-correction/       # MobileNetV3, bản Kaggle và script kiểm tra
├── docs/                      # Tài liệu kiến trúc
├── models/                    # Checkpoint cục bộ
├── outputs/                   # Kết quả và báo cáo của các run cục bộ
├── tests/                     # Kiểm thử inference và training
├── pyproject.toml
├── requirements.txt
└── requirements-notebook.txt
```

Mỗi nhóm notebook có thư mục `results/` khi có bản đã chạy lưu kèm output.
Báo cáo cục bộ trong `outputs/analysis/` được chia theo cùng ba nhóm.

| Nhóm | Notebook chạy mới |
|---|---|
| CCT + ResNet18 | [Local](notebooks/cct/fer2013-cct-resnet18-local.ipynb) · [Kaggle](notebooks/cct/fer2013-cct-resnet18-kaggle.ipynb) |
| Custom CNN | [Kaggle](notebooks/custom-cnn/fer2013-custom-cnn-kaggle.ipynb) |
| Pair Correction | [Kaggle](notebooks/pair-correction/fer2013-pair-correction-kaggle.ipynb) · [Hướng dẫn](notebooks/pair-correction/README.md) |

## Cài đặt

Đã kiểm tra với **Python 3.13**, PyTorch 2.11 và torchvision 0.26 trên Windows.
Gói Python hỗ trợ Python từ 3.10; bộ phiên bản cố định trong `requirements.txt`
dùng với Python 3.13. Với Python khác, dùng `pip install -e ".[train]"` để pip
chọn phiên bản thư viện tương thích.

```powershell
git clone https://github.com/SoiBeo1010/DeepLearning-HK261.git
cd DeepLearning-HK261
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

Nếu chỉ cần webcam/ảnh, cài `requirements.txt` thay cho
`requirements-notebook.txt`. Dùng bản PyTorch phù hợp GPU nếu cần CUDA;
`--device auto` chọn CUDA khi khả dụng, nếu không dùng CPU.

## Nhận diện

Đặt checkpoint `cct_resnet_fer2013_best.pth` trong `models/`. Checkpoint cục bộ
được sao chép từ dự án FER trước đó; **weights không nằm trong repo GitHub**.
Sau khi clone, cần cung cấp checkpoint tương thích hoặc huấn luyện lại.
Xem [models/README.md](models/README.md) để biết định dạng.

```powershell
# Webcam; nhấn Q hoặc Esc để thoát
.\.venv\Scripts\python.exe main.py

# Chọn camera và thiết bị
.\.venv\Scripts\python.exe main.py --camera 1 --device cpu --no-mirror

# Dùng checkpoint vừa huấn luyện
.\.venv\Scripts\python.exe main.py --weights outputs/training/cct_resnet_fer2013_best.pth

# Đọc ảnh, ghi kết quả; không mở cửa sổ webcam
.\.venv\Scripts\python.exe main.py --image data/example.jpg --output outputs/result.jpg --min-face-size 40
```

Sau khi cài editable có thể dùng `fer --help` hoặc `python -m fer --help`.
`main.py` tìm checkpoint mặc định theo vị trí dự án nên gọi được từ thư mục
khác; với lệnh `fer` / `python -m fer`, đường dẫn mặc định là `models/` trong
thư mục làm việc hiện tại. `--weights` luôn cho phép chọn đường dẫn cụ thể.

Thứ tự nhãn khớp cách `ImageFolder` sắp xếp thư mục:
`angry, disgust, fear, happy, neutral, sad, surprise`.
Ảnh OpenCV được đổi BGR → RGB, resize 224 × 224, grayscale 3 kênh rồi chuẩn hóa
theo ImageNet, đồng nhất với tiền xử lý validation. Giá trị phần trăm hiển thị
là xác suất softmax của mô hình.

## Huấn luyện

Dữ liệu FER2013 cần ở dạng thư mục ảnh, tên lớp viết thường:

```text
data/fer2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
├── val/       # Tùy chọn, có cùng 7 thư mục lớp
└── test/      # Cùng 7 thư mục lớp; chỉ dùng đánh giá cuối
```

Nếu không có `val/`, chương trình tách **10% train thành validation theo lớp**
với seed 42. Tập test không tham gia chọn checkpoint hay kích hoạt curriculum.
Lệnh mặc định train 50 epochs, batch 64; có thể giảm batch nếu thiếu bộ nhớ.

```powershell
.\.venv\Scripts\python.exe -m fer.training --data-dir data/fer2013 --output-dir outputs/training --epochs 50 --batch-size 64

# Mở notebook từ thư mục gốc dự án
.\.venv\Scripts\python.exe -m jupyterlab
```

Notebook CCT chạy local dùng `FER_DATA_DIR` nếu biến môi trường này được đặt,
mặc định là `data/fer2013` tính từ gốc repo. Đổi `TrainConfig` để chỉnh learning rate,
validation fraction, curriculum hoặc seed. Trên Windows/notebook giữ
`num_workers=0`; CLI có thể chỉnh `--num-workers`.

Huấn luyện khởi tạo backbone bằng ImageNet (có thể cần tải lần đầu).
`--no-pretrained` cho phép train từ đầu mà không tải weights. Pipeline dùng
Focal Loss + label smoothing, AdamW với learning rate backbone nhỏ hơn,
CosineAnnealingLR và gradient clipping. Khi validation accuracy đạt 50%,
curriculum bật MixUp và tăng Random Erasing. Train accuracy khi dùng MixUp
được tính theo trọng số hai nhãn.

Mỗi lần train tạo `history.csv`, `training.json` và checkpoint tốt nhất theo
validation accuracy trong `--output-dir`. Chọn thư mục output mới cho mỗi lần
chạy để tránh ghi đè checkpoint. Notebook nạp checkpoint tốt nhất trước khi
đánh giá trên `test/` và vẽ confusion matrix.

## Kiểm tra

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

Kiểm thử bao gồm thứ tự nhãn, chuyển màu, tính toàn vẹn của ảnh khi có nhiều
khuôn mặt, giải phóng camera khi lỗi và tách train/validation. Kiểm tra với
checkpoint thật tự bỏ qua khi clone chưa có weights. Dữ liệu, môi trường ảo,
checkpoint, log và kết quả sinh ra được bỏ qua bởi `.gitignore`.
