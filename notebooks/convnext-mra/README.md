# FER2013 — ConvNeXt-Tiny Multi-Region Attention (MRA) & Logit Adjustment

[Notebook chạy độc lập trên Kaggle](fer2013-convnext-mra-kaggle.ipynb) được thiết kế để thay thế
kiến trúc MobileNetV3-Small + PairCorrection cũ và giải quyết trực tiếp các nút thắt về
overfitting, data leakage, mất cân bằng lớp và tốc độ pipeline dữ liệu.

## Điểm mới so với nhánh PairCorrection

1. **Kiến trúc `FERMultiRegionNet`:**
   - Backbone mặc định **ConvNeXt-Tiny** pretrained ImageNet (LayerNorm, kernel depthwise 7×7,
     Stochastic Depth `0.20`), không cần `timm` hay thư viện ngoài.
   - **Multi-Region Spatial Attention (MRA)** trích đặc trưng ở **Stage 3** (384 kênh, 1/16 độ phân giải —
     14×14 ở ảnh 224 hoặc 8×8 ở ảnh 128) với 4 đầu attention có **Diversity Loss** giảm trùng vùng,
     kết hợp với vector toàn cục từ **Stage 4** (768 kênh).
   - Loại bỏ pha pilot chọn cặp trên validation và loại bỏ xung đột giữa residual logit vs BCE độc lập.
2. **Chống overfitting & ổn định hóa:**
   - Augmentation chuyên cho ảnh khuôn mặt (`RandomHorizontalFlip`, `RandomAffine`, `ColorJitter`,
     `RandomErasing 0.25`).
   - **MixUp (`alpha=0.4`) + CutMix (`alpha=1.0`)** ở xác suất `0.50` (thay cho `Beta(0.1, 0.1)`).
   - **Layer-wise LR Decay (LLRD `0.80`)**, **Model EMA (`0.998`)**, 1 pha Cosine liên tục (35 epoch).
3. **Chống leakage & chuẩn hóa đánh giá:**
   - Lọc trùng tuyệt đối theo pixel MD5 (loại bỏ cặp trùng nhãn và nhóm trùng khác nhãn trong `train/`)
     và chia tập bằng **StratifiedGroupKFold theo perceptual dHash 64-bit**.
   - Đo song song **Accuracy, Macro F1, Balanced Accuracy, unsmoothed Val NLL và ECE (15 bins)**;
     chọn checkpoint theo `composite = 0.5 * Acc + 0.5 * Macro F1` (có thể đổi sang `macro_f1` hoặc `accuracy`).
   - Hỗ trợ **Train-time Logit Adjustment**, **Post-hoc Logit Adjustment Pareto sweep**, **Temperature Scaling**
     và **Horizontal-Flip TTA** hoàn toàn trên validation; **`RUN_FINAL_TEST = False` mặc định**.
4. **Tối ưu tốc độ dữ liệu:**
   - Cache ảnh **1 kênh grayscale `uint8`** (~63 MiB cho 28.709 ảnh thay vì 189 MiB RGB).
   - Chạy `RandomAffine` và `ColorJitter` trên ảnh gốc 48×48 **trước khi `Resize`**.
   - `num_workers='auto'` tự dùng 2 worker trên Linux/Kaggle và 0 trên Windows.

## Các preset đối chứng

| Preset | Backbone | Ảnh | MRA | Train Logit Adj ($\tau$) | Mục đích |
|---|---|---:|---|---:|---|
| **`convnext_tiny_224_mra` (mặc định)** | ConvNeXt-Tiny | 224 | Bật | 0.0 | Cấu hình đề xuất chính |
| `convnext_tiny_224_global` | ConvNeXt-Tiny | 224 | Tắt | 0.0 | Ablation đo tác dụng thực của MRA |
| `convnext_tiny_224_mra_la` | ConvNeXt-Tiny | 224 | Bật | 0.4 | Tái cân bằng lớp yếu lúc train |
| `convnext_tiny_128_mra` | ConvNeXt-Tiny | 128 | Bật | 0.0 | Đối chứng tốc độ vs độ phân giải |
| `resnet34_128_mra` | ResNet-34 | 128 | Bật | 0.0 | Đối chứng backbone CNN chuẩn |
| `resnet18_112_mra` | ResNet-18 | 112 | Bật | 0.0 | Đối chứng nhẹ/nhanh |

## Cách chạy trên Kaggle

1. Import `fer2013-convnext-mra-kaggle.ipynb`, gắn dataset `msambare/fer2013`.
2. Bật **GPU T4** và **Internet** (để `torchvision` tải ImageNet weights lần đầu).
3. Chạy đối chứng `convnext_tiny_224_global` và `convnext_tiny_224_mra` với `RUN_FINAL_TEST = False`.
4. Kiểm tra bảng Validation (`Val Acc`, `Val Macro F1`, `Val NLL/ECE`, `val_posthoc_logit_adj.csv`) trên 3 seed
   trước khi bật `RUN_FINAL_TEST = True` đúng 1 lần cho cấu hình thắng cuối cùng.

## Kiểm tra cục bộ (không cần dataset thật / không tải weights)

```powershell
.\.venv\Scripts\python.exe notebooks/convnext-mra/verify_notebook.py
```

Script kiểm thử xác minh cấu trúc notebook, bộ lọc trùng MD5/dHash, cache 1 kênh, kiến trúc ConvNeXt-Tiny /
ResNet-34 / ResNet-18, Diversity Loss, LLRD, EMA, Logit Adjustment, Temperature Scaling, checkpoint round-trip
và chạy trọn vẹn toàn bộ notebook trên dữ liệu tổng hợp bằng CPU, đồng thời xác nhận không chỉnh sửa các
notebook khác trong dự án.

> **Lưu ý:** Chưa có kết quả thực nghiệm FER2013 cho các preset trong notebook này; cần chạy trên Kaggle GPU
> và đối chứng trên cùng split với các baseline trước khi kết luận mức cải thiện thực tế.
