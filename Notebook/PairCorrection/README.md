# FER2013: nhánh hiệu chỉnh theo cặp biểu cảm

[Notebook để import vào Kaggle](fer2013-pair-correction-kaggle.ipynb) chạy độc lập,
không cần tải các file Python trong repo lên Kaggle.

Kiến trúc: MobileNetV3-Small pretrained ImageNet → global classifier, kết hợp
feature map trung gian với attention theo cặp và hiệu chỉnh logits có gate.
Ba cặp được chọn từ validation của global-only pilot, cố định trước train chính.
Không dùng test để chọn cặp hoặc tính class weights.

## Chạy trên Kaggle

1. Import file `.ipynb`, gắn dataset `msambare/fer2013` có `train/` và `test/`.
2. Bật GPU và Internet để tải ImageNet weights lần đầu. Khi offline, gắn file
   state dict torchvision MobileNetV3-Small và đặt `backbone_weights_path`.
3. Giữ `EXPERIMENT = 'pair_balanced'` để thử kiến trúc kèm loss cân bằng lớp.
4. Chạy lần lượt các cell: pilot 5 epoch → chọn cặp → train chính 45 epoch.
5. Dùng validation để chọn cấu hình. Trong lúc so nhiều preset, đặt
   `RUN_FINAL_TEST=False`; bật lại sau khi chốt để xuất kết quả test.
6. Tải `pair_correction_best_checkpoint.pth` trong thư mục in ở dòng `Output:`.

Notebook mặc định một GPU cho model nhỏ. Có thể đặt `multi_gpu=True` để dùng
hai GPU khi khả dụng; batch 64 là tổng. DataParallel có chi phí và BatchNorm
có thể làm kết quả khác một GPU, nên phải ghi thiết bị/cách chạy khi so sánh.

## Cấu hình đối chứng

| Preset | Kiến trúc | Loss chính |
|---|---|---|
| `global_ce` | Chỉ global head | CE |
| `pair_ce` | Cặp từ validation, có gate | CE |
| `global_balanced` | Chỉ global head | Class-balanced CE |
| `pair_balanced` | Cặp từ validation, có gate | Class-balanced CE |
| `pair_no_gate` | Cùng nhánh cặp, gate=1 | Class-balanced CE |
| `pair_random` | Cặp ngẫu nhiên, có gate | Class-balanced CE |

Giữ split, pilot seed, train seed và ngân sách khi so preset. Mỗi run đều train
pilot; không bỏ chi phí này khỏi thời gian tổng. Mặc định global-only có 931.047
tham số, model 3 cặp có 933.805. Class-balanced loss dùng effective number, các
đầu nhị phân có loss phụ cân bằng hai phía theo số mẫu train.

`pairs_file` có thể trỏ đến `pairs.json` của run trước để giữ cặp qua nhiều seed;
notebook kiểm tra fingerprint split và thứ tự nhãn. Không resume weights hoặc
optimizer bằng tùy chọn này. `mining_seed` giữ pilot/cặp ổn định khi đổi `seed`.

## File kết quả

- `run_config.json`, `training.json`, `model_config.json`: cấu hình thực chạy.
- `split_manifest.csv`: ảnh train/validation; `class_weights.csv`: trọng số lớp.
- `pairs.json`, `pilot_pair_ranking.csv`: cặp và nguồn chọn cặp.
- `pilot_history.csv`, `main_history.csv`: tách chi phí hai giai đoạn.
- `pilot_global_best_checkpoint.pth`: best model giai đoạn pilot.
- **`pair_correction_best_checkpoint.pth`**: weights kèm kiến trúc/cặp/tiền xử lý.
- `pair_correction_best.pth`: chỉ raw state dict.
- Báo cáo validation/test, predictions, confusion matrix và pair diagnostics.

Diagnostics cho biết bao nhiêu dự đoán được hiệu chỉnh thành đúng hoặc thành sai.
So global và final logits của cùng model chỉ là kiểm tra nội bộ, chưa thay thế
run đối chứng global-only có cùng ngân sách. Nhánh vẫn được tính cho mọi ảnh;
gate điều chỉnh ảnh hưởng chứ không bỏ qua tính toán.

Chạy lại cell tạo dữ liệu tạo thư mục run mới. Các phase đã có output không bị
ghi đè khi chạy lại; tạo run mới để train lại. Checkpoint lưu best weights, không
lưu optimizer/scheduler để resume. Nạp model trong cell định nghĩa model:

```python
restored = load_trained_model(RUN_DIR / 'pair_correction_best_checkpoint.pth', DEVICE)
with torch.inference_mode():
    logits = restored(tensor)['logits']
```

Đầu vào phải là ảnh grayscale lặp 3 kênh, 112×112, chuẩn hóa ImageNet theo
notebook. Các weights này chưa dùng trực tiếp với ứng dụng CCT cũ trong `src/fer`.

## Kiểm tra local

Đã kiểm tra các công thức gate/hiệu chỉnh, gradient, loss nhị phân có MixUp,
class weights, chọn cặp/fingerprint, checkpoint, và toàn bộ notebook với dữ liệu
giả: 2 epoch pilot + 3 epoch chính trên CPU, thêm một run global-only.
Không phải kết quả accuracy FER2013; CUDA/AMP thực tế cần chạy trên Kaggle.

Để chạy lại kiểm tra từ môi trường có thư viện của dự án:

```powershell
.\.venv\Scripts\python.exe Notebook/PairCorrection/verify_notebook.py
```

Script kiểm tra dùng dữ liệu tạm, không tải weights, không sửa các notebook khác.

## Giới hạn nghiên cứu

Validation tham gia chọn cặp và checkpoint. Dataset test hiện có 7.178 ảnh,
không đồng nhất với PrivateTest 3.589 ảnh trong một số công trình. Các ý tưởng
backbone nhẹ, attention, specialists và class-balanced loss đã có tiền lệ;
notebook chưa xác nhận tính mới học thuật hoặc bảo đảm accuracy.

Nguồn thành phần: [MobileNetV3](https://arxiv.org/abs/1905.02244),
[Class-Balanced Loss](https://arxiv.org/abs/1901.05555).
