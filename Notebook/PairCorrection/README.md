# FER2013: nhánh hiệu chỉnh theo cặp biểu cảm

[Notebook để import vào Kaggle](fer2013-pair-correction-kaggle.ipynb) chạy độc lập,
không cần tải các file Python trong repo lên Kaggle.

Kiến trúc: MobileNetV3-Small pretrained ImageNet → global classifier, kết hợp
feature map trung gian với attention theo cặp và hiệu chỉnh logits có gate.
Ba cặp được chọn từ validation của global-only pilot, cố định trước train chính.
Không dùng test để chọn cặp hoặc tính class weights. **Mặc định mới là global-only
Cross-Entropy (`global_ce`)**, để xác lập chất lượng backbone trước khi thêm nhánh.

## Thay đổi sau review run 60,57%

- Backbone LR mặc định **1e-4** thay vì 3e-5; có `global_ce_lr3e5` để so riêng LR.
- Pilot **15 epoch**, main **35 epoch**, tổng ngân sách vẫn 50 như các run trước.
- Xếp cặp theo **trung bình 3 confusion cuối pilot**, thay vì một best pilot yếu
  sau 5 epoch. Lưu score std/min/max, số lần vào top-k và Jaccard thứ hạng.
- Báo cảnh báo nếu pilot cuối chưa đạt `pair_mining_min_accuracy=0.60` hoặc cặp
  thiếu ổn định; vẫn chạy theo ngân sách cố định, không bảo đảm accuracy tốt.
- Diagnostics gate/delta theo true class, chỉ dùng từng cặp và bỏ từng cặp.
- Test mặc định **tắt**; thêm hỗ trợ ảnh 224 và lưu resolution vào checkpoint.

Đây là các thử nghiệm để kiểm tra giả thuyết backbone chưa thích nghi và chọn
cặp quá sớm. Chưa có kết quả FER2013 mới, không khẳng định sẽ tăng accuracy.

## Chạy trên Kaggle

1. Import file `.ipynb`, gắn dataset `msambare/fer2013` có `train/` và `test/`.
2. Bật GPU và Internet để tải ImageNet weights lần đầu. Khi offline, gắn file
   state dict torchvision MobileNetV3-Small và đặt `backbone_weights_path`.
3. Giữ **`EXPERIMENT = 'global_ce'`** cho lần thử tiếp theo.
4. Chạy lần lượt: pilot 15 epoch → báo cáo xếp cặp → train chính 35 epoch.
   Global-only không tạo nhánh cặp; bảng xếp hạng được lưu để tham khảo.
5. Dùng validation để chọn cấu hình. `RUN_FINAL_TEST=False` là mặc định;
   chỉ bật True sau khi chốt để xuất kết quả test.
6. Tải `pair_correction_best_checkpoint.pth` trong thư mục in ở dòng `Output:`.

Notebook mặc định một GPU cho model nhỏ. Có thể đặt `multi_gpu=True` để dùng
hai GPU khi khả dụng; batch 64 là tổng. DataParallel có chi phí và BatchNorm
có thể làm kết quả khác một GPU, nên phải ghi thiết bị/cách chạy khi so sánh.

## Cấu hình đối chứng

| Preset | Kiến trúc | Loss chính |
|---|---|---|
| `global_ce` (mặc định) | Chỉ global head, backbone LR 1e-4 | CE |
| `global_ce_lr3e5` | Chỉ global head, backbone LR 3e-5 | CE |
| `pair_ce` | Cặp từ validation, có gate | CE |
| `global_balanced` | Chỉ global head | Class-balanced CE |
| `pair_balanced` | Cặp từ validation, có gate | Class-balanced CE |
| `pair_no_gate` | Cùng nhánh cặp, gate=1 | Class-balanced CE |
| `pair_random` | Cặp ngẫu nhiên, có gate | Class-balanced CE |

Giữ split, pilot seed, train seed và ngân sách khi so preset. Hai preset LR dùng
cùng lịch 15+35; không dùng run cũ 5+45 để quy kết riêng tác dụng của LR.
Sau baseline, so `pair_ce` với `global_ce`, và `pair_balanced` với
`global_balanced` để tách tác dụng nhánh. `pair_no_gate` so với `pair_balanced`
giúp kiểm tra gate; `pair_random` kiểm tra cách chọn cặp.

Mỗi run đều train pilot; không bỏ chi phí này khỏi thời gian tổng. Global-only
có 931.047 tham số, model 3 cặp có 933.805. Class-balanced loss dùng effective
number, các đầu nhị phân có loss phụ cân bằng hai phía theo số mẫu train.

`pairs_file` có thể trỏ đến `pairs.json` của run trước để giữ cặp qua nhiều seed;
notebook kiểm tra fingerprint split và thứ tự nhãn. Không resume weights hoặc
optimizer bằng tùy chọn này. `mining_seed` giữ pilot/cặp ổn định khi đổi `seed`.

`pair_mining_window=3` lấy ba epoch cuối theo thứ tự, chuẩn hóa confusion theo
support từng lớp, trung bình score rồi xếp cặp. Best pilot checkpoint dùng để
khởi tạo main có thể thuộc một epoch khác; hai lựa chọn này được ghi riêng.
Thứ hạng ổn định không đồng nghĩa đã tìm đúng các cặp khó nhất sau main.

Chọn `CFG.update(image_size=224)` sau khi áp dụng preset nếu muốn kiểm tra riêng
độ phân giải; mặc định 112. Resize không sinh chi tiết mới. So accuracy và thời
gian với cùng các yếu tố còn lại, không đồng thời thêm nhánh/loss.

## Tốc độ đọc/decode ảnh và tái dùng cache

Lần đầu vẫn phải mở/decode ảnh nguồn. Mặc định `cache_workers=8` dùng hàng đợi
liên tục, tối đa 16 công việc pending, thay vì chờ hết từng nhóm file. Các luồng
này chỉ đọc ảnh RGB gốc, không thực hiện augmentation hoặc tạo tiến trình con.
Có thể thử 4/8/16 luồng theo tốc độ thực tế của mount; nhiều luồng hơn không
đảm bảo tăng tốc tương ứng. Thanh tiến độ là thời gian cả đọc file và decode,
không chỉ CPU decode. GPU không thực hiện bước này.

- **Chạy lại cell trong cùng kernel:** `cache_reuse=True` giữ cache trong RAM;
  log in `Reusing RAM cache`, không decode ảnh nguồn lại.
- **RAM đã mất nhưng file còn:** `cache_disk=True` lưu pack `.npz` không nén
  dưới `/kaggle/working/.fer_image_cache`, log in `Cache loaded from disk` khi
  có thể dùng lại. Một file pack giúp tránh mở lại hàng chục nghìn file nhỏ.
- **Kaggle xóa session/working storage:** cả RAM và file cache có thể mất, lúc
  đó vẫn phải tạo lại. Cache pack không phải weights hay checkpoint.

Cache lưu RGB uint8 nguyên kích thước, giữ nguyên pixel. Loader trả PIL mới cho
transform mỗi lần, nên augmentation không bị đóng băng và không làm hỏng cache.
`cache_max_mib=256` giới hạn pixel đã lưu; ảnh còn lại đọc từ nguồn. Bộ nhớ tạm
khi pack/nạp và overhead object không nằm trong giới hạn pixel này.

Khóa cache gồm toàn bộ danh sách đường dẫn, budget, format và `cache_revision`.
Không stat/hash nội dung từng ảnh khi reuse để tránh lại bị chậm trên mount.
Nếu chỉnh nội dung ảnh nhưng giữ tên, tăng `cache_revision` hoặc đặt
`cache_reuse=False` để buộc tạo mới. Đặt `cache_disk=False` nếu chỉ muốn cache
RAM; đặt `cache_images=False` để tắt cache hoàn toàn (train lại phải đọc nguồn).

NPZ được nạp với `allow_pickle=False`, kiểm tra index/shape/pixel budget và ghi
atomic qua file tạm. Pack lỗi được dựng lại từ ảnh nguồn. Các cải tiến không đổi
split, model hoặc input sau transform; chưa đo mức tăng tốc trên Kaggle.

## File kết quả

- `run_config.json`, `training.json`, `model_config.json`: cấu hình thực chạy.
- `split_manifest.csv`: ảnh train/validation; `class_weights.csv`: trọng số lớp.
- `pairs.json`, `pilot_pair_ranking.csv`: cặp và nguồn chọn cặp.
- `pilot_validation_snapshots.json`: confusion/metric của từng pilot epoch.
- `pilot_mining_confusion_mean.csv`: confusion trung bình cửa sổ, có thể là số lẻ.
- `pilot_pair_stability.csv`: overlap/Jaccard của từng epoch với top-k tổng hợp.
- `pilot_history.csv`, `main_history.csv`: tách chi phí hai giai đoạn.
- `pilot_global_best_checkpoint.pth`: best model giai đoạn pilot.
- **`pair_correction_best_checkpoint.pth`**: weights kèm kiến trúc/cặp/tiền xử lý.
- `pair_correction_best.pth`: chỉ raw state dict.
- Báo cáo validation/test, predictions có gate/delta từng ảnh, confusion matrix,
  `*_pair_diagnostics.csv`, `*_pair_by_class.csv`.

Diagnostics cho biết bao nhiêu dự đoán được hiệu chỉnh thành đúng hoặc thành sai.
Có mean gate/delta trên toàn tập, chỉ các ảnh thuộc cặp, và theo từng nhãn thật.
`only_pair_*` chỉ thêm cặp đó vào global logits. `marginal_*` so model cuối với
model bỏ riêng cặp đó. Hai phép kiểm tra giữ feature/model weights cố định,
không train lại; số đóng góp các cặp không cộng được như các phần độc lập.
So global và final logits của cùng model chỉ là kiểm tra nội bộ, chưa thay thế
run đối chứng global-only có cùng ngân sách. Nhánh vẫn được tính cho mọi ảnh;
gate điều chỉnh ảnh hưởng chứ không bỏ qua tính toán.
Global-only có final=global và các bảng pair diagnostics trống có header.

Chạy lại cell tạo dữ liệu tạo thư mục run mới. Các phase đã có output không bị
ghi đè khi chạy lại; tạo run mới để train lại. Checkpoint lưu best weights, không
lưu optimizer/scheduler để resume. Nạp model trong cell định nghĩa model:

```python
restored = load_trained_model(RUN_DIR / 'pair_correction_best_checkpoint.pth', DEVICE)
with torch.inference_mode():
    logits = restored(tensor)['logits']
```

Đầu vào phải là ảnh grayscale lặp 3 kênh, kích thước khớp checkpoint (mặc định
112×112, có thể thử 224×224), chuẩn hóa ImageNet. Loader vẫn đọc checkpoint cũ
không có `image_size` và mặc định chúng là 112. Các weights này chưa dùng trực
tiếp với ứng dụng CCT cũ trong `src/fer`.

## Kiểm tra local

Đã kiểm tra các công thức gate/hiệu chỉnh, gradient, loss nhị phân có MixUp,
class weights, chọn cặp/fingerprint, checkpoint và cả hai độ phân giải 112/224.
Kiểm tra ranking theo cửa sổ qua nhiều epoch, hỗ trợ snapshot cùng validation,
gate/delta trên fixture có helpful/harmful biết trước, và toàn bộ notebook với
dữ liệu giả: 2 epoch pilot + 3 epoch chính trên CPU; thêm run global-only theo
mặc định mới với test bị giữ ngoài filesystem. Test không được đọc khi bị tắt.
Kiểm tra cache gồm hàng đợi không bị chặn bởi file chậm, pixel/augmentation
giống nguồn, bản PIL độc lập, pack disk/RAM reuse không decode lại nguồn,
cache hỏng được dựng lại, budget/fallback và đóng thread khi lỗi/dừng sớm.
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
