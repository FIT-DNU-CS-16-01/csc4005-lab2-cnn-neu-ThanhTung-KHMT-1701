# CSC4005 – Lab 2 Report

## 1. Thông tin chung
- Học phần: CSC4005 – Học sâu
- Lab: Lab 2 – CNN Image Classification (From Scratch vs Transfer)
- Repo: `csc4005-lab2-cnn-neu-ThanhTung-KHMT-1701`
- W&B project: [`csc4005-lab2-neu-cnn`](https://wandb.ai/thanhtung-contact-official-/csc4005-lab2-neu-cnn)
- Phần cứng: NVIDIA GeForce RTX 3050, CUDA 12.1, PyTorch 2.5.1
- Dataset: NEU Surface Defect (1830 ảnh grayscale, 6 lớp – gộp cả `train/` và `valid/`)
- Split: stratified 70 / 15 / 15 (seed 42), tổng 1290 train / 270 val / 270 test
- Số epoch thống nhất: 50 (early stopping patience = 8)

## 2. Bài toán
Phân loại 6 loại lỗi bề mặt thép (Crazing, Inclusion, Patches, Pitted_Surface, Rolled-in_Scale, Scratches) trên bộ NEU-CLS. Dữ liệu là ảnh grayscale 200×200 đã chuẩn hóa, được resize về 64×64 (scratch) hoặc 128×128 (transfer/finetune).

## 3. Mô hình và cấu hình

### 3.1. MLP baseline (chạy lại trên repo Lab 2)
- Kiến trúc: `Flatten(64×64) → FC(4096→512) → ReLU → Dropout(0.3) → FC(512→128) → ReLU → Dropout(0.3) → FC(128→6)`
- Optimizer AdamW, `lr=1e-3`, `wd=1e-4`, augment bật.

### 3.2. CNN from scratch (`cnn_small`)
- 3 khối `Conv3×3 → BN → ReLU → MaxPool2`, kênh `1→16→32→64`, `AdaptiveAvgPool(1×1)`, `FC(64→128→6)`.
- 2 cấu hình:
  - `cnn_small_baseline`: `lr=1e-3, wd=1e-4, dropout=0.3`
  - `cnn_small_reg`: `lr=5e-4, wd=1e-3, dropout=0.5` (regularize mạnh hơn)

### 3.3. Transfer learning – ResNet18 (ImageNet)
- Đầu vào 3 kênh + chuẩn hóa ImageNet, `img_size=128`.
- `resnet18_transfer`: đóng băng toàn bộ backbone, chỉ học `fc` (3.078 params).
- `resnet18_finetune`: mở băng toàn bộ, `lr=1e-4`, `batch_size=16`.

## 4. Bảng kết quả

| Run | Model | Train mode | Best Val Acc | Test Acc | Avg sec/epoch | Trainable Params | Nhận xét |
|---|---|---|---:|---:|---:|---:|---|
| `mlp_baseline` | MLP | scratch | 0.5037 | 0.4778 | 2.15 | 2.164.102 | Mất thông tin không gian, nhiều tham số nhưng kém |
| `cnn_small_baseline` | CNN-small | scratch | 0.9630 | 0.9778 | 2.02 | 32.614 | Nhảy vọt so với MLP, params ít hơn ~66× |
| `cnn_small_reg` | CNN-small | scratch | 0.9778 | 0.9741 | 2.10 | 32.614 | Regularize mạnh giảm overfitting nhẹ |
| `resnet18_transfer` | ResNet18 | transfer | 0.9815 | 0.9667 | 3.71 | 3.078 | Chỉ train head, hội tụ rất nhanh |
| `resnet18_finetune` | ResNet18 | finetune | **1.0000** | **1.0000** | 4.73 | 11.179.590 | Tốt nhất, gần như giải quyết hoàn toàn bài toán |

Ảnh tổng hợp: [outputs/_summary/01_accuracy_comparison.png](outputs/_summary/01_accuracy_comparison.png), [outputs/_summary/02_time_vs_accuracy.png](outputs/_summary/02_time_vs_accuracy.png), [outputs/_summary/03_trainable_params.png](outputs/_summary/03_trainable_params.png).

## 5. Phân tích learning curves
- **MLP** (xem [outputs/mlp_baseline/curves.png](outputs/mlp_baseline/curves.png)): loss giảm chậm, val_acc kẹt quanh 50% và dao động — đặc trưng cho mô hình thiếu inductive bias không gian. Khoảng cách train/val nhỏ → underfitting hơn là overfitting.
- **CNN baseline** ([outputs/cnn_small_baseline/curves.png](outputs/cnn_small_baseline/curves.png)): hội tụ nhanh trong ~10 epoch đầu, val_acc vượt 90% rất sớm. Có dấu hiệu overfitting nhẹ về cuối (train_acc cao hơn val_acc).
- **CNN reg** ([outputs/cnn_small_reg/curves.png](outputs/cnn_small_reg/curves.png)): khoảng cách train/val co lại; val_acc tốt hơn baseline 1.5 điểm phần trăm.
- **ResNet18 transfer** ([outputs/resnet18_transfer/curves.png](outputs/resnet18_transfer/curves.png)): val_acc đạt >94% ngay epoch 2 nhờ feature ImageNet; sau đó cải thiện chậm vì chỉ học 3.078 tham số.
- **ResNet18 finetune** ([outputs/resnet18_finetune/curves.png](outputs/resnet18_finetune/curves.png)): val_acc đạt 100% từ epoch 2; loss tiến về 0; early stop ở epoch 19.

## 6. Confusion matrix và lỗi dự đoán sai

### Mô hình tốt nhất: `resnet18_finetune`
Confusion matrix: [outputs/resnet18_finetune/confusion_matrix.png](outputs/resnet18_finetune/confusion_matrix.png) — đường chéo hoàn hảo, **không có ảnh test bị dự đoán sai** (270/270 đúng).

### Phân tích lỗi từ mô hình thứ 2: `resnet18_transfer` (test_acc = 96.67%, 9 mẫu sai trên 270)
Confusion matrix: [outputs/resnet18_transfer/confusion_matrix.png](outputs/resnet18_transfer/confusion_matrix.png).

| Nhãn thật | Bị dự đoán nhầm thành | Số mẫu | Diễn giải |
|---|---|---:|---|
| Inclusion | Pitted_Surface | 5 | Hai loại đều có các đốm tối nhỏ rải rác, dễ nhầm khi resolution chỉ 128 |
| Crazing | Patches | 1 | Vết nứt nhỏ trên nền đồng đều có thể trông như vùng patch mờ |
| Crazing | Rolled-in_Scale | 1 | Cả hai có pattern dạng vân ngang kéo dài |
| Inclusion | Scratches | 1 | Inclusion dài, mảnh có thể giống Scratches |
| Scratches | Inclusion | 1 | Scratches ngắn nhỏ giống Inclusion |

→ Cặp dễ nhầm nhất là **Inclusion ↔ Pitted_Surface**. Đây cũng là nhóm lỗi “hạt rời rạc” khó tách bằng đặc trưng cấp thấp.

### So sánh với `cnn_small_baseline` (test_acc = 97.78%, 6 mẫu sai)
| Nhãn thật | Bị dự đoán nhầm thành | Số mẫu |
|---|---|---:|
| Inclusion | Pitted_Surface | 3 |
| Pitted_Surface | Inclusion | 1 |
| Scratches | Inclusion | 2 |

→ Cùng quy luật: cả CNN scratch và ResNet18 transfer đều gặp khó với cặp Inclusion ↔ Pitted_Surface; chỉ khi mở băng toàn bộ ResNet18 (finetune) mới giải quyết triệt để.

## 7. Kết luận

**(1) CNN có cải thiện so với MLP không?** Có, rất rõ rệt. Test accuracy nhảy từ **47.8% (MLP) → 97.8% (CNN scratch)** với số tham số trainable nhỏ hơn ~66 lần (32.614 vs 2.164.102). Điều này khẳng định CNN khai thác local pattern + weight sharing tốt hơn nhiều so với việc flatten ảnh thành vector.

**(2) Transfer learning có luôn tốt hơn CNN from scratch không?**
- *Khi chỉ freeze head* (`resnet18_transfer`): val_acc cao hơn (98.15% vs 97.78%) nhưng test_acc thấp hơn nhẹ (96.67% vs 97.78%). Trên dataset NEU-CLS (grayscale, texture công nghiệp khác xa ImageNet), feature đóng băng không vượt được CNN nhỏ chuyên biệt.
- *Khi finetune toàn bộ* (`resnet18_finetune`): đạt **100% / 100%** — vượt trội tất cả. Pretrained weights cho điểm khởi đầu rất tốt, finetune tinh chỉnh để bắt đặc trưng riêng của NEU.

**(3) Khi nào nên chọn transfer learning thay vì train from scratch?**
- Khi dataset nhỏ (NEU-CLS chỉ ~1800 ảnh) và miền dữ liệu có thể tận dụng feature tổng quát: nên finetune để đạt accuracy tối đa.
- Khi tài nguyên hạn chế / cần model nhẹ: CNN-small from scratch (32K params) đã đủ tốt (97.78%) và rẻ hơn cả về thời gian (2 sec/epoch) lẫn dung lượng so với ResNet18 (3.71–4.73 sec/epoch, 11M params).
- Khi miền dữ liệu lệch mạnh khỏi ImageNet và chỉ freeze backbone: hiệu quả không vượt được CNN scratch nhỏ — nên ít nhất finetune block cuối.

**(4) Cặp lỗi khó nhất**: Inclusion ↔ Pitted_Surface là điểm yếu chung của các mô hình scratch và transfer-freeze; chỉ finetune toàn bộ mới phân biệt được. Nếu sau này cần triển khai mô hình nhẹ, có thể tăng độ phân giải đầu vào (96–128) hoặc thêm augmentation đặc trưng cho cặp này.

## 8. Liên kết W&B
- Project: <https://wandb.ai/thanhtung-contact-official-/csc4005-lab2-neu-cnn>
- Các run đã sync: `mlp_baseline`, `cnn_small_baseline`, `cnn_small_reg`, `resnet18_transfer`, `resnet18_finetune`.
