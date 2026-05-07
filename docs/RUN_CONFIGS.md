# Cấu hình các run của Lab 2

Tài liệu này liệt kê đầy đủ cấu hình của 5 run đã thực hiện trên dataset NEU-CLS, dùng để tái lập kết quả.

## Thông tin chung
- Người thực hiện: Lưu Thanh Tùng – KHMT 1701 – MSSV 1771040029
- Phần cứng: NVIDIA GeForce RTX 3050, CUDA 12.1
- Phần mềm: Python (môi trường `HocSau`), PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121, wandb 0.26.0
- Dataset: `data/NEU-CLS` (gộp `train/train/images` + `valid/valid/images` = 1830 ảnh, 6 lớp)
- Split chung: stratified 70 / 15 / 15, seed = 42 → 1290 train / 270 val / 270 test
- W&B project: `csc4005-lab2-neu-cnn`
- Loss: `CrossEntropyLoss`
- Scheduler: `ReduceLROnPlateau` (factor=0.5, patience=2, theo `val_loss`)
- Early stopping: `patience=8` theo `val_loss`
- Epochs cấu hình: 50 (số epoch thực tế có thể nhỏ hơn do early stopping)

## Bảng tổng hợp 5 run

| # | run_name | model_name | train_mode | optimizer | lr | weight_decay | dropout | batch_size | img_size | num_channels | normalization | augment | Epoch chạy thực tế | Best Val Acc | Test Acc | Avg sec/epoch | Trainable params |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|:---:|---:|---:|---:|---:|---:|
| 1 | `mlp_baseline` | mlp | scratch | adamw | 1e-3 | 1e-4 | 0.3 | 32 | 64 | 1 | none | ✅ | 50 | 0.5037 | 0.4778 | 2.15 | 2.164.102 |
| 2 | `cnn_small_baseline` | cnn_small | scratch | adamw | 1e-3 | 1e-4 | 0.3 | 32 | 64 | 1 | none | ✅ | 45 | 0.9630 | 0.9778 | 2.02 | 32.614 |
| 3 | `cnn_small_reg` | cnn_small | scratch | adamw | 5e-4 | 1e-3 | 0.5 | 32 | 64 | 1 | none | ✅ | 50 | 0.9778 | 0.9741 | 2.10 | 32.614 |
| 4 | `resnet18_transfer` | resnet18 | transfer | adamw | 1e-3 | 1e-4 | 0.3 | 32 | 128 | 3 | imagenet | ✅ | 35 | 0.9815 | 0.9667 | 3.71 | 3.078 |
| 5 | `resnet18_finetune` | resnet18 | finetune | adamw | 1e-4 | 1e-4 | 0.3 | 16 | 128 | 3 | imagenet | ✅ | 19 | **1.0000** | **1.0000** | 4.73 | 11.179.590 |

## Lệnh tái lập từng run

> Tất cả lệnh chạy trong môi trường `conda activate HocSau`, từ thư mục gốc repo. Mỗi run sinh thư mục `outputs/<run_name>/` chứa `best_model.pt`, `history.csv`, `curves.png`, `confusion_matrix.png`, `metrics.json`.

### Run 1 – MLP baseline (đối chứng từ Lab 1)
```powershell
python -m src.train --data_dir data/NEU-CLS `
  --project csc4005-lab2-neu-cnn --run_name mlp_baseline `
  --model_name mlp --train_mode scratch `
  --optimizer adamw --lr 0.001 --weight_decay 0.0001 --dropout 0.3 `
  --epochs 50 --batch_size 32 --img_size 64 --patience 8 --augment --use_wandb
```

### Run 2 – CNN scratch baseline
```powershell
python -m src.train --data_dir data/NEU-CLS `
  --project csc4005-lab2-neu-cnn --run_name cnn_small_baseline `
  --model_name cnn_small --train_mode scratch `
  --optimizer adamw --lr 0.001 --weight_decay 0.0001 --dropout 0.3 `
  --epochs 50 --batch_size 32 --img_size 64 --patience 8 --augment --use_wandb
```

### Run 3 – CNN scratch (regularize mạnh hơn)
```powershell
python -m src.train --data_dir data/NEU-CLS `
  --project csc4005-lab2-neu-cnn --run_name cnn_small_reg `
  --model_name cnn_small --train_mode scratch `
  --optimizer adamw --lr 0.0005 --weight_decay 0.001 --dropout 0.5 `
  --epochs 50 --batch_size 32 --img_size 64 --patience 8 --augment --use_wandb
```

### Run 4 – ResNet18 transfer (freeze backbone)
```powershell
python -m src.train --data_dir data/NEU-CLS `
  --project csc4005-lab2-neu-cnn --run_name resnet18_transfer `
  --model_name resnet18 --train_mode transfer `
  --optimizer adamw --lr 0.001 --weight_decay 0.0001 --dropout 0.3 `
  --epochs 50 --batch_size 32 --img_size 128 --patience 8 --augment --use_wandb
```

### Run 5 – ResNet18 finetune (mở băng toàn bộ)
```powershell
python -m src.train --data_dir data/NEU-CLS `
  --project csc4005-lab2-neu-cnn --run_name resnet18_finetune `
  --model_name resnet18 --train_mode finetune `
  --optimizer adamw --lr 0.0001 --weight_decay 0.0001 --dropout 0.3 `
  --epochs 50 --batch_size 16 --img_size 128 --patience 8 --augment --use_wandb
```

## Sinh biểu đồ tổng hợp
```powershell
python -m ci.make_summary           # Sinh 01_, 02_, 03_ tại outputs/_summary/
python -m ci.render_misclassified   # Sinh 04_misclassified_samples.png
```

## Logic ràng buộc cấu hình ([src/train.py](src/train.py) `validate_args`)
- `mlp` và `cnn_small` chỉ chấp nhận `--train_mode scratch`.
- `resnet18`, `mobilenet_v2`, `vgg11_bn` chỉ chấp nhận `--train_mode transfer` hoặc `--train_mode finetune`.
- `transfer` → freeze toàn bộ backbone, chỉ học classifier head.
- `finetune` → mở băng toàn bộ tham số.

## Ghi chú
- Số epoch chạy thực tế thấp hơn 50 ở 3 run là do early stopping kích hoạt khi `val_loss` không cải thiện trong 8 epoch liên tiếp. Đây là hành vi mong đợi.
- Toàn bộ 5 run đã được đồng bộ lên W&B project `csc4005-lab2-neu-cnn` để so sánh trực quan.
