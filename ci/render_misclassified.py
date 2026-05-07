"""Render thumbnail các mẫu test bị dự đoán sai cho mô hình resnet18_transfer."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.dataset import create_dataloaders
from src.model import build_model
from src.utils import set_seed

ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = 'resnet18_transfer'
RUN_DIR = ROOT / 'outputs' / RUN_NAME
OUTPUT_PATH = ROOT / 'outputs' / '_summary' / '04_misclassified_samples.png'

plt.rcParams.update({'font.family': 'Arial', 'font.size': 11})


def main() -> None:
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = create_dataloaders(
        data_dir=str(ROOT / 'data' / 'NEU-CLS'),
        img_size=128,
        batch_size=32,
        val_size=0.15,
        test_size=0.15,
        random_state=42,
        augment=False,
        num_channels=3,
        normalization='imagenet',
    )

    model = build_model('resnet18', 'transfer', len(data.class_names), dropout=0.3, img_size=128).to(device)
    state = torch.load(RUN_DIR / 'best_model.pt', map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    misclassified: list[tuple[Path, int, int]] = []
    test_samples = data.test_loader.dataset.samples
    idx = 0
    with torch.no_grad():
        for x, y in data.test_loader:
            x = x.to(device)
            preds = torch.argmax(model(x), dim=1).cpu().tolist()
            y_list = y.tolist()
            for pred, true in zip(preds, y_list):
                if pred != true:
                    path, _ = test_samples[idx]
                    misclassified.append((path, true, pred))
                idx += 1

    print(f'Found {len(misclassified)} misclassified samples')
    show = misclassified[:5]
    n = len(show)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6))
    if n == 1:
        axes = [axes]
    for ax, (path, true_idx, pred_idx) in zip(axes, show):
        img = Image.open(path).convert('L')
        ax.imshow(np.asarray(img), cmap='gray')
        ax.set_title(
            f'True: {data.class_names[true_idx]}\nPred: {data.class_names[pred_idx]}',
            color='#FF351F', fontsize=11,
        )
        ax.axis('off')

    fig.suptitle(f'Misclassified Test Samples ({RUN_NAME})', fontsize=12)
    fig.tight_layout(pad=1.5)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved', OUTPUT_PATH)


if __name__ == '__main__':
    main()
