"""Tổng hợp metrics từ outputs/* và vẽ biểu đồ so sánh."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'

RUN_ORDER = [
    'mlp_baseline',
    'cnn_small_baseline',
    'cnn_small_reg',
    'resnet18_transfer',
    'resnet18_finetune',
]

# Color palette per instructions: blue / red / green tones (max 3 tones)
COLOR_VAL = '#1F62FF'
COLOR_TEST = '#FF351F'
COLOR_TIME = '#1FFF2A'

plt.rcParams.update({'font.family': 'Arial', 'font.size': 11})


def load_all() -> list[dict]:
    rows = []
    for name in RUN_ORDER:
        path = OUT / name / 'metrics.json'
        if not path.exists():
            continue
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        rows.append({
            'run': name,
            'model': data['model_name'],
            'mode': data['train_mode'],
            'val_acc': data['best_val_acc'],
            'test_acc': data['test_acc'],
            'epoch_time': data['avg_epoch_time_sec'],
            'trainable': data['trainable_params'],
        })
    return rows


def plot_accuracy_bars(rows: list[dict], output_path: Path) -> None:
    labels = [r['run'] for r in rows]
    val = [r['val_acc'] for r in rows]
    test = [r['test_acc'] for r in rows]
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, val, width, label='Best Val Acc', color=COLOR_VAL)
    bars2 = ax.bar(x + width / 2, test, width, label='Test Acc', color=COLOR_TEST)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Run')
    ax.set_ylim(0, 1.05)
    ax.set_title('NEU-CLS: Validation vs Test Accuracy by Run')
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f'{h:.3f}',
                ha='center', va='bottom', fontsize=10, color='black')

    fig.tight_layout(pad=1.5)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_time_vs_acc(rows: list[dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    times = [r['epoch_time'] for r in rows]
    accs = [r['test_acc'] for r in rows]
    sizes = [max(40, np.log10(r['trainable']) * 40) for r in rows]

    ax.scatter(times, accs, s=sizes, color=COLOR_VAL, alpha=0.85, edgecolors='white')
    for r in rows:
        ax.annotate(r['run'], (r['epoch_time'], r['test_acc']),
                    xytext=(8, 6), textcoords='offset points', fontsize=10, color=COLOR_TEST)

    ax.set_xlabel('Avg Epoch Time (sec)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('NEU-CLS: Test Accuracy vs Epoch Time (bubble size = log10(trainable params))')
    ax.set_ylim(0, 1.05)
    ax.grid(linestyle='--', alpha=0.4)
    fig.tight_layout(pad=1.5)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_trainable_params(rows: list[dict], output_path: Path) -> None:
    labels = [r['run'] for r in rows]
    params = [r['trainable'] for r in rows]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, params, color=COLOR_VAL)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('Trainable Parameters (log scale)')
    ax.set_xlabel('Run')
    ax.set_title('NEU-CLS: Trainable Parameters per Run')
    ax.grid(axis='y', which='both', linestyle='--', alpha=0.4)
    for xi, p in zip(x, params):
        ax.text(xi, p * 1.15, f'{p:,}', ha='center', va='bottom', fontsize=10, color='black')
    fig.tight_layout(pad=1.5)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    rows = load_all()
    summary_dir = OUT / '_summary'
    summary_dir.mkdir(parents=True, exist_ok=True)

    plot_accuracy_bars(rows, summary_dir / '01_accuracy_comparison.png')
    plot_time_vs_acc(rows, summary_dir / '02_time_vs_accuracy.png')
    plot_trainable_params(rows, summary_dir / '03_trainable_params.png')

    with (summary_dir / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print('Summary written to', summary_dir)
    for r in rows:
        print(f"{r['run']:24s} model={r['model']:9s} mode={r['mode']:9s} "
              f"val={r['val_acc']:.4f} test={r['test_acc']:.4f} "
              f"sec/epoch={r['epoch_time']:.2f} trainable={r['trainable']:,}")


if __name__ == '__main__':
    main()
