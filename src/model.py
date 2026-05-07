from __future__ import annotations

from typing import Literal

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MLP(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 1, img_size: int = 64, dropout: float = 0.3):
        super().__init__()
        input_dim = in_channels * img_size * img_size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def _freeze_all_except(module: nn.Module, trainable_prefixes: tuple[str, ...]) -> None:
    for name, param in module.named_parameters():
        param.requires_grad = any(name.startswith(prefix) for prefix in trainable_prefixes)


def build_torchvision_model(
    backbone: Literal['resnet18', 'mobilenet_v2', 'vgg11_bn'],
    num_classes: int,
    dropout: float = 0.3,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    try:
        from torchvision.models import (
            MobileNet_V2_Weights,
            ResNet18_Weights,
            VGG11_BN_Weights,
            mobilenet_v2,
            resnet18,
            vgg11_bn,
        )
    except Exception as exc:  # pragma: no cover - environment-specific
        raise ImportError(
            'Transfer learning cần torchvision cài đúng cặp phiên bản với torch. '            'Hãy kiểm tra lại `pip install torch torchvision` hoặc dùng môi trường của môn học.'
        ) from exc

    if backbone == 'resnet18':
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        if freeze_backbone:
            _freeze_all_except(model, ('fc',))
        return model

    if backbone == 'mobilenet_v2':
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        if freeze_backbone:
            _freeze_all_except(model, ('classifier',))
        return model

    if backbone == 'vgg11_bn':
        weights = VGG11_BN_Weights.DEFAULT if pretrained else None
        model = vgg11_bn(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        if freeze_backbone:
            _freeze_all_except(model, ('classifier',))
        return model

    raise ValueError(f'Unsupported backbone: {backbone}')


def build_model(
    model_name: str,
    train_mode: str,
    num_classes: int,
    dropout: float = 0.3,
    img_size: int = 64,
) -> nn.Module:
    if model_name == 'mlp':
        return MLP(num_classes=num_classes, in_channels=1, img_size=img_size, dropout=dropout)
    if model_name == 'cnn_small':
        return SmallCNN(num_classes=num_classes, in_channels=1, dropout=dropout)

    if train_mode not in {'transfer', 'finetune'}:
        raise ValueError('Với backbone pretrained, train_mode phải là transfer hoặc finetune.')
    freeze_backbone = train_mode == 'transfer'
    return build_torchvision_model(
        backbone=model_name,
        num_classes=num_classes,
        dropout=dropout,
        pretrained=True,
        freeze_backbone=freeze_backbone,
    )
