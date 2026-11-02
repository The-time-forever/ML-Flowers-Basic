"""A compact from-scratch CNN baseline for flower classification.

Architecture: 4 convolutional blocks (2x conv-BN-ReLU + max-pool each),
global average pooling, dropout and a linear classifier. It is deliberately
small (~1.2M parameters) so it trains quickly on a single GPU and serves as a
reference point for pretrained models later.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_convs: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        ch = in_channels
        for _ in range(num_convs):
            layers += [
                nn.Conv2d(ch, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            ch = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),     # 224 -> 112
            ConvBlock(32, 64),    # 112 -> 56
            ConvBlock(64, 128),   # 56  -> 28
            ConvBlock(128, 256),  # 28  -> 14
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SimpleCNN()
    print(model)
    print(f"parameters: {count_parameters(model):,}")
    print(f"output shape: {model(torch.zeros(2, 3, 224, 224)).shape}")
