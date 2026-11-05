"""Compact from-scratch CNNs for flower classification.

Two variants share the same channel schedule (32 -> 64 -> 128 -> 256) and
downsampling rhythm (224 -> 112 -> 56 -> 28 -> 14):

- ``SimpleCNN``: 4 plain blocks, each ``Conv-BN-ReLU-Conv-BN-ReLU-MaxPool``.
- ``ResidualCNN``: the same 4 blocks with a shortcut connection; a 1x1
  conv + BN projects the identity when channels change.
- ``residual_deep``: adds one extra residual block to each of the last two
  stages, separating the effect of depth from the effect of shortcuts.
- ``residual_se``: same depth as ``residual``, with squeeze-and-excitation
  channel attention (reduction 16) in the last two stages.
- ``mobilenet_v3_small`` / ``efficientnet_v2_m`` / ``convnext_base``: cached
  ImageNet-pretrained backbones with a new 5-class head (transfer learning).

The residual version adds only ~44K parameters and keeps the depth identical,
which isolates the effect of the shortcut connection.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models as tv_models

PRETRAINED_ARCHS = ("mobilenet_v3_small", "efficientnet_v2_m", "convnext_base")


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


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (reduction=16 by default)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class ResidualBlock(nn.Module):
    """Basic residual block: conv-BN-ReLU-conv-BN-(SE)-shortcut."""

    def __init__(self, in_channels: int, out_channels: int,
                 se: bool = False, se_reduction: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = (SEBlock(out_channels, reduction=se_reduction)
                   if se else nn.Identity())
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.relu(out + identity)
        return self.pool(out)


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


class ResidualCNN(nn.Module):
    def __init__(self, num_classes: int = 5, dropout: float = 0.3,
                 extra_blocks: bool = False, se_last_two: bool = False):
        super().__init__()
        blocks = [
            ResidualBlock(3, 32),                            # 224 -> 112
            ResidualBlock(32, 64),                           # 112 -> 56
            ResidualBlock(64, 128, se=se_last_two),          # 56  -> 28
        ]
        if extra_blocks:
            blocks.append(ResidualBlock(128, 128))  # extra block, stage 3
        blocks.append(ResidualBlock(128, 256, se=se_last_two))  # 28 -> 14
        if extra_blocks:
            blocks.append(ResidualBlock(256, 256))  # extra block, stage 4
        self.features = nn.Sequential(*blocks)
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


def _build_pretrained(arch: str, num_classes: int) -> nn.Module:
    """Load cached ImageNet weights and replace the classifier head."""
    if arch == "mobilenet_v3_small":
        model = tv_models.mobilenet_v3_small(
            weights=tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
    elif arch == "efficientnet_v2_m":
        model = tv_models.efficientnet_v2_m(
            weights=tv_models.EfficientNet_V2_M_Weights.IMAGENET1K_V1
        )
    elif arch == "convnext_base":
        model = tv_models.convnext_base(
            weights=tv_models.ConvNeXt_Base_Weights.IMAGENET1K_V1
        )
    else:
        raise ValueError(f"unknown pretrained architecture: {arch!r}")

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def build_model(arch: str = "simple", num_classes: int = 5,
                dropout: float = 0.3) -> nn.Module:
    if arch == "simple":
        return SimpleCNN(num_classes=num_classes, dropout=dropout)
    if arch == "residual":
        return ResidualCNN(num_classes=num_classes, dropout=dropout)
    if arch == "residual_deep":
        return ResidualCNN(num_classes=num_classes, dropout=dropout,
                           extra_blocks=True)
    if arch == "residual_se":
        return ResidualCNN(num_classes=num_classes, dropout=dropout,
                           se_last_two=True)
    if arch in PRETRAINED_ARCHS:
        return _build_pretrained(arch, num_classes)
    raise ValueError(f"unknown architecture: {arch!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    x = torch.zeros(2, 3, 224, 224)
    for name in ("simple", "residual", "residual_deep", "residual_se"):
        model = build_model(name)
        print(f"\n{name}: {count_parameters(model):,} parameters, "
              f"output {tuple(model(x).shape)}")
        print(model)
