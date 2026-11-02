"""Dataset and preprocessing transforms for the flower_photos splits.

Augmentation is only applied to the training split; val/test use a
deterministic resize + center crop. All images are converted to RGB,
scaled to [0, 1] and normalized with ImageNet statistics (the standard
choice when fine-tuning a pretrained CNN).

Usage (from training code):
    from dataset import CLASSES, build_transforms, FlowersDataset, build_dataloaders
    loaders = build_dataloaders(batch_size=32, num_workers=4)
    for images, labels in loaders["train"]:
        ...
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

CLASSES = ("daisy", "dandelion", "roses", "sunflowers", "tulips")
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(
    split: str,
    image_size: int = 224,
    resize_size: int = 256,
) -> v2.Compose:
    """Return the transform pipeline for ``train`` / ``val`` / ``test``."""
    if split == "train":
        return v2.Compose([
            # random scale + aspect jitter, then crop to the network input size
            v2.RandomResizedCrop(
                image_size, scale=(0.65, 1.0), ratio=(0.8, 1.25), antialias=True
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    if split in ("val", "test"):
        return v2.Compose([
            # resize the shorter side to 256, then take the central 224x224
            v2.Resize(resize_size, antialias=True),
            v2.CenterCrop(image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    raise ValueError(f"unknown split: {split!r}")


class FlowersDataset(Dataset):
    """Reads a split manifest (path,label,...) and returns (tensor, label)."""

    def __init__(
        self,
        manifest: str | Path,
        data_dir: str | Path = "flower_photos",
        transform: v2.Compose | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.transform = transform
        manifest = Path(manifest)

        with manifest.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise ValueError(f"empty manifest: {manifest}")

        self.samples: list[tuple[Path, int, str]] = []
        for row in rows:
            rel = row["path"]
            if row["label"] not in CLASS_TO_IDX:
                raise ValueError(f"unknown label {row['label']!r} in {rel}")
            path = self.data_dir / rel
            if not path.is_file():
                raise FileNotFoundError(path)
            self.samples.append((path, CLASS_TO_IDX[row["label"]], rel))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label, _ = self.samples[index]
        with Image.open(path) as im:
            image = im.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, label

    @property
    def labels(self) -> list[int]:
        return [label for _, label, _ in self.samples]


def build_dataloaders(
    data_dir: str | Path = "flower_photos",
    splits_dir: str | Path = "splits",
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: int = 224,
    resize_size: int = 256,
) -> dict[str, DataLoader]:
    """Create train/val/test loaders from the CSV manifests."""
    splits_dir = Path(splits_dir)
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        dataset = FlowersDataset(
            splits_dir / f"{split}.csv",
            data_dir,
            build_transforms(split, image_size=image_size, resize_size=resize_size),
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            persistent_workers=num_workers > 0,
        )
    return loaders


def denormalize(
    tensor: torch.Tensor,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
):
    """CHW normalized tensor -> HWC uint8-safe float array in [0, 1]."""
    x = tensor.detach().cpu().clone()
    x = x * torch.tensor(std).view(-1, 1, 1) + torch.tensor(mean).view(-1, 1, 1)
    return x.clamp(0, 1).permute(1, 2, 0).numpy()
