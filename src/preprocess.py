"""Verify and visualise the preprocessing pipeline.

Checks the train/val/test transforms, optionally computes the dataset's own
channel mean/std, and renders a grid of training augmentations.

Usage:
    D:\\SoftWare\\Anaconda\\envs\\ml\\python.exe src/preprocess.py
    ... src/preprocess.py --stats --full-check
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2

try:  # `python src/preprocess.py`
    from dataset import (
        CLASSES,
        IMAGENET_MEAN,
        IMAGENET_STD,
        FlowersDataset,
        build_dataloaders,
        build_transforms,
        denormalize,
    )
except ImportError:  # `python -m src.preprocess`
    from .dataset import (
        CLASSES,
        IMAGENET_MEAN,
        IMAGENET_STD,
        FlowersDataset,
        build_dataloaders,
        build_transforms,
        denormalize,
    )

SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def check_loaders(loaders: dict[str, DataLoader], full: bool) -> dict:
    """Validate shapes/dtypes and report per-split tensor statistics."""
    results = {}
    for split, loader in loaders.items():
        n_images = 0
        n_batches = 0
        channel_sum = torch.zeros(3)
        channel_sq = torch.zeros(3)
        label_counts: Counter[int] = Counter()

        for images, labels in loader:
            if not torch.isfinite(images).all():
                raise ValueError(f"{split}: non-finite values in batch")
            if images.dtype != torch.float32:
                raise TypeError(f"{split}: expected float32, got {images.dtype}")
            if images.ndim != 4 or images.shape[1] != 3:
                raise ValueError(f"{split}: unexpected shape {tuple(images.shape)}")

            batch = images.shape[0]
            channel_sum += images.mean(dim=(0, 2, 3)) * batch
            channel_sq += (images ** 2).mean(dim=(0, 2, 3)) * batch
            n_images += batch
            n_batches += 1
            label_counts.update(labels.tolist())

            if not full and n_batches >= 3:
                break

        mean = channel_sum / n_images
        std = (channel_sq / n_images - mean ** 2).clamp_min(0).sqrt()
        results[split] = {
            "images_checked": n_images,
            "shape": tuple(images.shape[1:]),
            "pixel_range": (float(images.min()), float(images.max())),
            "channel_mean": [round(float(v), 4) for v in mean],
            "channel_std": [round(float(v), 4) for v in std],
            "label_counts": dict(sorted(label_counts.items())),
        }
    return results


def check_deterministic_val(
    data_dir: Path, splits_dir: Path, image_size: int, resize_size: int
) -> bool:
    """Val/test transforms must yield identical tensors for the same image."""
    ds = FlowersDataset(
        splits_dir / "val.csv",
        data_dir,
        build_transforms("val", image_size=image_size, resize_size=resize_size),
    )
    a, label_a = ds[0]
    b, label_b = ds[0]
    return bool(torch.equal(a, b) and label_a == label_b)


def compute_mean_std(
    manifest: Path,
    data_dir: Path,
    image_size: int,
    resize_size: int,
    max_samples: int,
) -> dict:
    """Channel mean/std of the training images in [0, 1] (before normalisation)."""
    raw_tf = v2.Compose([
        v2.Resize(resize_size, antialias=True),
        v2.CenterCrop(image_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    ds = FlowersDataset(manifest, data_dir, raw_tf)
    n = min(len(ds), max_samples)
    loader = DataLoader(Subset(ds, range(n)), batch_size=64, num_workers=0)

    total = torch.zeros(3)
    total_sq = torch.zeros(3)
    seen = 0
    for images, _ in loader:
        b = images.shape[0]
        total += images.mean(dim=(0, 2, 3)) * b
        total_sq += (images ** 2).mean(dim=(0, 2, 3)) * b
        seen += b
    mean = total / seen
    std = (total_sq / seen - mean ** 2).clamp_min(0).sqrt()
    return {
        "samples": seen,
        "mean": [round(float(v), 4) for v in mean],
        "std": [round(float(v), 4) for v in std],
    }


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
def plot_augmentations(
    data_dir: Path,
    splits_dir: Path,
    out_path: Path,
    image_size: int,
    resize_size: int,
    n_aug: int = 4,
) -> None:
    val_tf = build_transforms("val", image_size=image_size, resize_size=resize_size)
    train_tf = build_transforms("train", image_size=image_size, resize_size=resize_size)
    train_ds = FlowersDataset(splits_dir / "train.csv", data_dir, None)

    picks: dict[int, int] = {}
    for idx, (_, label, _) in enumerate(train_ds.samples):
        picks.setdefault(label, idx)
        if len(picks) == len(CLASSES):
            break

    fig, axes = plt.subplots(
        len(CLASSES), n_aug + 1,
        figsize=(2.6 * (n_aug + 1), 2.9 * len(CLASSES)),
    )
    for row, name in enumerate(CLASSES):
        path = train_ds.samples[picks[row]][0]
        with Image.open(path) as im:
            image = im.convert("RGB")

        panels = [("val view (256 -> 224 center crop)", val_tf(image))]
        panels += [(f"train aug {i + 1}", train_tf(image)) for i in range(n_aug)]

        for col, (title, tensor) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(np.clip(denormalize(tensor), 0, 1))
            ax.axis("off")
            if row == 0:
                ax.set_title(title, fontsize=9)
            if col == 0:
                ax.text(-0.06, 0.5, name, transform=ax.transAxes,
                        ha="right", va="center", fontsize=10)
    fig.suptitle("Preprocessing: deterministic val view vs. random train augmentations",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def read_manifest_counts(splits_dir: Path) -> dict[str, Counter]:
    counts = {}
    for split in SPLITS:
        with (splits_dir / f"{split}.csv").open(newline="", encoding="utf-8") as fh:
            counts[split] = Counter(row["label"] for row in csv.DictReader(fh))
    return counts


def write_report(
    out_dir: Path,
    args: argparse.Namespace,
    counts: dict[str, Counter],
    checks: dict,
    det_ok: bool,
    image_stats: dict | None,
) -> None:
    total = sum(sum(c.values()) for c in counts.values())
    lines = []
    a = lines.append
    a("# 数据预处理报告\n")
    a(f"- 数据目录：`{args.data_dir}`")
    a(f"- 划分清单：`{args.splits_dir}`")
    a(f"- 训练输入尺寸：**{args.image_size}×{args.image_size}**；"
      f"验证/测试先缩放短边到 **{args.resize_size}** 再中心裁剪")
    a(f"- 归一化：ImageNet 均值 `{IMAGENET_MEAN}`，标准差 `{IMAGENET_STD}`")
    a(f"- 总图片数：**{total}**\n")

    a("## 各 split 张量检查\n")
    a("| split | 检查张数 | 张量形状 | 像素范围 | 通道均值 (R,G,B) | 通道标准差 (R,G,B) |")
    a("|---|---:|---|---|---|---|")
    for split in SPLITS:
        c = checks.get(split)
        if not c:
            continue
        a(f"| {split} | {c['images_checked']} | {c['shape']} | "
          f"{c['pixel_range'][0]:.2f} ~ {c['pixel_range'][1]:.2f} | "
          f"{c['channel_mean']} | {c['channel_std']} |")
    a("\n张量已转为 float32 并完成归一化，因此像素范围不再固定在 [0, 1]；"
      "均值和标准差偏离 0/1 是因为花卉图像的色彩分布与 ImageNet 不同（尤其绿色背景）。\n")

    a("## 训练增强\n")
    a(f"1. `RandomResizedCrop({args.image_size}, scale=(0.65, 1.0), ratio=(0.8, 1.25))`：随机缩放 + 宽高比扰动")
    a("2. `RandomHorizontalFlip(p=0.5)`：水平翻转")
    a("3. `ColorJitter(0.2, 0.2, 0.2, 0.05)`：亮度/对比度/饱和度/色调扰动")
    a("4. `ToImage → float32 [0,1] → Normalize(ImageNet)`")
    a("\n验证/测试只做 `Resize(256) → CenterCrop(224) → Normalize`，保证可复现。\n")

    a("## 归一化统计\n")
    a(f"- 当前使用：ImageNet mean = `{IMAGENET_MEAN}`，std = `{IMAGENET_STD}`")
    if image_stats:
        a(f"- 训练集自身统计（{image_stats['samples']} 张，[0,1] 范围）："
          f"mean = `{tuple(image_stats['mean'])}`，std = `{tuple(image_stats['std'])}`")
        a("- 若从零训练可改用上述统计；微调 ImageNet 预训练模型时保持 ImageNet 统计更合适。")
    else:
        a("- 未计算训练集统计，可加 `--stats` 查看。")
    a("")

    a("## 校验结果\n")
    a(f"- 验证集确定性：{'通过（同一张图两次输出完全一致）' if det_ok else '失败'}")
    a("- 所有检查批次均为有限值、float32、形状 (3, H, W)")
    a("")

    a("## 示例\n")
    a("![预处理与增强示例](figures/09_preprocessing.png)\n")

    (out_dir / "preprocessing_report.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Verify flower preprocessing pipeline")
    parser.add_argument("--data-dir", default="flower_photos")
    parser.add_argument("--splits-dir", default="splits")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--stats", action="store_true",
                        help="compute training-set channel mean/std")
    parser.add_argument("--stats-samples", type=int, default=1000)
    parser.add_argument("--full-check", action="store_true",
                        help="iterate every image instead of the first 3 batches")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    splits_dir = Path(args.splits_dir)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Building dataloaders ...")
    loaders = build_dataloaders(
        data_dir=data_dir,
        splits_dir=splits_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        resize_size=args.resize_size,
    )
    for split, loader in loaders.items():
        print(f"  {split:<5} {len(loader.dataset):>5} images, {len(loader):>3} batches")

    print("Checking batches ...")
    checks = check_loaders(loaders, full=args.full_check)
    for split, c in checks.items():
        print(f"  {split:<5} checked={c['images_checked']:>4} "
              f"shape={c['shape']} range=({c['pixel_range'][0]:.2f}, "
              f"{c['pixel_range'][1]:.2f})")
        print(f"        mean={c['channel_mean']} std={c['channel_std']}")

    print("Checking val determinism ...")
    det_ok = check_deterministic_val(
        data_dir, splits_dir, args.image_size, args.resize_size
    )
    print(f"  {'ok' if det_ok else 'FAILED'}")

    image_stats = None
    if args.stats:
        print(f"Computing train mean/std ({args.stats_samples} samples) ...")
        image_stats = compute_mean_std(
            splits_dir / "train.csv", data_dir,
            args.image_size, args.resize_size, args.stats_samples,
        )
        print(f"  mean={image_stats['mean']} std={image_stats['std']}")

    if not args.no_figure:
        print("Rendering augmentation grid ...")
        plot_augmentations(
            data_dir, splits_dir, fig_dir / "09_preprocessing.png",
            args.image_size, args.resize_size,
        )

    counts = read_manifest_counts(splits_dir)
    write_report(out_dir, args, counts, checks, det_ok, image_stats)
    print(f"  wrote {out_dir / 'preprocessing_report.md'}")


if __name__ == "__main__":
    main()
