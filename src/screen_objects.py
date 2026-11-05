"""Flag likely non-flower images with an ImageNet classifier.

This is a review aid only: every flagged image still needs manual confirmation
before it is excluded from the cleaned splits.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import models

try:
    from dataset import FlowersDataset, build_transforms, denormalize
except ImportError:
    from .dataset import FlowersDataset, build_transforms, denormalize

NON_FLOWER = (
    "car", "cab", "limousine", "jeep", "minivan", "truck", "bus", "van",
    "ambulance", "motorcycle", "moped", "bicycle", "tricycle", "unicycle",
    "train", "locomotive", "streetcar", "boat", "ship", "canoe", "yawl",
    "airplane", "airship", "warplane", "tank", "missile", "cannon", "bomb",
    "church", "mosque", "monastery", "palace", "dome", "castle", "barn",
    "boathouse", "lighthouse", "bridge", "viaduct", "triumphal",
    "library", "theater", "cinema", "hospital", "hotel", "prison",
    "planetarium", "stupa", "pagoda", "yurt", "mobile home",
    "dog", "cat", "bird", "eagle", "owl", "parrot", "hummingbird",
    "butterfly", "moth", "dragonfly", "ladybug", "grasshopper", "beetle",
    "spider", "scorpion", "snake", "lizard", "frog", "turtle", "fish",
    "shark", "whale", "dolphin", "bear", "zebra", "horse", "cow", "sheep",
    "goat", "pig", "elephant", "tiger", "lion", "monkey", "rabbit",
    "squirrel", "deer", "camel", "kangaroo", "fox", "wolf",
    "pizza", "cake", "ice cream", "burger", "hotdog", "bagel", "pretzel",
    "burrito", "guacamole", "carbonara", "rifle", "revolver", "assault",
    "valley", "lakeside", "seashore", "cliff", "geyser", "promontory",
    "sandbar", "coral reef", "volcano", "iceberg", "glacier", "snowfield",
)


def is_non_flower(name: str) -> bool:
    name = name.lower()
    return any(word in name for word in NON_FLOWER)


def main() -> None:
    parser = argparse.ArgumentParser(description="ImageNet non-flower screening")
    parser.add_argument("--data-dir", default="flower_photos")
    parser.add_argument("--splits-dir", default="splits")
    parser.add_argument("--out-dir", default="reports/label_audit/object_screen")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1
    model = models.convnext_base(weights=weights).to(device).eval()
    categories = weights.meta["categories"]

    tf = build_transforms("val", image_size=224, resize_size=256)
    datasets, owners = [], []
    for split in ("train", "val", "test"):
        ds = FlowersDataset(Path(args.splits_dir) / f"{split}.csv",
                            args.data_dir, tf)
        datasets.append(ds)
        owners += [(split, ds.samples[i][2]) for i in range(len(ds))]
    loader = DataLoader(ConcatDataset(datasets), batch_size=args.batch_size,
                        shuffle=False, num_workers=0)

    rows = []
    offset = 0
    with torch.inference_mode():
        for images, _ in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits.float(), dim=1).cpu()
            top_probs, top_idx = probs.topk(5, dim=1)
            for k in range(images.shape[0]):
                split, path = owners[offset + k]
                names = [categories[i] for i in top_idx[k].tolist()]
                rows.append({
                    "split": split,
                    "path": path,
                    "top1": names[0],
                    "prob1": round(float(top_probs[k, 0]), 4),
                    "nonflower_flag": is_non_flower(names[0])
                                     and float(top_probs[k, 0]) >= 0.35,
                    "top5": " | ".join(
                        f"{n}({p:.2f})" for n, p in
                        zip(names, top_probs[k].tolist())
                    ),
                })
            offset += images.shape[0]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "object_screen.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    flagged = [r for r in rows if r["nonflower_flag"]]
    print(f"{len(rows)} images, {len(flagged)} flagged; wrote {csv_path}")

    for split in ("train", "val", "test"):
        items = [r for r in flagged if r["split"] == split]
        items.sort(key=lambda r: -r["prob1"])
        print(f"  {split}: {len(items)} flagged")
        if not items:
            continue
        shown = items[:80]
        cols = 5
        n_rows = int(np.ceil(len(shown) / cols))
        fig, axes = plt.subplots(n_rows, cols, figsize=(cols * 2.6, n_rows * 2.8))
        axes = np.atleast_2d(axes)
        for ax in axes.flat:
            ax.axis("off")
        ds_index = {"train": 0, "val": 1, "test": 2}
        ds = datasets[ds_index[split]]
        local = {samples[2]: i for i, samples in enumerate(ds.samples)}
        for ax, r in zip(axes.flat, shown):
            image, _ = ds[local[r["path"]]]
            ax.imshow(np.clip(denormalize(image), 0, 1))
            ax.set_title(f"{r['top1']} {r['prob1']:.2f}\n{r['path'][:34]}",
                         fontsize=6.5)
            ax.axis("off")
        fig.suptitle(f"{split}: ImageNet non-flower candidates ({len(items)})",
                     fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out_dir / f"objects_{split}.jpg", dpi=120,
                    bbox_inches="tight", pil_kwargs={"quality": 88})
        plt.close(fig)
        print(f"    wrote {out_dir / f'objects_{split}.jpg'}")


if __name__ == "__main__":
    main()
