"""Screen every image by model confidence to surface likely non-flower samples.

The model's confidence is used only as a review aid: images are sorted from
least to most confident, and the least confident ones are rendered as contact
sheets. Ensure both correct and incorrect predictions are covered.

Example:
    python src/screen_nonflower.py --checkpoint runs/exp_conv256/best.pt
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
from torch.utils.data import DataLoader

try:
    from dataset import CLASSES, FlowersDataset, build_transforms, denormalize
    from ensemble import infer, load_checkpoint
except ImportError:
    from .dataset import CLASSES, FlowersDataset, build_transforms, denormalize
    from .ensemble import infer, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Confidence screening")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--data-dir", default="flower_photos")
    parser.add_argument("--splits-dir", default="splits")
    parser.add_argument("--out-dir", default="reports/label_audit/screen")
    parser.add_argument("--top-n", type=int, default=60)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    amp = device.type == "cuda"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, arch, image_size, resize_size = load_checkpoint(args.checkpoint, device)
    print(f"checkpoint: {arch}, {image_size}px (resize {resize_size})")

    for split in args.splits:
        ds = FlowersDataset(
            Path(args.splits_dir) / f"{split}.csv",
            args.data_dir,
            build_transforms("val", image_size=image_size,
                             resize_size=resize_size),
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        probs, labels = infer(model, loader, device, amp, tta=not args.no_tta)
        preds = probs.argmax(axis=1)
        conf = probs.max(axis=1)
        order = np.argsort(conf)  # least confident first

        rows = [{
            "path": ds.samples[int(i)][2],
            "label": CLASSES[int(labels[i])],
            "predicted": CLASSES[int(preds[i])],
            "confidence": round(float(conf[i]), 4),
            "correct": bool(preds[i] == labels[i]),
            "split": split,
        } for i in order]
        csv_path = out_dir / f"screen_{split}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{split}: {len(rows)} images, "
              f"{sum(not r['correct'] for r in rows)} errors, wrote {csv_path.name}")

        shown = order[: args.top_n]
        row_by_idx = {int(i): rows[k] for k, i in enumerate(order)}
        cols = 5
        n_rows = int(np.ceil(len(shown) / cols))
        fig, axes = plt.subplots(n_rows, cols, figsize=(cols * 2.5, n_rows * 2.7))
        axes = np.atleast_2d(axes)
        for ax in axes.flat:
            ax.axis("off")
        for ax, idx in zip(axes.flat, shown):
            image, _ = ds[int(idx)]
            ax.imshow(np.clip(denormalize(image), 0, 1))
            r = row_by_idx[int(idx)]
            ax.set_title(
                f"{r['label']} -> {r['predicted']} {conf[idx]:.2f}\n"
                f"{ds.samples[int(idx)][2][:32]}",
                fontsize=6.5,
            )
            ax.axis("off")
        fig.suptitle(f"{split}: {len(shown)} lowest-confidence predictions",
                     fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out_dir / f"screen_{split}.jpg", dpi=120,
                    bbox_inches="tight", pil_kwargs={"quality": 88})
        plt.close(fig)
        print(f"wrote {out_dir / f'screen_{split}.jpg'}")


if __name__ == "__main__":
    main()
