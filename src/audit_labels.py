"""List likely mislabeled samples for a trained checkpoint.

Outputs one CSV + one image grid per split under ``reports/label_audit``.
Test-set findings are written separately so the original benchmark stays intact.

Example:
    python src/audit_labels.py --checkpoint runs/exp_conv256/best.pt `
        --splits train val test
"""

from __future__ import annotations

import argparse
import csv
import json
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
    parser = argparse.ArgumentParser(description="Audit suspicious labels")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        choices=("train", "val", "test"))
    parser.add_argument("--data-dir", default="flower_photos")
    parser.add_argument("--splits-dir", default="splits")
    parser.add_argument("--out-dir", default="reports/label_audit")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--plot-k", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    amp = device.type == "cuda"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, arch, image_size, resize_size = load_checkpoint(
        args.checkpoint, device
    )
    print(f"checkpoint: {arch}, {image_size}px (resize {resize_size})")

    summary = {}
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
        conf = probs[np.arange(len(preds)), preds]

        wrong = np.where(preds != labels)[0]
        order = wrong[np.argsort(-conf[wrong])]
        rows = [{
            "path": ds.samples[int(i)][2],
            "label": CLASSES[int(labels[i])],
            "predicted": CLASSES[int(preds[i])],
            "confidence": round(float(conf[i]), 4),
            "split": split,
        } for i in order[: args.top_k]]

        csv_path = out_dir / f"{split}_suspicious.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else
                                    ["path", "label", "predicted", "confidence", "split"])
            writer.writeheader()
            writer.writerows(rows)

        summary[split] = {
            "images": len(ds),
            "errors": int(len(wrong)),
            "suspicious_written": len(rows),
        }
        print(f"{split}: {len(wrong)}/{len(ds)} errors, wrote top {len(rows)}")

        if not rows:
            continue
        shown = order[: min(args.plot_k, len(order))]
        cols = 4
        n_rows = int(np.ceil(len(shown) / cols))
        fig, axes = plt.subplots(n_rows, cols,
                                 figsize=(cols * 3.2, n_rows * 3.6))
        axes = np.atleast_2d(axes)
        for ax in axes.flat:
            ax.axis("off")
        for ax, idx in zip(axes.flat, shown):
            image, true_label = ds[int(idx)]
            ax.imshow(np.clip(denormalize(image), 0, 1))
            ax.set_title(
                f"{CLASSES[int(true_label)]} -> {CLASSES[int(preds[idx])]} "
                f"(p={conf[idx]:.2f})\n{ds.samples[int(idx)][2][:38]}",
                fontsize=7.5,
            )
            ax.axis("off")
        fig.suptitle(f"{split}: most confident label mismatches", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.2)
        fig.savefig(out_dir / f"{split}_suspicious.jpg", dpi=110,
                    bbox_inches="tight", pil_kwargs={"quality": 88})
        plt.close(fig)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
