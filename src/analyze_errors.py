"""Inspect misclassified validation/test images for a trained checkpoint.

Example:
    & "D:\\SoftWare\\Anaconda\\envs\\ml\\python.exe" src/analyze_errors.py `
        --checkpoint runs/cnn_baseline_80/best.pt --split val `
        --classes roses tulips --out-dir reports/cnn_baseline_80
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

try:
    from dataset import (
        CLASSES,
        FlowersDataset,
        build_transforms,
        denormalize,
    )
    from model import build_model
except ImportError:
    from .dataset import (
        CLASSES,
        FlowersDataset,
        build_transforms,
        denormalize,
    )
    from .model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse checkpoint errors")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--data-dir", default="flower_photos")
    parser.add_argument("--splits-dir", default="splits")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--classes", nargs="+", default=["roses", "tulips"],
                        help="only show errors where true and predicted labels "
                             "are both in this set")
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = ckpt.get("config", {})
    image_size = int(cfg.get("image_size", 224))
    resize_size = int(cfg.get("resize_size", 256))

    dataset = FlowersDataset(
        Path(args.splits_dir) / f"{args.split}.csv",
        args.data_dir,
        build_transforms("val", image_size=image_size, resize_size=resize_size),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    model = build_model(cfg.get("arch", "simple"), num_classes=len(CLASSES)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_probs, all_preds, all_labels = [], [], []
    with torch.inference_mode():
        for images, labels in loader:
            probs = torch.softmax(model(images.to(device)), dim=1).cpu()
            all_probs.append(probs)
            all_preds.append(probs.argmax(1))
            all_labels.append(labels)
    probs = torch.cat(all_probs).numpy()
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASSES))))
    report = classification_report(
        labels, preds, labels=list(range(len(CLASSES))),
        target_names=CLASSES, output_dict=True, zero_division=0,
    )

    focus = [CLASSES.index(c) for c in args.classes]
    wrong = np.where((preds != labels) & np.isin(labels, focus) & np.isin(preds, focus))[0]
    wrong = wrong[np.argsort(-probs[wrong, preds[wrong]])][: args.max_images]

    pair_counts = {
        f"{CLASSES[i]}->{CLASSES[j]}": int(cm[i, j])
        for i in focus for j in focus if i != j and cm[i, j]
    }
    print(f"{args.split}: {(preds != labels).sum()} errors / {len(labels)}")
    for name in args.classes:
        m = report[name]
        print(f"  {name:<9} precision={m['precision']:.3f} "
              f"recall={m['recall']:.3f} f1={m['f1-score']:.3f}")
    print(f"  bidirectional: {pair_counts}")

    metrics = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "accuracy": float((preds == labels).mean()),
        "errors": int((preds != labels).sum()),
        "per_class": report,
        "confusion_matrix": cm.tolist(),
        "focus_pairs": pair_counts,
    }
    (out_dir / f"{args.split}_errors_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if len(wrong) == 0:
        print("no matching errors to plot")
        return

    cols = 4
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.6))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for ax, idx in zip(axes.flat, wrong):
        image, true_label = dataset[int(idx)]
        ax.imshow(np.clip(denormalize(image), 0, 1))
        conf = float(probs[idx, preds[idx]])
        ax.set_title(
            f"true: {CLASSES[true_label]} -> pred: {CLASSES[preds[idx]]}\n"
            f"p={conf:.2f}",
            fontsize=8.5,
        )
        ax.axis("off")
    fig.suptitle(
        f"{args.split} errors among {'/'.join(args.classes)} "
        f"(sorted by confidence, top {len(wrong)})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.4)
    path = out_dir / f"{args.split}_errors_{'_'.join(args.classes)}.jpg"
    fig.savefig(path, dpi=120, bbox_inches="tight", pil_kwargs={"quality": 88})
    plt.close(fig)
    print(f"wrote {path}")
    print(f"wrote {out_dir / f'{args.split}_errors_metrics.json'}")


if __name__ == "__main__":
    main()
