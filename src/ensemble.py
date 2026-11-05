"""Evaluate single models and probability ensembles on val/test.

Example:
    python src/ensemble.py `
        --checkpoints runs/exp_conv_ema/best.pt runs/exp_eff_ema/best.pt runs/exp_mobilenet/best.pt `
        --names convnext_ema efficientnet_ema mobilenet

Combination weights are equal and the best subset is selected on the validation
split; the test split is only reported for that chosen subset.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

try:
    from dataset import CLASSES, FlowersDataset, build_transforms
    from model import build_model
except ImportError:
    from .dataset import CLASSES, FlowersDataset, build_transforms
    from .model import build_model


def load_checkpoint(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    arch = cfg.get("arch", "simple")
    image_size = int(cfg.get("image_size", 224))
    resize_size = int(cfg.get("resize_size", 256))
    model = build_model(arch, num_classes=len(CLASSES)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, arch, image_size, resize_size


@torch.inference_mode()
def infer(model, loader, device, amp: bool, tta: bool = True):
    probs_all, labels_all = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=amp):
            logits = model(images)
            probs = torch.softmax(logits.float(), dim=1)
            if tta:
                logits_flip = model(torch.flip(images, dims=[3]))
                probs = (probs + torch.softmax(logits_flip.float(), dim=1)) / 2.0
        probs_all.append(probs.cpu())
        labels_all.append(labels)
    return torch.cat(probs_all).numpy(), torch.cat(labels_all).numpy()


def score(probs: np.ndarray, labels: np.ndarray) -> dict:
    preds = probs.argmax(axis=1)
    report = classification_report(
        labels, preds, labels=list(range(len(CLASSES))),
        target_names=CLASSES, output_dict=True, zero_division=0,
    )
    return {
        "accuracy": float((preds == labels).mean()),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class": report,
        "confusion_matrix": confusion_matrix(
            labels, preds, labels=list(range(len(CLASSES)))
        ).tolist(),
    }


def plot_confusion(cm: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(len(CLASSES)), yticks=np.arange(len(CLASSES)),
           xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted", ylabel="True", title=title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probability ensemble evaluation")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--data-dir", default="flower_photos")
    parser.add_argument("--splits-dir", default="splits")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--suffix", default="",
                        help="suffix for output filenames, e.g. _256")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    names = args.names or [Path(p).parent.parent.name for p in args.checkpoints]
    if len(names) != len(args.checkpoints):
        parser.error("--names must match --checkpoints length")

    device = torch.device(args.device)
    amp = device.type == "cuda"

    specs = []
    for i, path in enumerate(args.checkpoints):
        model, arch, image_size, resize_size = load_checkpoint(path, device)
        specs.append((names[i], model, image_size, resize_size))
        print(f"loaded {path} ({arch}, {image_size}px, resize {resize_size})")

    loader_cache = {}
    def get_loader(split: str, image_size: int, resize_size: int) -> DataLoader:
        key = (split, image_size, resize_size)
        if key not in loader_cache:
            ds = FlowersDataset(
                Path(args.splits_dir) / f"{split}.csv",
                args.data_dir,
                build_transforms("val", image_size=image_size,
                                 resize_size=resize_size),
            )
            loader_cache[key] = DataLoader(
                ds, batch_size=args.batch_size, shuffle=False
            )
        return loader_cache[key]

    single, probs_by_split, labels_by_split = {}, {}, {}
    for split in ("val", "test"):
        probs_by_split[split] = []
        for name, model, image_size, resize_size in specs:
            loader = get_loader(split, image_size, resize_size)
            probs, labels = infer(model, loader, device, amp,
                                  tta=not args.no_tta)
            probs_by_split[split].append(probs)
            if split not in labels_by_split:
                labels_by_split[split] = labels
            elif not np.array_equal(labels_by_split[split], labels):
                raise RuntimeError("label order differs between splits")
            m = score(probs, labels)
            single[f"{name} ({split})"] = {**m, "image_size": image_size}
            print(f"{name:<18} {split:<4} {image_size}px "
                  f"acc={m['accuracy']:.4f} f1={m['macro_f1']:.4f}")

    # equal-weight subsets, selected on validation accuracy
    combos = []
    for r in range(1, len(names) + 1):
        for idx in combinations(range(len(names)), r):
            val_probs = np.mean([probs_by_split["val"][i] for i in idx], axis=0)
            val_m = score(val_probs, labels_by_split["val"])
            combos.append({
                "members": [names[i] for i in idx],
                "idx": idx,
                "val_acc": val_m["accuracy"],
                "val_f1": val_m["macro_f1"],
            })
    combos.sort(key=lambda c: (-c["val_acc"], -c["val_f1"]))

    best = combos[0]
    test_probs = np.mean([probs_by_split["test"][i] for i in best["idx"]], axis=0)
    test_m = score(test_probs, labels_by_split["test"])
    print(f"best subset on val: {'+'.join(best['members'])} "
          f"val={best['val_acc']:.4f} test={test_m['accuracy']:.4f}")

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion(np.array(test_m["confusion_matrix"]),
                   fig_dir / f"14_ensemble_confusion{args.suffix}.png",
                   f"Ensemble: {'+'.join(best['members'])} (test)")

    lines = ["# 迁移模型集成结果\n"]
    lines.append(f"- 各模型使用自身训练分辨率（见下表）；"
                 f"TTA：{'off' if args.no_tta else 'horizontal flip'}")
    lines.append(f"- 集成方式：等权概率平均；组合在 val 上选择\n")
    lines.append("## 单模型\n")
    lines.append("| 模型 | split | input | acc | macro F1 |")
    lines.append("|---|---|---:|---:|---:|")
    for key, m in single.items():
        name, split = key.rsplit(" (", 1)
        lines.append(f"| {name} | {split[:-1]} | {m['image_size']} | "
                     f"{m['accuracy']:.4f} | {m['macro_f1']:.4f} |")

    lines.append("\n## 集成组合（按 val 排序，前 10）\n")
    lines.append("| 组合 | val acc | val F1 | test acc | test F1 |")
    lines.append("|---|---:|---:|---:|---:|")
    for c in combos[:10]:
        tp = np.mean([probs_by_split["test"][i] for i in c["idx"]], axis=0)
        tm = score(tp, labels_by_split["test"])
        lines.append(f"| {'+'.join(c['members'])} | {c['val_acc']:.4f} | "
                     f"{c['val_f1']:.4f} | {tm['accuracy']:.4f} | "
                     f"{tm['macro_f1']:.4f} |")

    lines.append(f"\n## 最终选择：{'+'.join(best['members'])}\n")
    lines.append("| 类别 | precision | recall | f1 | support |")
    lines.append("|---|---:|---:|---:|---:|")
    for name in CLASSES:
        m = test_m["per_class"][name]
        lines.append(f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | "
                     f"{m['f1-score']:.3f} | {int(m['support'])} |")
    lines.append(f"\n- val acc = {best['val_acc']:.4f}")
    lines.append(f"- test acc = **{test_m['accuracy']:.4f}**")
    lines.append(f"- test macro F1 = **{test_m['macro_f1']:.4f}**")
    lines.append("\n![ensemble confusion](figures/14_ensemble_confusion"
                 f"{args.suffix}.png)\n")

    (out_dir / f"ensemble_report{args.suffix}.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    (table_dir / f"ensemble_metrics{args.suffix}.json").write_text(
        json.dumps({
            "single": single,
            "combos": [
                {"members": c["members"], "val_acc": c["val_acc"],
                 "val_f1": c["val_f1"]}
                for c in combos
            ],
            "best": {"members": best["members"], "val_acc": best["val_acc"],
                     "test": test_m},
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out_dir / f'ensemble_report{args.suffix}.md'}")


if __name__ == "__main__":
    main()
