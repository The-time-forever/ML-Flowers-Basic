"""Train and evaluate the SimpleCNN baseline on the flower splits.

Example:
    & "D:\\SoftWare\\Anaconda\\envs\\ml\\python.exe" src/train.py
    ... src/train.py --epochs 30 --batch-size 32 --lr 1e-3 --num-workers 4

Artefacts:
    runs/cnn_baseline/best.pt, last.pt, history.csv   (checkpoints / log)
    reports/cnn_baseline_report.md                    (Chinese report)
    reports/tables/cnn_baseline_metrics.json
    reports/figures/10_training_curves.png
    reports/figures/11_confusion_matrix.png
    reports/figures/12_misclassified.png
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn

try:  # `python src/train.py`
    from dataset import (
        CLASSES,
        IMAGENET_MEAN,
        IMAGENET_STD,
        build_dataloaders,
        build_eval_loader,
        denormalize,
    )
    from model import build_model, count_parameters
except ImportError:  # `python -m src.train`
    from .dataset import (
        CLASSES,
        IMAGENET_MEAN,
        IMAGENET_STD,
        build_dataloaders,
        build_eval_loader,
        denormalize,
    )
    from .model import build_model, count_parameters


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
class ModelEMA:
    """Exponential moving average of model weights and buffers."""

    def __init__(self, model: nn.Module, decay: float = 0.998):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for param in self.ema.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for ema_v, model_v in zip(self.ema.state_dict().values(),
                                  model.state_dict().values()):
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(d).add_(model_v.detach(), alpha=1.0 - d)
            else:
                ema_v.copy_(model_v)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler=None,
    amp: bool = False,
    ema: ModelEMA | None = None,
    hflip: bool = False,
):
    """One pass over ``loader``; training when ``optimizer`` is given."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0.0
    total = 0
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_probs: list[torch.Tensor] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if hflip:
            images = torch.flip(images, dims=[3])

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                if ema is not None:
                    ema.update(model)

        batch = labels.size(0)
        total_loss += loss.item() * batch
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch

        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_probs.append(torch.softmax(logits.detach(), dim=1).float().cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = torch.cat(all_probs).numpy()
    return total_loss / total, correct / total, preds, labels, probs


def save_history(history: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_curves(history: list[dict], best_epoch: int, path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    has_clean = "train_eval_acc" in history[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    ax.plot(epochs, [h["train_loss"] for h in history], label="train (augmented)")
    if has_clean:
        ax.plot(epochs, [h["train_eval_loss"] for h in history],
                label="train (no aug)", color="green", lw=1.5)
    ax.plot(epochs, [h["val_loss"] for h in history], label="val")
    ax.axvline(best_epoch, ls="--", c="gray", lw=1, label=f"best epoch ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.plot(epochs, [h["train_acc"] for h in history], label="train (augmented)")
    if has_clean:
        ax.plot(epochs, [h["train_eval_acc"] for h in history],
                label="train (no aug)", color="green", lw=1.5)
    ax.plot(epochs, [h["val_acc"] for h in history], label="val")
    ax.axvline(best_epoch, ls="--", c="gray", lw=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(CLASSES)),
        yticks=np.arange(len(CLASSES)),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion matrix (test split)",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_misclassified(
    dataset,
    preds: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    path: Path,
    max_images: int = 12,
) -> int:
    wrong = np.where(preds != labels)[0]
    if len(wrong) == 0:
        return 0
    # most confident mistakes first
    wrong = wrong[np.argsort(-probs[wrong, preds[wrong]])][:max_images]

    cols = 4
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.1, rows * 3.5))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")

    for ax, idx in zip(axes.flat, wrong):
        image, true_label = dataset[int(idx)]
        ax.imshow(np.clip(denormalize(image), 0, 1))
        confidence = float(probs[idx, preds[idx]])
        ax.set_title(
            f"true: {CLASSES[true_label]}  →  pred: {CLASSES[preds[idx]]}\n"
            f"confidence {confidence:.2f}",
            fontsize=8.5,
        )
        ax.axis("off")

    fig.suptitle("Most confident test-set mistakes", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.4)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return len(wrong)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def write_report(
    args, params: int, history, best_epoch, best_val_acc,
    test_acc, macro_f1, report_dict, cm, n_wrong, fig_dir, report_path,
) -> None:
    lines = []
    a = lines.append
    a("# CNN Baseline 训练报告\n")
    model_info = {
        "simple": ("SimpleCNN", "4 个 conv block，Global Average Pooling"),
        "residual": ("ResidualCNN",
                     "4 个 residual block（shortcut），Global Average Pooling"),
        "residual_deep": ("ResidualCNN (deep)",
                          "6 个 residual block（后两个 stage 各多一个），"
                          "Global Average Pooling"),
        "residual_se": ("ResidualCNN + SE",
                        "4 个 residual block（后两个 stage 加 SE，reduction=16），"
                        "Global Average Pooling"),
        "mobilenet_v3_small": ("MobileNetV3-Small (pretrained)",
                               "ImageNet 预训练，全部参数微调"),
        "efficientnet_v2_m": ("EfficientNetV2-M (pretrained)",
                              "ImageNet 预训练，全部参数微调"),
        "convnext_base": ("ConvNeXt-Base (pretrained)",
                          "ImageNet 预训练，全部参数微调"),
    }
    model_name, block_desc = model_info.get(
        args.arch, (args.arch, "unknown architecture")
    )
    a(f"- 模型：`{model_name}`（{block_desc}，"
      f"**{params:,}** 可训练参数）")
    aug_bits = [f"RandomResizedCrop({args.crop_scale_min}-{args.crop_scale_max})"]
    if args.rotation > 0:
        aug_bits.append(f"RandomRotation(±{args.rotation:g}°)")
    aug_bits += ["水平翻转", f"ColorJitter(hue={args.hue})"]
    a(f"- 输入：{args.image_size}×{args.image_size}（验证 Resize({args.resize_size}) "
      f"→ CenterCrop({args.image_size})），ImageNet 归一化；"
      f"训练增强：" + "、".join(aug_bits))
    a(f"- 优化：AdamW lr={args.lr}, weight_decay={args.weight_decay}, "
      f"cosine schedule, label smoothing={args.label_smoothing}, AMP={not args.no_amp}"
      + (f", EMA decay={args.ema_decay}" if args.ema_decay > 0 else "")
      + (", test TTA=horizontal flip" if args.tta else ""))
    a(f"- 训练：{args.epochs} epochs（early stopping patience={args.patience}），"
      f"batch size={args.batch_size}，seed={args.seed}")
    a("- 诊断指标：每轮训练后用**验证预处理**（无增强、dropout off）再评估一次训练集，"
      "得到可与 val 直接比较的 `train_eval_acc`\n")

    best_row = next((h for h in history if h["epoch"] == best_epoch), {})
    clean_acc = best_row.get("train_eval_acc")

    a("## 结果\n")
    a(f"| 指标 | 数值 |")
    a("|---|---:|")
    a(f"| 最佳 epoch | {best_epoch} |")
    a(f"| 最佳验证准确率 | {best_val_acc:.4f} |")
    if clean_acc is not None:
        a(f"| 最佳 epoch 训练集准确率（无增强） | {clean_acc:.4f} |")
        a(f"| 验证 - 训练差距 | {best_val_acc - clean_acc:+.4f} |")
    a(f"| 测试准确率 | **{test_acc:.4f}** |")
    a(f"| 测试 Macro-F1 | **{macro_f1:.4f}** |")
    a(f"| 测试错误数 | {n_wrong} / {int(cm.sum())} |")
    a("")

    if clean_acc is not None:
        gap = best_val_acc - clean_acc
        if gap < -0.05:
            diagnosis = (f"验证集比无增强训练集低 {abs(gap)*100:.1f} 个百分点，"
                         "存在一定过拟合；可加强增强、提高 weight decay、使用 EMA 或增加数据。")
        elif gap > 0.01:
            diagnosis = (f"验证集比无增强训练集高 {gap*100:.1f} 个百分点，"
                         "说明增强/正则较强或训练集更难，模型尚未充分利用训练集。")
        else:
            diagnosis = (f"验证集与无增强训练集仅差 {abs(gap)*100:.1f} 个百分点，"
                         "拟合程度合理，瓶颈更可能在模型容量或数据本身。")
        a(f"**拟合诊断**：{diagnosis}\n")

    a("## 各类别指标（test）\n")
    a("| 类别 | precision | recall | f1-score | support |")
    a("|---|---:|---:|---:|---:|")
    for name in CLASSES:
        m = report_dict[name]
        a(f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | "
          f"{m['f1-score']:.3f} | {int(m['support'])} |")
    avg = report_dict["macro avg"]
    a(f"| **macro avg** | **{avg['precision']:.3f}** | **{avg['recall']:.3f}** | "
      f"**{avg['f1-score']:.3f}** | **{int(avg['support'])}** |")
    a("")

    a("## 训练曲线\n")
    a(f"![训练曲线]({fig_dir}/10_training_curves.png)\n")
    a("## 混淆矩阵\n")
    a(f"![混淆矩阵]({fig_dir}/11_confusion_matrix.png)\n")
    if n_wrong:
        a("## 高置信度错误样本\n")
        a(f"![错误样本]({fig_dir}/12_misclassified.png)\n")

    a("## 说明\n")
    a("- 这是**从零训练**的小型 CNN，不使用预训练权重，用于建立基线。")
    a("- 若验证/测试差距较大，说明过拟合，可加强增强、加大 weight decay 或改用预训练模型。")
    a("- 混淆矩阵可用于判断类别间的视觉相似性（如 daisy/tulips、dandelion/sunflowers）。")
    a("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def read_history(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [
        {k: (int(v) if k == "epoch" else float(v)) for k, v in row.items()}
        for row in rows
    ]


def train_model(model, loaders, criterion, optimizer, scheduler, scaler,
                device, amp, args, out_dir, train_eval_loader=None):
    """Run the training loop and return (history, best_epoch, best_acc)."""
    history: list[dict] = []
    best_acc = 0.0
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    ema = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc, _, _, _ = run_epoch(
            model, loaders["train"], criterion, device,
            optimizer=optimizer, scaler=scaler, amp=amp, ema=ema,
        )
        # start EMA after one epoch so the new classifier head is not averaged
        # against its random initialisation on short fine-tuning schedules
        if args.ema_decay > 0 and ema is None:
            ema = ModelEMA(model, args.ema_decay)
        eval_model = ema.ema if ema is not None else model
        val_loss, val_acc, _, _, _ = run_epoch(
            eval_model, loaders["val"], criterion, device, amp=amp,
        )
        clean_loss = clean_acc = None
        if train_eval_loader is not None:
            clean_loss, clean_acc, _, _, _ = run_epoch(
                eval_model, train_eval_loader, criterion, device, amp=amp,
            )
        scheduler.step()
        elapsed = time.time() - t0

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 5),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 5),
            "seconds": round(elapsed, 2),
        }
        if clean_acc is not None:
            row["train_eval_loss"] = round(clean_loss, 5)
            row["train_eval_acc"] = round(clean_acc, 5)
        history.append(row)

        improved = (val_acc > best_acc + 1e-9) or (
            abs(val_acc - best_acc) <= 1e-9 and val_loss < best_loss
        )
        if improved:
            best_acc, best_loss, best_epoch = val_acc, val_loss, epoch
            epochs_without_improvement = 0
            torch.save({
                "model": eval_model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "config": vars(args),
                "classes": CLASSES,
                "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
            }, out_dir / "best.pt")
        else:
            epochs_without_improvement += 1

        mark = " *" if improved else ""
        clean_text = f" clean {clean_acc:.4f}" if clean_acc is not None else ""
        print(f"epoch {epoch:02d}/{args.epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f}{clean_text} | "
              f"{elapsed:.1f}s{mark}", flush=True)

        if epochs_without_improvement >= args.patience:
            print(f"early stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)")
            break

    torch.save({
        "model": (ema.ema if ema is not None else model).state_dict(),
        "epoch": history[-1]["epoch"],
        "config": vars(args),
        "classes": CLASSES,
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
    }, out_dir / "last.pt")
    save_history(history, out_dir / "history.csv")
    return history, best_epoch, best_acc


def evaluate_and_export(model, loaders, criterion, device, amp, args,
                        history, best_epoch, best_acc, params,
                        out_dir, reports_dir) -> None:
    """Load the best checkpoint, evaluate test, write figures/report/metrics."""
    fig_dir = reports_dir / "figures"
    table_dir = reports_dir / "tables"

    print(f"loading best checkpoint (epoch {best_epoch}, val acc {best_acc:.4f}) ...")
    checkpoint = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_loss, test_acc, preds, labels, probs = run_epoch(
        model, loaders["test"], criterion, device, amp=amp,
    )
    if args.tta:
        _, _, _, _, probs_flip = run_epoch(
            model, loaders["test"], criterion, device, amp=amp, hflip=True,
        )
        probs = (probs + probs_flip) / 2.0
        preds = probs.argmax(axis=1)
        test_acc = float((preds == labels).mean())
        print("applied horizontal-flip TTA")
    report_dict = classification_report(
        labels, preds, labels=list(range(len(CLASSES))),
        target_names=CLASSES, output_dict=True, zero_division=0,
    )
    macro_f1 = float(report_dict["macro avg"]["f1-score"])
    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASSES))))
    n_wrong = int((preds != labels).sum())
    print(f"test: loss={test_loss:.4f} acc={test_acc:.4f} macro_f1={macro_f1:.4f}")

    plot_curves(history, best_epoch, fig_dir / "10_training_curves.png")
    plot_confusion(cm, fig_dir / "11_confusion_matrix.png")
    plot_misclassified(
        loaders["test"].dataset, preds, labels, probs,
        fig_dir / "12_misclassified.png",
    )

    metrics = {
        "config": {k: v for k, v in vars(args).items() if k != "eval_only"},
        "parameters": params,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_acc": round(best_acc, 5),
        "test_loss": round(test_loss, 5),
        "test_acc": round(test_acc, 5),
        "test_macro_f1": round(macro_f1, 5),
        "confusion_matrix": cm.tolist(),
        "per_class": report_dict,
        "misclassified": n_wrong,
    }
    (table_dir / "cnn_baseline_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(
        args, params, history, best_epoch, best_acc, test_acc, macro_f1,
        report_dict, cm, n_wrong, "figures", reports_dir / "cnn_baseline_report.md",
    )

    print(f"wrote {out_dir}/best.pt, history.csv")
    print(f"wrote {reports_dir}/cnn_baseline_report.md and figures")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Train a flower CNN")
    parser.add_argument("--arch", choices=("simple", "residual", "residual_deep",
                                           "residual_se", "mobilenet_v3_small",
                                           "efficientnet_v2_m", "convnext_base"),
                        default="residual_se",
                        help="final candidate: residual_se; control: simple; "
                             "pretrained: mobilenet_v3_small/efficientnet_v2_m/convnext_base")
    parser.add_argument("--data-dir", default="flower_photos")
    parser.add_argument("--splits-dir", default="splits")
    parser.add_argument("--out-dir", default="runs/cnn_baseline")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--resize-size", type=int, default=256,
                        help="validation resize before center crop; use 288 for 256 input")
    parser.add_argument("--rotation", type=float, default=0.0,
                        help="random rotation degrees (0 disables)")
    parser.add_argument("--crop-scale-min", type=float, default=0.65)
    parser.add_argument("--crop-scale-max", type=float, default=1.0)
    parser.add_argument("--hue", type=float, default=0.05,
                        help="ColorJitter hue range (0 keeps colour intact)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers; 0 is fastest on Windows for this dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="Model EMA decay (e.g. 0.998); 0 disables EMA")
    parser.add_argument("--tta", action="store_true",
                        help="horizontal-flip test-time augmentation")
    parser.add_argument("--no-train-eval", action="store_true",
                        help="skip the clean train-set evaluation each epoch")
    parser.add_argument("--eval-only", action="store_true",
                        help="skip training and re-evaluate the saved best.pt")
    parser.add_argument("--init-from", default=None,
                        help="initialize model weights from a checkpoint before training")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    if args.crop_scale_min <= 0 or args.crop_scale_max < args.crop_scale_min:
        parser.error("--crop-scale-min must be > 0 and <= --crop-scale-max")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    print(f"device: {device}  amp: {amp}")

    out_dir = Path(args.out_dir)
    reports_dir = Path(args.reports_dir)
    fig_dir = reports_dir / "figures"
    table_dir = reports_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = None
    if args.eval_only:
        checkpoint = torch.load(out_dir / "best.pt", map_location=device,
                                weights_only=False)
        # restore the config that produced the checkpoint, not CLI defaults
        for key, value in checkpoint.get("config", {}).items():
            if key not in ("out_dir", "reports_dir", "eval_only", "data_dir",
                           "splits_dir", "num_workers"):
                setattr(args, key, value)
        print(f"eval-only: arch={args.arch}, epochs={args.epochs}")

    loaders = build_dataloaders(
        data_dir=args.data_dir,
        splits_dir=args.splits_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        resize_size=args.resize_size,
        rotation=args.rotation,
        crop_scale=(args.crop_scale_min, args.crop_scale_max),
        hue=args.hue,
    )
    for split, loader in loaders.items():
        print(f"  {split:<5} {len(loader.dataset):>5} images, {len(loader):>3} batches")

    train_eval_loader = None
    if not args.no_train_eval and not args.eval_only:
        train_eval_loader = build_eval_loader(
            Path(args.splits_dir) / "train.csv",
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            resize_size=args.resize_size,
        )
        print(f"  clean train eval: {len(train_eval_loader.dataset)} images "
              "(val transforms, no augmentation)")

    model = build_model(args.arch, num_classes=len(CLASSES)).to(device)
    if args.init_from and not args.eval_only:
        init_ckpt = torch.load(args.init_from, map_location=device,
                               weights_only=False)
        init_arch = init_ckpt.get("config", {}).get("arch")
        if init_arch is not None and init_arch != args.arch:
            raise ValueError(
                f"--init-from arch mismatch: {init_arch} vs --arch {args.arch}"
            )
        model.load_state_dict(init_ckpt["model"])
        print(f"initialized weights from {args.init_from}")
    params = count_parameters(model)
    print(f"model: {type(model).__name__} ({params:,} parameters)")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    scaler = (
        torch.amp.GradScaler("cuda", enabled=amp)
        if device.type == "cuda"
        else None
    )

    if args.eval_only:
        history = read_history(out_dir / "history.csv")
        best_epoch = int(checkpoint["epoch"])
        best_acc = float(checkpoint["val_acc"])
        print(f"eval-only: best epoch {best_epoch}, val acc {best_acc:.4f}")
    else:
        history, best_epoch, best_acc = train_model(
            model, loaders, criterion, optimizer, scheduler, scaler,
            device, amp, args, out_dir, train_eval_loader,
        )

    evaluate_and_export(
        model, loaders, criterion, device, amp, args, history,
        best_epoch, best_acc, params, out_dir, reports_dir,
    )


if __name__ == "__main__":
    main()
