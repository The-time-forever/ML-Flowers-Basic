"""Compare experiment reports produced by train.py + analyze_errors.py.

Example:
    python src/compare_experiments.py `
        reports/cnn_baseline_80 reports/exp_rot20 reports/exp_crop08 reports/exp_hue0
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

CLASSES = ("daisy", "dandelion", "roses", "sunflowers", "tulips")


def load_run(reports_dir: Path) -> dict:
    metrics_path = reports_dir / "tables" / "cnn_baseline_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    cfg = metrics.get("config", {})

    clean_acc = None
    hist_path = Path(cfg.get("out_dir", "")) / "history.csv"
    if hist_path.is_file():
        with hist_path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        best = next(
            (r for r in rows if int(r["epoch"]) == int(metrics["best_epoch"])), None
        )
        if best and best.get("train_eval_acc"):
            clean_acc = float(best["train_eval_acc"])

    val = None
    val_path = reports_dir / "val_errors_metrics.json"
    if val_path.is_file():
        val = json.loads(val_path.read_text(encoding="utf-8"))

    cm = np.array(metrics["confusion_matrix"])
    ri, ti = CLASSES.index("roses"), CLASSES.index("tulips")
    pair = {
        "roses->tulips": int(cm[ri, ti]),
        "tulips->roses": int(cm[ti, ri]),
    }
    val_f1 = {}
    if val:
        pair = val.get("focus_pairs", pair)
        val_f1 = {
            c: val["per_class"][c]["f1-score"] for c in ("roses", "tulips")
        }

    return {
        "name": reports_dir.name,
        "best_epoch": int(metrics["best_epoch"]),
        "val_acc": float(metrics["best_val_acc"]),
        "test_acc": float(metrics["test_acc"]),
        "macro_f1": float(metrics["test_macro_f1"]),
        "clean_train_acc": clean_acc,
        "val_roses_f1": val_f1.get("roses"),
        "val_tulips_f1": val_f1.get("tulips"),
        "roses_to_tulips": pair.get("roses->tulips"),
        "tulips_to_roses": pair.get("tulips->roses"),
        "has_val": val is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare experiment metrics")
    parser.add_argument("reports_dirs", nargs="+")
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    runs = [load_run(Path(d)) for d in args.reports_dirs]
    runs.sort(key=lambda r: r["name"])

    def fmt(v, nd=4):
        return "—" if v is None else f"{v:.{nd}f}"

    header = (
        "| experiment | best ep | val acc | test acc | macro F1 | clean train | "
        "roses F1 (val) | tulips F1 (val) | roses→tulips | tulips→roses |"
    )
    lines = ["# 增强实验对比\n", header, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in runs:
        lines.append(
            f"| {r['name']} | {r['best_epoch']} | {fmt(r['val_acc'])} | "
            f"{fmt(r['test_acc'])} | {fmt(r['macro_f1'])} | "
            f"{fmt(r['clean_train_acc'])} | {fmt(r['val_roses_f1'], 3)} | "
            f"{fmt(r['val_tulips_f1'], 3)} | {r['roses_to_tulips']} | "
            f"{r['tulips_to_roses']} |"
        )
    lines += [
        "",
        "roses/tulips 的 F1 与混淆数来自验证集（`analyze_errors.py`）；"
        "test 指标仅供最终参考。clean train 为最佳 epoch 的无增强训练集准确率。",
        "",
    ]

    out_dir = Path(args.out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "experiments_comparison.md").write_text("\n".join(lines), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(runs))
    width = 0.36

    ax = axes[0]
    ax.bar(x - width / 2, [r["val_acc"] for r in runs], width,
           label="val", color="#4C72B0")
    ax.bar(x + width / 2, [r["test_acc"] for r in runs], width,
           label="test", color="#DD8452")
    ax.set_xticks(x, [r["name"] for r in runs], rotation=20, ha="right")
    ax.set_ylim(0.7, 0.95)
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation / test accuracy")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.bar(x - width / 2, [r["val_roses_f1"] or 0 for r in runs], width,
           label="roses", color="#C94C6D")
    ax.bar(x + width / 2, [r["val_tulips_f1"] or 0 for r in runs], width,
           label="tulips", color="#7A5FB5")
    ax.set_xticks(x, [r["name"] for r in runs], rotation=20, ha="right")
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("F1 (validation)")
    ax.set_title("Hard classes: roses / tulips")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "13_experiment_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n".join(lines))
    print(f"wrote {out_dir / 'experiments_comparison.md'}")
    print(f"wrote {out_dir / 'figures' / '13_experiment_comparison.png'}")


if __name__ == "__main__":
    main()
