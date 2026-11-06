"""Generate numeric tables and image montages for the final report.

The report in ``reports/final_report`` draws its figures with pgfplots/TikZ so
that all curves and bars stay vector graphics.  This script exports the small
``.dat`` tables those plots consume and builds two compact JPEG montages of
real photographs (preprocessing examples and audited label-noise examples).

Usage (from the repository root):
    python src/make_report_figures.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont

CLASSES = ("daisy", "dandelion", "roses", "sunflowers", "tulips")

# (run directory, legend label) for the transfer-learning validation curves.
TRANSFER_CURVES = (
    ("exp_convnext", "convnext224"),
    ("exp_conv_ema", "convnext_ema"),
    ("exp_efficientnet", "efficientnet"),
    ("exp_mobilenet", "mobilenet"),
    ("exp_residual_se", "residual_se"),
)

# (run directory, column name) for the per-class F1 comparison.  All three
# models are evaluated on the original 551-image test split.
PER_CLASS_RUNS = (
    ("cnn_baseline_80", "simple"),
    ("exp_residual_se", "residual_se"),
    ("exp_conv_ema", "convnext"),
)

# Six confirmed exclusions spanning the main label-noise categories.
NOISE_EXAMPLES = (
    ("roses/8462246855_1bdfee7478.jpg", "a"),
    ("roses/17953368844_be3d18cf30_m.jpg", "b"),
    ("roses/2408236801_f43c6bcff2.jpg", "c"),
    ("roses/7211616670_2d49ecb3a5_m.jpg", "d"),
    ("tulips/13513644515_a51470b899.jpg", "e"),
    ("sunflowers/418056361_1dfac1c151_n.jpg", "f"),
)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in ("arialbd.ttf", "arial.ttf"):
        path = Path("C:/Windows/Fonts") / candidate
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _write_rows(path: Path, header: list[str], rows: list[list[object]]) -> None:
    """Write a whitespace-separated table (pgfplots' default table format)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(" ".join(header) + "\n")
        for row in rows:
            fh.write(" ".join(str(value) for value in row) + "\n")
    print(f"wrote {path}")


def _read_history(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def export_scratch_curves(root: Path, out_dir: Path) -> None:
    """Export the 80-epoch from-scratch training curves."""
    rows = _read_history(root / "runs" / "cnn_baseline_80" / "history.csv")
    table: list[list[object]] = []
    for row in rows:
        train_eval = float(row["train_eval_acc"])
        val_acc = float(row["val_acc"])
        table.append([
            int(row["epoch"]),
            float(row["train_loss"]),
            float(row["train_eval_loss"]),
            float(row["val_loss"]),
            100 * float(row["train_acc"]),
            100 * train_eval,
            100 * val_acc,
            100 * (train_eval - val_acc),
        ])
    _write_rows(
        out_dir / "scratch_curves.dat",
        ["epoch", "train_loss", "clean_loss", "val_loss",
         "train_acc", "clean_acc", "val_acc", "gap_pp"],
        table,
    )


def export_transfer_curves(root: Path, out_dir: Path) -> None:
    """Export per-epoch validation accuracy for five representative models."""
    for run_name, column in TRANSFER_CURVES:
        rows = _read_history(root / "runs" / run_name / "history.csv")
        table = [
            [int(row["epoch"]), 100 * float(row["val_acc"])] for row in rows
        ]
        _write_rows(out_dir / f"transfer_{column}.dat", ["epoch", "val_acc"], table)


def export_short_side(root: Path, out_dir: Path) -> None:
    """Export the cumulative distribution of the shorter image side."""
    with (root / "reports" / "tables" / "image_metadata.csv").open(
        newline="", encoding="utf-8"
    ) as fh:
        rows = list(csv.DictReader(fh))
    sides = sorted(min(int(r["width"]), int(r["height"])) for r in rows)
    total = len(sides)
    table: list[list[object]] = []
    for threshold in range(120, 481, 4):
        share = 100 * sum(side >= threshold for side in sides) / total
        table.append([threshold, round(share, 3)])
    _write_rows(out_dir / "short_side_cumulative.dat",
                ["threshold", "cum_pct"], table)


def export_per_class(root: Path, out_dir: Path) -> None:
    """Export test-set F1 per class for three milestone models."""
    scores: dict[str, dict[str, float]] = {}
    for run_name, column in PER_CLASS_RUNS:
        metrics = json.loads(
            (root / "reports" / run_name / "tables" /
             "cnn_baseline_metrics.json").read_text(encoding="utf-8")
        )
        scores[column] = {
            name: 100 * metrics["per_class"][name]["f1-score"] for name in CLASSES
        }
    table = [[name, *(round(scores[col][name], 3) for col in scores)]
             for name in CLASSES]
    _write_rows(out_dir / "per_class_f1.dat",
                ["class", *scores.keys()], table)


def _audit_category(issue: str) -> str:
    text = issue.lower()
    if text.startswith("class mismatch"):
        return "mismatch"
    if ("illustration" in text or "decoration" in text or "light art" in text
            or "non-photographic" in text or text.startswith("irrelevant")):
        return "nonphoto"
    return "scene"


def export_label_audit(root: Path, out_dir: Path) -> None:
    """Aggregate the 57 exclusions into three categories per split."""
    with (root / "reports" / "label_audit" / "cleaning_list.csv").open(
        newline="", encoding="utf-8"
    ) as fh:
        rows = list(csv.DictReader(fh))
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        if row["action"] != "exclude":
            continue
        category = _audit_category(row["issue"])
        split_counts = counts.setdefault(
            category, {split: 0 for split in ("train", "val", "test")}
        )
        split_counts[row["split"]] += 1
    table = [
        [category, split_counts["train"], split_counts["val"],
         split_counts["test"], sum(split_counts.values())]
        for category, split_counts in counts.items()
    ]
    _write_rows(out_dir / "label_audit.dat",
                ["category", "train", "val", "test", "total"], table)


def export_confusion(root: Path, out_dir: Path) -> None:
    """Export the final ensemble confusion matrix with row-normalised shares."""
    metrics = json.loads(
        (root / "reports" / "tables" / "final_ensemble_metrics.json").read_text(
            encoding="utf-8")
    )
    matrix = metrics["confusion_matrix"]
    table = []
    for i, row in enumerate(matrix):
        total = sum(row)
        for j, count in enumerate(row):
            table.append([j, i, count, round(100 * count / total, 2)])
    _write_rows(out_dir / "confusion_final.dat", ["x", "y", "count", "pct"], table)


def _resize_short_side(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    scale = size / min(width, height)
    return image.resize((round(width * scale), round(height * scale)),
                        Image.Resampling.LANCZOS)


def _center_crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def _jitter(image: Image.Image) -> Image.Image:
    """Approximate the training ColorJitter with PIL operations."""
    out = ImageEnhance.Brightness(image).enhance(1.18)
    out = ImageEnhance.Contrast(out).enhance(1.12)
    out = ImageEnhance.Color(out).enhance(1.25)
    hue, saturation, value = out.convert("HSV").split()
    hue = hue.point(lambda v: (v + 13) % 256)
    return Image.merge("HSV", (hue, saturation, value)).convert("RGB")


def _random_resized_crop(image: Image.Image, size: int, rng: random.Random) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    crop_side = round(side * rng.uniform(0.68, 0.8))
    left = rng.randint(0, width - crop_side)
    top = rng.randint(0, height - crop_side)
    return image.crop((left, top, left + crop_side, top + crop_side)).resize(
        (size, size), Image.Resampling.LANCZOS)


def _montage(panels: list[Image.Image], labels: list[str], columns: int,
             path: Path, quality: int = 88) -> None:
    """Write a labelled grid of equally sized panels as a compact JPEG."""
    cell = panels[0].width
    rows = (len(panels) + columns - 1) // columns
    gap, margin, strip = 8, 10, 30
    width = margin * 2 + columns * cell + (columns - 1) * gap
    height = margin * 2 + rows * (cell + strip) + (rows - 1) * gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font(20)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        row, col = divmod(index, columns)
        x = margin + col * (cell + gap)
        y = margin + row * (cell + strip + gap)
        canvas.paste(panel, (x, y))
        draw.text((x + 4, y + cell + 4), label, fill=(35, 78, 112), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=quality, optimize=True)
    print(f"wrote {path} ({canvas.width}x{canvas.height})")


def build_preprocess_montage(root: Path, out_dir: Path) -> None:
    """Show the deterministic crop plus three training augmentations."""
    source = root / "flower_photos" / "tulips" / "100930342_92e8746431_n.jpg"
    with Image.open(source) as handle:
        base = _resize_short_side(handle.convert("RGB"), 256)
    size = 224
    center = base.crop(((base.width - size) // 2, (base.height - size) // 2,
                        (base.width + size) // 2, (base.height + size) // 2))
    rng = random.Random(42)
    panels = [
        center,
        _random_resized_crop(base, size, rng),
        center.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        _jitter(center),
    ]
    _montage(panels, ["(a)", "(b)", "(c)", "(d)"], 4,
             out_dir / "preprocess_examples.jpg")


def build_noise_montage(root: Path, out_dir: Path) -> None:
    """Show six confirmed label-noise examples from the audit."""
    panels, labels = [], []
    for relative, label in NOISE_EXAMPLES:
        with Image.open(root / "flower_photos" / relative) as handle:
            image = _center_crop_square(handle.convert("RGB"))
        panels.append(image.resize((300, 300), Image.Resampling.LANCZOS))
        labels.append(f"({label})")
    _montage(panels, labels, 3, out_dir / "label_noise_examples.jpg", quality=85)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export report figure data and montages")
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument("--out-dir", default="reports/final_report/figures")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    export_scratch_curves(root, out_dir)
    export_transfer_curves(root, out_dir)
    export_short_side(root, out_dir)
    export_per_class(root, out_dir)
    export_label_audit(root, out_dir)
    export_confusion(root, out_dir)
    build_preprocess_montage(root, out_dir)
    build_noise_montage(root, out_dir)


if __name__ == "__main__":
    main()
