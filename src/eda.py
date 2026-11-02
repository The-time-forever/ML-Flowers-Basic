"""EDA for the flower_photos dataset.

Scans every image, extracts metadata and simple image statistics, then writes
tables, figures and a Markdown report under ``reports/``.

Usage:
    python src/eda.py
    python src/eda.py --data-dir flower_photos --out-dir reports
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import rgb_to_hsv
from PIL import Image, ImageOps

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
THUMB = 64          # size used for colour / brightness statistics
SAMPLE_SIZE = 160   # size used for the sample grid
DHASH_MAX_DIST = 4  # Hamming distance for "near duplicate"
RNG_SEED = 42

CLASS_COLORS = {
    "daisy": "#4C9F70",
    "dandelion": "#E8B33C",
    "roses": "#C94C6D",
    "sunflowers": "#E07B27",
    "tulips": "#7A5FB5",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def dhash(img: Image.Image, hash_size: int = 8) -> int:
    """Difference hash (64-bit) used for near-duplicate detection."""
    small = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    px = np.asarray(small, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def file_md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def hue_histogram(rgb: np.ndarray, bins: int = 12) -> np.ndarray:
    """Saturation-weighted hue histogram of an RGB uint8 thumbnail."""
    hsv = rgb_to_hsv(rgb.astype(np.float32) / 255.0)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (sat > 0.15) & (val > 0.15)
    hist, _ = np.histogram(
        hue[mask], bins=np.linspace(0.0, 1.0, bins + 1), weights=sat[mask]
    )
    total = hist.sum()
    return hist / total if total > 0 else hist


def scan(data_dir: Path) -> tuple[list[dict], list[str], dict[str, list[np.ndarray]]]:
    """Return (records, errors, class -> thumbnails) after scanning all images."""
    records: list[dict] = []
    errors: list[str] = []
    class_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
    thumbs: dict[str, list[np.ndarray]] = defaultdict(list)

    for class_dir in class_dirs:
        label = class_dir.name
        files = sorted(
            p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS
        )
        print(f"  scanning {label:<12} {len(files):>5} images", flush=True)
        for path in files:
            try:
                with Image.open(path) as im:
                    im.load()
                    fmt, mode = im.format, im.mode
                    width, height = im.size
                    thumb = im.convert("RGB").resize((THUMB, THUMB), Image.LANCZOS)
                    sample = ImageOps.fit(
                        im.convert("RGB"), (SAMPLE_SIZE, SAMPLE_SIZE), Image.LANCZOS
                    )
            except Exception as exc:  # corrupt / unreadable file
                errors.append(f"{path.relative_to(data_dir)} :: {exc}")
                continue

            rgb = np.asarray(thumb, dtype=np.uint8)
            gray = np.asarray(thumb.convert("L"), dtype=np.float32)
            hsv = rgb_to_hsv(rgb.astype(np.float32) / 255.0)
            thumbs[label].append(np.asarray(sample, dtype=np.uint8))

            records.append(
                {
                    "path": str(path.relative_to(data_dir)).replace("\\", "/"),
                    "label": label,
                    "filename": path.name,
                    "format": fmt,
                    "mode": mode,
                    "width": width,
                    "height": height,
                    "aspect_ratio": round(width / height, 4),
                    "megapixels": round(width * height / 1e6, 4),
                    "file_size_kb": round(path.stat().st_size / 1024, 1),
                    "brightness": round(float(hsv[..., 2].mean() * 255), 2),
                    "contrast": round(float(gray.std()), 2),
                    "saturation": round(float(hsv[..., 1].mean()), 4),
                    "r_mean": round(float(rgb[..., 0].mean()), 2),
                    "g_mean": round(float(rgb[..., 1].mean()), 2),
                    "b_mean": round(float(rgb[..., 2].mean()), 2),
                    "md5": file_md5(path),
                    "dhash": dhash(im),
                    "_hue_hist": hue_histogram(rgb),
                }
            )
    return records, errors, thumbs


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def summarise(records: list[dict], labels: list[str]) -> dict:
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_class[r["label"]].append(r)

    def col(rows, key):
        return np.array([r[key] for r in rows], dtype=float)

    per_class = {}
    for label in labels:
        rows = by_class[label]
        if not rows:
            continue
        w, h = col(rows, "width"), col(rows, "height")
        min_side = np.minimum(w, h)
        per_class[label] = {
            "count": len(rows),
            "width_mean": w.mean(), "width_std": w.std(),
            "width_min": w.min(), "width_max": w.max(),
            "height_mean": h.mean(), "height_std": h.std(),
            "height_min": h.min(), "height_max": h.max(),
            "aspect_mean": col(rows, "aspect_ratio").mean(),
            "aspect_std": col(rows, "aspect_ratio").std(),
            "aspect_min": col(rows, "aspect_ratio").min(),
            "aspect_max": col(rows, "aspect_ratio").max(),
            "mp_mean": col(rows, "megapixels").mean(),
            "brightness_mean": col(rows, "brightness").mean(),
            "brightness_std": col(rows, "brightness").std(),
            "saturation_mean": col(rows, "saturation").mean(),
            "contrast_mean": col(rows, "contrast").mean(),
            "r_mean": col(rows, "r_mean").mean(),
            "g_mean": col(rows, "g_mean").mean(),
            "b_mean": col(rows, "b_mean").mean(),
            "file_size_mean": col(rows, "file_size_kb").mean(),
            "pct_min_side_224": 100.0 * (min_side >= 224).mean(),
            "pct_square_5pct": 100.0 * (np.abs(col(rows, "aspect_ratio") - 1) <= 0.05).mean(),
        }

    counts = np.array([len(by_class[l]) for l in labels], dtype=float)
    return {
        "by_class": by_class,
        "per_class": per_class,
        "total": len(records),
        "counts": counts,
        "imbalance_ratio": counts.max() / counts.min(),
    }


def duplicate_analysis(records: list[dict]) -> dict:
    """Exact (md5) and perceptual (dhash) duplicate detection."""
    md5_groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        md5_groups[r["md5"]].append(i)
    exact_groups = [g for g in md5_groups.values() if len(g) > 1]

    hashes = np.array([r["dhash"] for r in records], dtype=np.uint64)
    n = len(hashes)
    pairs = []
    block = 512
    for start in range(0, n, block):
        stop = min(start + block, n)
        xor = hashes[start:stop, None] ^ hashes[None, :]
        dist = np.bitwise_count(xor)
        for i_local, j in zip(*np.nonzero(np.triu(dist <= DHASH_MAX_DIST, k=0))):
            i = start + int(i_local)
            j = int(j)
            if i == j or j < i:
                continue
            pairs.append((i, j, int(dist[i_local, j])))

    # connected components of the near-duplicate graph
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j, _ in pairs:
        union(i, j)
    comps: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        comps[find(i)].append(i)
    clusters = [c for c in comps.values() if len(c) > 1]

    cross_class_pairs = [(i, j, d) for i, j, d in pairs if records[i]["label"] != records[j]["label"]]
    exact_cross = [
        g for g in exact_groups if len({records[i]["label"] for i in g}) > 1
    ]
    return {
        "exact_groups": exact_groups,
        "exact_cross_class": exact_cross,
        "near_pairs": pairs,
        "clusters": clusters,
        "cross_class_pairs": cross_class_pairs,
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def plot_class_distribution(rows, labels, path: Path) -> None:
    counts = [len(rows[l]) for l in labels]
    total = sum(counts)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, counts, color=[CLASS_COLORS.get(l, "#888") for l in labels])
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, c + 8,
                f"{c}\n({c / total:.1%})", ha="center", va="bottom", fontsize=10)
    ax.axhline(total / len(labels), ls="--", c="gray", lw=1, label=f"Ideal balance ({total / len(labels):.0f})")
    ax.set_ylabel("Number of images")
    ax.set_title(f"Class distribution (total = {total})")
    ax.set_ylim(0, max(counts) * 1.18)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, path)


def plot_dimensions(recs, labels, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    for label in labels:
        sub = [r for r in recs if r["label"] == label]
        ax.scatter([r["width"] for r in sub], [r["height"] for r in sub],
                   s=9, alpha=0.45, label=label, color=CLASS_COLORS.get(label))
    lim = max(max(r["width"] for r in recs), max(r["height"] for r in recs)) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="square")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("Image dimensions")
    ax.legend(markerscale=2, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    min_sides = np.array([min(r["width"], r["height"]) for r in recs])
    bins = np.geomspace(max(16, min_sides.min()), min_sides.max(), 40)
    ax.hist(min_sides, bins=bins, color="#4C72B0", alpha=0.85)
    ax.set_xscale("log")
    ax.axvline(224, color="crimson", ls="--", lw=1.5, label="224 px (typical CNN input)")
    ax.set_xlabel("Shorter side (px, log scale)")
    ax.set_ylabel("Number of images")
    ax.set_title("Short-side distribution")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, path)


def plot_aspect(recs, labels, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.3))
    data = [[r["aspect_ratio"] for r in recs if r["label"] == l] for l in labels]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", markersize=3, alpha=0.4))
    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor(CLASS_COLORS.get(label, "#888"))
        patch.set_alpha(0.75)
    ax.axhline(1.0, ls="--", c="gray", lw=1)
    ax.set_ylabel("Width / height")
    ax.set_title("Aspect ratio by class")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, path)


def plot_brightness(recs, labels, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    specs = [
        ("brightness", "Brightness (0-255)"),
        ("contrast", "Grayscale contrast (std)"),
        ("saturation", "Mean saturation (0-1)"),
    ]
    for ax, (key, title) in zip(axes, specs):
        data = [[r[key] for r in recs if r["label"] == l] for l in labels]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                        flierprops=dict(marker="o", markersize=2.5, alpha=0.35))
        for patch, label in zip(bp["boxes"], labels):
            patch.set_facecolor(CLASS_COLORS.get(label, "#888"))
            patch.set_alpha(0.75)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, path)


def plot_colour(recs, labels, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]
    x = np.arange(len(labels))
    width = 0.26
    for offset, (key, colour, name) in zip(
        (-width, 0, width),
        (("r_mean", "#D1495B", "R"), ("g_mean", "#3F8F5B", "G"), ("b_mean", "#3A6EA5", "B")),
    ):
        vals = [np.mean([r[key] for r in recs if r["label"] == l]) for l in labels]
        ax.bar(x + offset, vals, width, color=colour, label=name)
    ax.set_xticks(x, labels, rotation=20)
    ax.set_ylim(0, 255)
    ax.set_ylabel("Mean channel value")
    ax.set_title("Mean RGB per class")
    ax.legend(title="channel")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    centers = (np.arange(12) + 0.5) / 12 * 360
    for label in labels:
        hists = np.array([r["_hue_hist"] for r in recs if r["label"] == label])
        if len(hists) == 0:
            continue
        # roll bins so the red region (around 0/360) is contiguous
        mean_hist = np.roll(hists.mean(axis=0), 1)
        ax.plot(centers - 30, mean_hist, lw=2, label=label,
                color=CLASS_COLORS.get(label))
    ax.set_xlim(-30, 330)
    ax.set_xticks(np.arange(0, 301, 60))
    ax.set_xlabel("Hue (degrees; red sits at 0 / 360)")
    ax.set_ylabel("Saturation-weighted share")
    ax.set_title("Hue profile (saturated pixels only)")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, path)


def plot_mean_images(recs, thumbs, labels, path: Path) -> None:
    fig, axes = plt.subplots(1, len(labels), figsize=(3 * len(labels), 3.9))
    for ax, label in zip(np.atleast_1d(axes), labels):
        arr = np.asarray(thumbs[label], dtype=np.float32) / 255.0
        mean = arr.mean(axis=0)
        ax.imshow(np.clip(mean, 0, 1))
        ax.set_title(f"{label} (n={len(arr)})", fontsize=10)
        ax.axis("off")
    fig.suptitle("Average image per class (shows background / framing bias)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, path)


def plot_samples(recs, labels, data_dir: Path, path: Path, per_class: int = 5) -> None:
    rng = random.Random(RNG_SEED)
    fig, axes = plt.subplots(len(labels), per_class,
                             figsize=(per_class * 2.4, len(labels) * 2.5))
    for row, label in enumerate(labels):
        files = sorted(data_dir.glob(f"{label}/*"))
        files = [p for p in files if p.suffix.lower() in IMG_EXTS]
        chosen = rng.sample(files, min(per_class, len(files)))
        for col, fp in enumerate(chosen):
            ax = axes[row, col]
            with Image.open(fp) as im:
                ax.imshow(ImageOps.fit(im.convert("RGB"), (SAMPLE_SIZE, SAMPLE_SIZE)))
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(label, fontsize=10)
        if len(chosen) < per_class:
            for col in range(len(chosen), per_class):
                axes[row, col].axis("off")
    fig.suptitle("Random samples per class (all images center-cropped to a square)", fontsize=13)
    fig.tight_layout()
    save(fig, path)


def plot_duplicates(recs, dup, data_dir: Path, path: Path, max_pairs: int = 6) -> bool:
    pairs = sorted(dup["near_pairs"], key=lambda t: t[2])[:max_pairs]
    if not pairs:
        return False
    fig, axes = plt.subplots(len(pairs), 2, figsize=(7.2, 3.05 * len(pairs)))
    axes = np.atleast_2d(axes)
    for row, (i, j, dist) in enumerate(pairs):
        for col, idx in enumerate((i, j)):
            ax = axes[row, col]
            with Image.open(data_dir / recs[idx]["path"]) as im:
                ax.imshow(ImageOps.fit(im.convert("RGB"), (SAMPLE_SIZE, SAMPLE_SIZE)))
            ax.axis("off")
            name = Path(recs[idx]["path"]).name
            ax.set_title(f"{recs[idx]['label']}  ·  {name[:28]}", fontsize=7.5)
        axes[row, 0].annotate(
            f"d={dist}", xy=(1.005, 0.5), xycoords="axes fraction",
            ha="left", va="center", fontsize=9, fontweight="bold",
        )
    fig.suptitle("Closest near-duplicate pairs (dHash Hamming distance)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.975), w_pad=2.4)
    save(fig, path)
    return True


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def fmt(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}"


def write_tables(records, per_class, labels, table_dir: Path) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)

    fields = [k for k in records[0] if not k.startswith("_")]
    with (table_dir / "image_metadata.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    with (table_dir / "class_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        cols = ["label", *list(next(iter(per_class.values())).keys())]
        writer = csv.writer(fh)
        writer.writerow(cols)
        for label in labels:
            writer.writerow([label, *[round(per_class[label][c], 4) for c in cols[1:]]])


def write_report(recs, errors, labels, per_class, info, dup, fig_dir, report_path) -> None:
    fig_dir = Path(fig_dir)
    fig_rel = fig_dir.name
    total = info["total"]
    counts = info["counts"]
    ratio = info["imbalance_ratio"]
    largest = labels[int(np.argmax(counts))]
    smallest = labels[int(np.argmin(counts))]
    widths = np.array([r["width"] for r in recs])
    heights = np.array([r["height"] for r in recs])
    min_sides = np.minimum(widths, heights)
    aspects = np.array([r["aspect_ratio"] for r in recs])
    brightness = np.array([r["brightness"] for r in recs])
    sizes_kb = np.array([r["file_size_kb"] for r in recs])

    near_pairs = dup["near_pairs"]
    clusters = dup["clusters"]
    exact_groups = dup["exact_groups"]
    cluster_sizes = Counter(len(c) for c in clusters)
    dup_total = sum(len(c) for c in clusters)
    within = sum(1 for i, j, _ in near_pairs if recs[i]["label"] == recs[j]["label"])

    split = {name: {l: round(len([r for r in recs if r["label"] == l]) * frac)
                    for l in labels}
             for name, frac in (("train (70%)", 0.70), ("val (15%)", 0.15), ("test (15%)", 0.15))}

    lines = []
    a = lines.append
    a("# 花卉数据集 EDA 报告\n")
    a(f"- 数据目录：`flower_photos/`")
    a(f"- 有效图片：**{total}** 张，类别数：**{len(labels)}**")
    a(f"- 损坏/无法读取：**{len(errors)}** 张")
    a(f"- 类别：{', '.join(f'`{l}`' for l in labels)}\n")

    a("## 1. 数据集概况\n")
    a("| 类别 | 图片数 | 占比 | 平均宽×高 | 宽高比均值 | 最小边≥224px | 平均亮度 | 平均饱和度 |")
    a("|---|---:|---:|---|---:|---:|---:|---:|")
    for label in labels:
        s = per_class[label]
        a(f"| {label} | {s['count']} | {s['count']/total:.1%} | "
          f"{s['width_mean']:.0f}×{s['height_mean']:.0f} | {s['aspect_mean']:.2f} | "
          f"{s['pct_min_side_224']:.1f}% | {s['brightness_mean']:.0f} | {s['saturation_mean']:.2f} |")
    a(f"| **合计** | **{total}** | **100%** | — | — | "
      f"**{100*(min_sides>=224).mean():.1f}%** | — | — |\n")
    a(f"类别不平衡比（最多/最少）为 **{ratio:.2f}**（{largest} {int(counts.max())} 张 vs "
      f"{smallest} {int(counts.min())} 张）。")
    a("这个比例属于**轻度不平衡**，通常不需要过采样；使用分层划分 + `class_weight` "
      "或按类别均衡采样即可。\n")

    a("## 2. 类别分布\n")
    a(f"![类别分布]({fig_rel}/01_class_distribution.png)\n")
    a("```")
    for label in labels:
        n = int(counts[list(labels).index(label)])
        a(f"{label:<12} {n:>4}  {'█' * round(n / counts.max() * 40)}")
    a("```\n")

    a("## 3. 图像尺寸与宽高比\n")
    a(f"- 宽度范围：{widths.min()}–{widths.max()} px（中位数 {np.median(widths):.0f}）")
    a(f"- 高度范围：{heights.min()}–{heights.max()} px（中位数 {np.median(heights):.0f}）")
    a(f"- 宽高比：{aspects.min():.2f}–{aspects.max():.2f}，"
      f"其中 {100*(np.abs(aspects-1)<=0.05).mean():.1f}% 近似正方形，"
      f"{100*(min_sides>=224).mean():.1f}% 的短边 ≥ 224 px")
    a(f"- 文件大小：中位数 {np.median(sizes_kb):.0f} KB，最大 {sizes_kb.max():.0f} KB")
    a(f"- 低于 100 px 的极小图：{int((min_sides < 100).sum())} 张")
    res_counts = Counter((r["width"], r["height"]) for r in recs)
    a("- 最常见分辨率：" + "，".join(
        f"{w}×{h}（{n} 张）" for (w, h), n in res_counts.most_common(5)))
    a("  尺寸呈明显的离散簇，符合 Flickr 不同缩略图规格（500/320/240 px）的特征，"
      "而非连续分布。\n")
    a(f"![尺寸分布]({fig_rel}/02_image_dimensions.png)\n")
    a(f"![宽高比]({fig_rel}/03_aspect_ratio.png)\n")
    a("**含义**：尺寸和宽高比差异很大，训练时不能直接 resize 到网络输入而不考虑形变；"
      "建议采用 `Resize(256) → CenterCrop(224)`（或随机裁剪）而非直接拉伸，"
      "并保留原始宽高比信息以检查数据泄漏式偏差。\n")

    a("## 4. 颜色与光照\n")
    a(f"![亮度/对比度/饱和度]({fig_rel}/04_brightness_contrast.png)\n")
    a(f"![颜色分布]({fig_rel}/05_color_profile.png)\n")
    b_lo = min(per_class[l]["brightness_mean"] for l in labels)
    b_hi = max(per_class[l]["brightness_mean"] for l in labels)
    a(f"各类别平均亮度在 {b_lo:.0f}–{b_hi:.0f}（0–255）之间，差异有限；"
      "但**色相分布差异明显**：`dandelion`/`sunflowers` 集中在黄橙色相，"
      "`roses` 偏红/品红，`daisy` 与 `tulips` 分布更分散。")
    a("这意味着仅用颜色直方图就能获得相当高的基线准确率；同时也要警惕模型学到"
      "「背景颜色/光照」而不是花朵形状，建议训练时使用颜色增强（jitter）并检查 Grad-CAM。\n")
    a(f"![各类平均图]({fig_rel}/06_class_mean_images.png)\n")
    a("平均图显示各类存在明显的构图/背景差异（中心花朵、绿色背景占比不同），"
      "属于数据集中常见的**背景偏置**。\n")

    a("## 5. 样本示例\n")
    a(f"![样本网格]({fig_rel}/07_sample_grid.png)\n")

    a("## 6. 重复与近重复检查\n")
    a(f"- 完全重复（文件 MD5 相同）：{len(exact_groups)} 组，涉及 "
      f"{sum(len(g) for g in exact_groups)} 张；跨类别完全重复：{len(dup['exact_cross_class'])} 组")
    a(f"- 近重复（dHash 汉明距离 ≤ {DHASH_MAX_DIST}）：{len(near_pairs)} 对，"
      f"构成 {len(clusters)} 个簇，涉及 {dup_total} 张（占 {dup_total/total:.1%}）")
    a(f"  - 其中同类别 {within} 对，跨类别 {len(dup['cross_class_pairs'])} 对")
    a(f"  - 簇大小分布：{dict(sorted(cluster_sizes.items()))}")
    if dup["exact_cross_class"]:
        a("\n**跨类别完全重复（同一张图片被放进两个类别，属于标签冲突）**：")
        for group in dup["exact_cross_class"]:
            entries = "; ".join(f"`{recs[i]['path']}`" for i in group)
            a(f"- {entries}")
    if dup_total:
        a("\n近重复会造成**训练/验证泄漏**，使验证指标虚高。建议在划分数据集前按 dHash 聚类，"
          "把同一簇的图片整体分到同一 split，或直接剔除冗余图片。")
    if (fig_dir / "08_duplicates.png").exists():
        a(f"\n![近重复示例]({fig_rel}/08_duplicates.png)")
    a("")

    a("## 7. 数据质量检查\n")
    a(f"- 损坏文件：{len(errors)}")
    for e in errors[:10]:
        a(f"  - `{e}`")
    modes = Counter(r["mode"] for r in recs)
    formats = Counter(r["format"] for r in recs)
    a(f"- 色彩模式：{dict(modes)}")
    a(f"- 文件格式：{dict(formats)}")
    odd_ext = []
    ext_norm = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "bmp": "bmp",
                "gif": "gif", "webp": "webp"}
    for r in recs:
        ext = Path(r["path"]).suffix.lower().lstrip(".")
        fmt = (r["format"] or "").lower()
        if ext_norm.get(ext, ext) != ext_norm.get(fmt, fmt):
            odd_ext.append(r["path"])
    a(f"- 扩展名与真实格式不一致：{len(odd_ext)} 张"
      + (f"（如 `{odd_ext[0]}`）" if odd_ext else ""))
    a(f"- 灰度/非 RGB 图片：{sum(1 for r in recs if r['mode'] not in ('RGB',))} 张（已统一转为 RGB）\n")

    a("## 8. 建模建议\n")
    a(f"1. **划分**：分层 70/15/15（train/val/test），随机种子固定；"
      f"约 {round(total*0.7)} / {round(total*0.15)} / {round(total*0.15)} 张。")
    a("2. **去重后再划分**：近重复簇整体归入同一 split，避免验证集泄漏。")
    a("3. **输入管线**：`Resize(256) → RandomResizedCrop(224) → RandomHorizontalFlip`，"
      "ImageNet 归一化；测试时 `Resize(256) → CenterCrop(224)`。")
    a(f"4. **分辨率**：{100*(min_sides>=224).mean():.1f}% 的图片短边 ≥ 224 px，"
      "先下采样到 224 不会损失太多信息；若追求更高精度可用 300–384 px。")
    a("5. **数据增强**：颜色抖动、随机裁剪、水平翻转、轻微旋转；避免过强的垂直翻转（花卉有朝向语义）。")
    a("6. **不平衡**：轻度不平衡，用分层采样即可；可对比 `class_weight` 与普通训练的效果。")
    a("7. **基线顺序**：颜色直方图 + Logistic Regression → 小型 CNN → ImageNet 预训练"
      "（ResNet18/EfficientNet-B0）微调；每步都记录混淆矩阵，重点看 daisy/tulips 与 "
      "dandelion/sunflowers 的混淆。")
    a("8. **可解释性**：用 Grad-CAM 检查模型是否聚焦花朵而非背景；若背景偏置严重，"
      "可尝试 RandomErasing / 更强裁剪。\n")

    a("## 9. 产物\n")
    a("- `reports/tables/image_metadata.csv`：每张图片的尺寸、颜色、亮度、哈希等元数据")
    a("- `reports/tables/class_summary.csv`：按类别聚合的统计量")
    a(f"- `reports/figures/*.png`：{len(list(fig_dir.glob('*.png')))} 张分析图")
    a("- `src/eda.py`：可复现的 EDA 脚本\n")

    report_path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="EDA for the flower_photos dataset")
    parser.add_argument("--data-dir", default="flower_photos")
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {data_dir} ...")
    records, errors, thumbs = scan(data_dir)
    labels = sorted({r["label"] for r in records})
    info = summarise(records, labels)
    per_class = info["per_class"]

    print("Detecting duplicates ...")
    dup = duplicate_analysis(records)

    print("Writing tables ...")
    write_tables(records, per_class, labels, table_dir)

    print("Plotting ...")
    plot_class_distribution(info["by_class"], labels, fig_dir / "01_class_distribution.png")
    plot_dimensions(records, labels, fig_dir / "02_image_dimensions.png")
    plot_aspect(records, labels, fig_dir / "03_aspect_ratio.png")
    plot_brightness(records, labels, fig_dir / "04_brightness_contrast.png")
    plot_colour(records, labels, fig_dir / "05_color_profile.png")
    plot_mean_images(records, thumbs, labels, fig_dir / "06_class_mean_images.png")
    plot_samples(records, labels, data_dir, fig_dir / "07_sample_grid.png")
    plot_duplicates(records, dup, data_dir, fig_dir / "08_duplicates.png")

    report_path = out_dir / "eda_report.md"
    write_report(records, errors, labels, per_class, info, dup, fig_dir, report_path)
    print(f"  wrote {report_path}")

    counts = info["counts"]
    min_sides = np.array([min(r["width"], r["height"]) for r in records])
    print("\n=== Summary ===")
    print(f"images      : {info['total']} ({len(errors)} unreadable)")
    print(f"classes     : {len(labels)} -> {dict(zip(labels, counts.astype(int)))}")
    print(f"imbalance   : {info['imbalance_ratio']:.2f}")
    print(f"short side  : min={min_sides.min()}, median={np.median(min_sides):.0f}, "
          f"max={min_sides.max()}; >=224px: {100*(min_sides>=224).mean():.1f}%")
    print(f"duplicates  : exact={len(dup['exact_groups'])} groups, "
          f"near={len(dup['near_pairs'])} pairs / {len(dup['clusters'])} clusters "
          f"({sum(len(c) for c in dup['clusters'])} images)")
    print(f"report      : {report_path}")


if __name__ == "__main__":
    main()
