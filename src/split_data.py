"""Build a dedup-aware, stratified train/val/test split for flower_photos.

Images that are exact or near duplicates (dHash Hamming distance <= threshold)
are grouped into clusters, and every cluster is assigned to exactly one split.
This prevents the same photo (or a re-encoded/resized copy) from leaking into
both training and validation/test sets.

Exact duplicates are removed (one copy per MD5 group), and cross-class
duplicates are dropped because they carry conflicting labels.
Near duplicates are never discarded: they stay grouped so that a whole cluster
lands in one split.

Outputs (under ``splits/`` by default):
    train.csv, val.csv, test.csv   manifests: path,label,split,cluster_id,cluster_size
    summary.json                   counts + configuration
    split_report.md                human-readable report

Usage:
    python src/split_data.py
    python src/split_data.py --seed 42 --val-frac 0.15 --test-frac 0.15
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

try:  # `python src/split_data.py`
    from eda import IMG_EXTS, dhash
except ImportError:  # `python -m src.split_data`
    from .eda import IMG_EXTS, dhash

SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# Dataset scanning and duplicate clustering
# --------------------------------------------------------------------------- #
def scan(data_dir: Path) -> list[dict]:
    records = []
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        label = class_dir.name
        files = sorted(
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        )
        print(f"  hashing {label:<12} {len(files):>5} images", flush=True)
        for path in files:
            try:
                with Image.open(path) as im:
                    im.load()
                    h = dhash(im)
                    width, height = im.size
            except Exception as exc:
                raise RuntimeError(f"cannot read {path}: {exc}") from exc
            records.append({
                "path": str(path.relative_to(data_dir)).replace("\\", "/"),
                "label": label,
                "width": width,
                "height": height,
                "bytes": path.stat().st_size,
                "dhash": h,
                "md5": hashlib.md5(path.read_bytes()).hexdigest(),
            })
    return records


def build_clusters(records: list[dict], max_dist: int) -> list[dict]:
    """Union-find over dHash pairs, returns deterministic cluster list."""
    n = len(records)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    hashes = [r["dhash"] for r in records]
    for i in range(n):
        hi = hashes[i]
        for j in range(i + 1, n):
            if (hi ^ hashes[j]).bit_count() <= max_dist:
                union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    clusters = []
    for members in sorted(groups.values(), key=lambda m: m[0]):
        labels = Counter(records[i]["label"] for i in members)
        cluster = {
            "id": len(clusters),
            "members": members,
            "size": len(members),
            "label": labels.most_common(1)[0][0],
            "labels": sorted(labels),
        }
        clusters.append(cluster)
    return clusters


def find_conflicts(records: list[dict], clusters: list[dict]) -> dict:
    """Exact duplicates and cross-class clusters that are unsafe for training."""
    md5_groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        md5_groups[r["md5"]].append(i)
    exact_groups = [g for g in md5_groups.values() if len(g) > 1]
    exact_cross = [
        g for g in exact_groups
        if len({records[i]["label"] for i in g}) > 1
    ]
    mixed_clusters = [c for c in clusters if len(c["labels"]) > 1]
    return {
        "exact_groups": exact_groups,
        "exact_cross": exact_cross,
        "mixed_clusters": mixed_clusters,
    }


def select_drops(
    records: list[dict],
    clusters: list[dict],
    conflicts: dict,
) -> dict[int, str]:
    """Decide which image indices to exclude, with a human-readable reason.

    - cross-class exact duplicates (label conflicts) are removed entirely
    - exact duplicates (same MD5) keep a single copy
    - near duplicates are never removed, only kept in the same cluster
    """
    drop: dict[int, str] = {}

    conflict_indices: set[int] = set()
    for group in conflicts["exact_cross"]:
        conflict_indices.update(group)
    for c in clusters:
        if conflict_indices.intersection(c["members"]):
            for i in c["members"]:
                drop[i] = "cross-class duplicate (label conflict)"

    def quality(i: int) -> tuple[int, int, int]:
        r = records[i]
        return (r["width"] * r["height"], r["bytes"], -i)

    for group in conflicts["exact_groups"]:
        alive = [i for i in group if i not in drop]
        if len(alive) < 2:
            continue
        if len({records[i]["label"] for i in alive}) > 1:
            continue  # handled as a label conflict above
        keep = max(alive, key=quality)
        for i in alive:
            if i != keep:
                drop[i] = f"exact duplicate of `{records[keep]['path']}`"
    return drop


def remap(
    records: list[dict], clusters: list[dict], drop: dict[int, str]
) -> tuple[list[dict], list[dict]]:
    """Remove dropped images and rebuild clusters with contiguous ids."""
    kept_indices = [i for i in range(len(records)) if i not in drop]
    index_map = {old: new for new, old in enumerate(kept_indices)}
    kept_records = [records[i] for i in kept_indices]

    kept_clusters = []
    for c in clusters:
        members = [index_map[i] for i in c["members"] if i in index_map]
        if not members:
            continue
        labels = Counter(kept_records[i]["label"] for i in members)
        kept_clusters.append({
            "id": len(kept_clusters),
            "members": members,
            "size": len(members),
            "label": labels.most_common(1)[0][0],
            "labels": sorted(labels),
        })
    return kept_records, kept_clusters


# --------------------------------------------------------------------------- #
# Stratified group split
# --------------------------------------------------------------------------- #
def target_sizes(n: int, val_frac: float, test_frac: float) -> dict[str, int]:
    n_val = round(n * val_frac)
    n_test = round(n * test_frac)
    return {"train": n - n_val - n_test, "val": n_val, "test": n_test}


def split_clusters(
    clusters: list[dict],
    labels: list[str],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, list[dict]]:
    """Assign whole clusters to splits, stratified by class and size-balanced."""
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for c in clusters:
        by_label[c["label"]].append(c)

    assignment: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for label in labels:
        items = by_label[label][:]
        rng.shuffle(items)
        # large clusters first so they fit before capacity is fragmented
        items.sort(key=lambda c: -c["size"])

        n = sum(c["size"] for c in items)
        targets = target_sizes(n, val_frac, test_frac)
        remaining = dict(targets)

        for c in items:
            # prefer the split with the largest remaining capacity; ties are
            # broken by the fixed SPLITS order for determinism
            split = max(SPLITS, key=lambda s: (remaining[s], -SPLITS.index(s)))
            assignment[split].append(c)
            remaining[split] -= c["size"]
    return assignment


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def write_manifest(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["path", "label", "split", "cluster_id", "cluster_size"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    rows: list[dict],
    labels: list[str],
    assignment: dict[str, list[dict]],
    clusters: list[dict],
    conflicts: dict,
    conflict_paths: list[list[str]],
    dropped: list[dict],
    images_available: int,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict:
    per_class = {
        label: {s: 0 for s in SPLITS} for label in labels
    }
    for r in rows:
        per_class[r["label"]][r["split"]] += 1

    totals = {s: len([r for r in rows if r["split"] == s]) for s in SPLITS}
    n_total = len(rows)

    split_to_clusters = defaultdict(set)
    for s, cs in assignment.items():
        for c in cs:
            split_to_clusters[c["id"]].add(s)
    leaking = [cid for cid, ss in split_to_clusters.items() if len(ss) > 1]
    multi = [c for c in clusters if c["size"] > 1]

    n_conflict_drop = sum(1 for d in dropped if "label conflict" in d["reason"])
    n_exact_drop = sum(1 for d in dropped if d["reason"].startswith("exact duplicate"))

    summary = {
        "seed": args.seed,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "near_dup_distance": args.near_dup_dist,
        "images_used": n_total,
        "images_available": images_available,
        "images_dropped": len(dropped),
        "images_dropped_conflict": n_conflict_drop,
        "images_dropped_exact": n_exact_drop,
        "clusters_total": len(clusters),
        "clusters_multi_image": len(multi),
        "clusters_spanning_splits": len(leaking),
        "exact_duplicate_groups": len(conflicts["exact_groups"]),
        "cross_class_duplicate_groups": len(conflicts["exact_cross"]),
        "totals": totals,
        "per_class": per_class,
    }

    lines = []
    a = lines.append
    a("# 数据划分报告\n")
    a(f"- 数据集：`{args.data_dir}`")
    a(f"- 可用图片：**{images_available}** 张；使用 **{n_total}** 张，剔除 **{len(dropped)}** 张")
    a(f"- 划分比例：train / val / test = "
      f"{1 - args.val_frac - args.test_frac:.0%} / {args.val_frac:.0%} / {args.test_frac:.0%}")
    a(f"- 随机种子：`{args.seed}`")
    a(f"- 近重复阈值：dHash 汉明距离 ≤ {args.near_dup_dist}（仅用于同 split 分组，不删除）")
    a(f"- 重复簇：{len(clusters)} 个，其中多图簇 {len(multi)} 个；"
      f"**跨 split 的簇：{len(leaking)} 个**\n")

    a("## 各类别分布\n")
    a("| 类别 | train | val | test | 合计 |")
    a("|---|---:|---:|---:|---:|")
    for label in labels:
        c = per_class[label]
        a(f"| {label} | {c['train']} | {c['val']} | {c['test']} | "
          f"{c['train'] + c['val'] + c['test']} |")
    a(f"| **合计** | **{totals['train']}** | **{totals['val']}** | "
      f"**{totals['test']}** | **{n_total}** |")
    a(f"| 占比 | {totals['train']/n_total:.1%} | {totals['val']/n_total:.1%} | "
      f"{totals['test']/n_total:.1%} | 100% |\n")

    a("每个类别的 val/test 数量按比例四舍五入，train 为剩余部分，因此各类别占比最大偏差很小。\n")

    a("## 去重与泄漏防护\n")
    a(f"- 完全重复（MD5 相同）：{len(conflicts['exact_groups'])} 组，"
      f"涉及 {sum(len(g) for g in conflicts['exact_groups'])} 张；"
      f"跨类别 {len(conflicts['exact_cross'])} 组")
    for paths in conflict_paths:
        a("  - " + "; ".join(f"`{p}`" for p in paths))
    a("- 处理方式：完全重复每组只保留 1 张；跨类别冲突整组剔除")
    a(f"- 因标签冲突剔除 **{n_conflict_drop}** 张；"
      f"因完全重复剔除 **{n_exact_drop}** 张")
    a(f"- 近重复（非完全同一文件）：保留，{len(multi)} 个多图簇整体分入同一 split")
    a(f"- 校验结果：**{len(leaking)}** 个簇跨越多个 split（应为 0）\n")

    if dropped:
        a("### 已剔除的图片\n")
        a("| 路径 | 原因 |")
        a("|---|---|")
        for d in dropped:
            a(f"| `{d['path']}` | {d['reason']} |")
        a("\n跨类别冲突样本无法确定唯一标签，直接剔除；"
          "完全重复只保留分辨率/文件最大的一张。\n")

    a("## 产物\n")
    a("- `train.csv` / `val.csv` / `test.csv`：划分清单（`path,label,split,cluster_id,cluster_size`）")
    a("- `summary.json`：机器可读的划分统计")
    a("- `split_report.md`：本报告\n")

    (out_dir / "split_report.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Dedup-aware stratified split")
    parser.add_argument("--data-dir", default="flower_photos")
    parser.add_argument("--out-dir", default="splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--near-dup-dist", type=int, default=4,
                        help="dHash Hamming distance treated as near-duplicate "
                             "(clusters stay in one split; images are kept)")
    args = parser.parse_args()

    table_frac = args.val_frac + args.test_frac
    if not 0 < table_frac < 1:
        parser.error("--val-frac + --test-frac must be in (0, 1)")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {data_dir} ...")
    records = scan(data_dir)

    print("Clustering duplicates ...")
    clusters = build_clusters(records, args.near_dup_dist)
    conflicts = find_conflicts(records, clusters)

    drop = select_drops(records, clusters, conflicts)
    dropped = [
        {"path": records[i]["path"], "reason": reason}
        for i, reason in sorted(drop.items(), key=lambda kv: records[kv[0]]["path"])
    ]
    conflict_paths = [
        [records[i]["path"] for i in group]
        for group in conflicts["exact_cross"]
    ]

    kept_records, kept_clusters = remap(records, clusters, drop)
    labels = sorted({r["label"] for r in kept_records})
    print(f"  dropped {len(dropped)} images "
          f"({len(records)} -> {len(kept_records)}), "
          f"{len(kept_clusters)} clusters remain")

    print("Splitting ...")
    assignment = split_clusters(
        kept_clusters, labels, args.val_frac, args.test_frac, args.seed
    )

    rows = []
    for split in SPLITS:
        for c in sorted(assignment[split], key=lambda c: (c["label"], c["id"])):
            for i in c["members"]:
                rows.append({
                    "path": kept_records[i]["path"],
                    "label": kept_records[i]["label"],
                    "split": split,
                    "cluster_id": c["id"],
                    "cluster_size": c["size"],
                })
        write_manifest(
            [r for r in rows if r["split"] == split], out_dir / f"{split}.csv"
        )

    summary = write_report(
        rows, labels, assignment, kept_clusters,
        conflicts, conflict_paths, dropped, len(records), args, out_dir,
    )

    print("\n=== Split summary ===")
    print(f"used {summary['images_used']} / {summary['images_available']} images "
          f"(dropped {summary['images_dropped']})")
    print(f"clusters: {summary['clusters_total']} "
          f"(multi-image {summary['clusters_multi_image']}, "
          f"spanning splits {summary['clusters_spanning_splits']})")
    for label in labels:
        c = summary["per_class"][label]
        print(f"  {label:<12} train={c['train']:>4} val={c['val']:>4} "
              f"test={c['test']:>4} total={sum(c.values()):>4}")
    print(f"  {'all':<12} train={summary['totals']['train']:>4} "
          f"val={summary['totals']['val']:>4} test={summary['totals']['test']:>4}")
    print(f"wrote {out_dir}/train.csv, val.csv, test.csv, summary.json, split_report.md")


if __name__ == "__main__":
    main()
