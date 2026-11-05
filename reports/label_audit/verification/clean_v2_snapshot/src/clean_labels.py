"""Build versioned clean manifests from the canonical manual decision CSV.

Run from the repository root: python src/clean_labels.py
Only explicit exclude decisions remove rows. Keep and review preserve original
labels and split membership. No image files or original manifests are changed.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

SPLITS = ("train", "val", "test")
POLICY = (
    "Exclude confirmed irrelevant images and non-photographic depictions; "
    "retain target flowers and seed heads; retain unresolved species, "
    "authenticity and vegetative-part cases with review flags."
)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    data_dir = Path("flower_photos")
    clean_dir = Path("splits_clean")
    audit_dir = Path("reports/label_audit")
    decision_path = audit_dir / "cleaning_list.csv"
    decisions = read_csv(decision_path)
    original = {s: read_csv(Path("splits") / f"{s}.csv") for s in SPLITS}
    owner = {}
    for split, rows in original.items():
        if not rows:
            raise ValueError(f"empty original split: {split}")
        for row in rows:
            path = row["path"]
            if path in owner or row["split"] != split:
                raise ValueError(f"duplicate path or wrong split: {path}")
            if row["label"] != Path(path).parts[0]:
                raise ValueError(f"label/path mismatch: {path}")
            if not (data_dir / path).is_file():
                raise FileNotFoundError(data_dir / path)
            owner[path] = split

    seen = set()
    for row in decisions:
        path = row["path"]
        if path in seen or path not in owner:
            raise ValueError(f"duplicate or unknown decision path: {path}")
        if row["action"] not in {"exclude", "keep", "review"}:
            raise ValueError(f"unknown action for {path}: {row['action']}")
        if row["split"] != owner[path] or not row["evidence"].strip():
            raise ValueError(f"wrong split or missing evidence: {path}")
        seen.add(path)

    exclude = {r["path"] for r in decisions if r["action"] == "exclude"}
    filtered = {
        s: [r for r in rows if r["path"] not in exclude]
        for s, rows in original.items()
    }
    clusters = {}
    for split, rows in filtered.items():
        for row in rows:
            clusters.setdefault(row["cluster_id"], set()).add(split)
    if any(len(splits) > 1 for splits in clusters.values()):
        raise ValueError("duplicate cluster crosses split boundaries")

    summary = {
        "version": "clean-v2",
        "policy": POLICY,
        "decisions_file": decision_path.as_posix(),
        "decisions_sha256": hashlib.sha256(decision_path.read_bytes()).hexdigest(),
        "decision_counts": dict(Counter(r["action"] for r in decisions)),
        "before": {}, "after": {}, "excluded": {},
        "cross_split_clusters": 0,
    }
    clean_dir.mkdir(exist_ok=True)
    for split, rows in original.items():
        kept = filtered[split]
        with (clean_dir / f"{split}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(kept)
        summary["before"][split] = len(rows)
        summary["after"][split] = len(kept)
        summary["excluded"][split] = len(rows) - len(kept)
        summary[f"classes_before_{split}"] = dict(Counter(r["label"] for r in rows))
        summary[f"classes_after_{split}"] = dict(Counter(r["label"] for r in kept))
    (clean_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# 标签清洗记录", "", "- 版本：`clean-v2`。",
        "- 唯一决策来源：`reports/label_audit/cleaning_list.csv`；脚本直接读取该清单。",
        "- 排除：确定无关的照片，以及绘画、插画、食品装饰等非真实花卉图像。",
        "- 保留：真实目标花卉、种子头；物种、真实性或范围存疑的样本保留原标签并标记复核。",
        "- 原始图片及原始划分保持不变；`review` 样本继续留在 clean 中。", "",
        "## 汇总", "", "| split | 原始 | 排除 | 清洗后 |",
        "|---|---:|---:|---:|",
    ]
    for split in SPLITS:
        lines.append(
            f"| {split} | {summary['before'][split]} | "
            f"{summary['excluded'][split]} | {summary['after'][split]} |"
        )
    lines += [
        f"| **合计** | **{sum(summary['before'].values())}** | "
        f"**{len(exclude)}** | **{sum(summary['after'].values())}** |", "",
        "## 本轮修订", "",
        "- 恢复 4 张被误排除的花卉或种子特写。",
        "- 新增排除 14 张经图片及来源复核确认的遗漏样本。",
        "- 拖拉机花田、室内花卉摆设 2 张撤销确定排除，改为保留待复核。",
        "- 玫瑰枝叶特写保留并标记范围待复核；原有 7 张物种存疑图片继续保留。",
        "- 初版清单、报告、脚本和划分备份在 `verification/clean_v1_snapshot/`。",
        "- 本轮定向复核范围为 383 张候选图片；其余图片尚未完成逐张物种鉴定。", "",
    ]
    for action, title in (
        ("exclude", "已排除样本"),
        ("keep", "已确认恢复或物种存疑保留样本"),
        ("review", "待复核样本（当前保留）"),
    ):
        lines += [f"## {title}", "", "| 路径 | split | 问题 | 依据 | 来源 |",
                  "|---|---|---|---|---|"]
        for row in decisions:
            if row["action"] != action:
                continue
            url = row.get("source_url", "")
            source = f"[原作者]({url})" if url else "本地图像复核"
            lines.append(
                f"| `{row['path']}` | {row['split']} | {row['issue']} | "
                f"{row['evidence']} | {source} |"
            )
        lines.append("")
    (audit_dir / "cleaning_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps({k: summary[k] for k in (
        "version", "decision_counts", "before", "after", "excluded"
    )}, indent=2))


if __name__ == "__main__":
    main()
