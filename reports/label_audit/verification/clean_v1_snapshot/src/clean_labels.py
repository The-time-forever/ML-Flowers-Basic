"""Apply the manual label-cleaning decisions and build splits_clean/.

Policy:
- exclude images that contain no flower of any kind (vehicle, building,
  landscape, person/event, food/object, illustration/decoration);
- keep images that contain flowers but whose species label is doubtful.

The decision list below was produced by confidence screening + ImageNet object
screening + manual inspection. Only ``action == "exclude"`` removes a row.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path

SPLITS = ("train", "val", "test")

# path, issue, action, evidence
DECISIONS = [
    # ---- val ----
    ("roses/8462246855_1bdfee7478.jpg", "non-flower: vehicle", "exclude",
     "photo of a white Hummer; no rose"),
    ("roses/17953368844_be3d18cf30_m.jpg", "non-flower: building", "exclude",
     "gothic ceiling vault; no rose"),
    ("roses/15419696882_9394168a10_n.jpg", "non-flower: landscape", "exclude",
     "sunset over water; no rose"),
    ("roses/4256169180_55df2048a0.jpg", "non-flower: person/event", "exclude",
     "rodeo performer on horseback; no rose"),
    ("roses/8775267816_726ddc6d92_n.jpg", "species mismatch (flower present)",
     "keep", "pink gerbera-like flower; species doubtful, flower present"),
    ("tulips/13513644515_a51470b899.jpg", "species mismatch (flower present)",
     "keep", "daffodil field; species doubtful, flowers present"),
    # ---- test ----
    ("roses/12323085443_8ac0cdb713_n.jpg", "non-flower: person/event", "exclude",
     "horse-drawn parade carriage; no rose"),
    ("tulips/16862422576_5226e8d1d0.jpg", "non-flower: building/landscape",
     "exclude", "Gardens by the Bay Supertrees; no tulip"),
    ("tulips/15275199229_962387f24d.jpg", "non-flower: building/landscape",
     "exclude", "windmills on water; no tulip"),
    ("tulips/13513851673_9d813dc7b0.jpg", "species mismatch (flower present)",
     "keep", "hyacinth field with barn; flowers present, species doubtful"),
    # ---- train ----
    ("daisy/5700781400_65761f3fce.jpg", "non-flower: object", "exclude",
     "tree bark texture; no flower"),
    ("sunflowers/7176729016_d73ff2211e.jpg", "non-flower: person/event",
     "exclude", "outdoor market stalls; no sunflower"),
    ("tulips/17907238905_1ae121f8d9_m.jpg", "non-flower: building", "exclude",
     "war memorial; no tulip"),
    ("roses/5148639829_781eb7d346.jpg", "non-flower: landscape", "exclude",
     "field/fence landscape; no rose"),
    ("tulips/3511776685_3635087b12_n.jpg", "non-flower: food", "exclude",
     "strawberries; no tulip"),
    ("tulips/4522153453_06437ca3af_m.jpg", "non-flower: person/event", "exclude",
     "person painting outdoors; no tulip"),
    ("tulips/14116780333_7836f4448c.jpg", "non-flower: object", "exclude",
     "water fountain; no tulip"),
    ("tulips/6325571510_7544b27e57_n.jpg", "non-flower: object", "exclude",
     "teacup and biscuits; no tulip"),
    ("tulips/15275144259_f9a18ec9cb.jpg", "non-flower: building", "exclude",
     "steel arch bridge; no tulip"),
    ("roses/3921794817_276eb4386b.jpg", "non-flower: interior", "exclude",
     "bedroom interior; no rose"),
    ("roses/2863863372_605e29c03e_m.jpg", "non-flower: building", "exclude",
     "brick pub building; no rose"),
    ("dandelion/14648777167_1d92d403c9_n.jpg", "non-flower: landscape",
     "exclude", "green field and tree; no dandelion"),
    ("roses/15750320284_22ef21c682.jpg", "non-flower: person/event", "exclude",
     "marching band; no rose"),
    ("daisy/1355787476_32e9f2a30b.jpg", "non-flower: object", "exclude",
     "hand holding a water drop; no daisy"),
    ("roses/12450781274_eb78723921.jpg", "non-flower: vehicle", "exclude",
     "tram street scene; no rose"),
    ("roses/4292443009_3a2831b0b9_m.jpg", "non-flower: person/event", "exclude",
     "rodeo performer; no rose"),
    ("roses/12045735155_42547ce4e9_n.jpg", "non-flower: person/event", "exclude",
     "marching band; no rose"),
    ("roses/3655527028_0fab2b547d_n.jpg", "non-flower: interior/object",
     "exclude", "decorative china cabinet; no living rose"),
    ("tulips/14097328354_4f1469a170.jpg", "non-flower: person/event", "exclude",
     "person with camera; no tulip"),
    ("tulips/13910126337_53faf1d214_n.jpg", "non-flower: landscape", "exclude",
     "branches over water; no tulip"),
    ("roses/9406573080_60eab9278e_n.jpg", "non-flower: person/event", "exclude",
     "crowd at a festival; no rose"),
    ("roses/2364976562_a184463083_m.jpg", "non-flower: object", "exclude",
     "dolls; no rose"),
    ("tulips/15275504998_ca9eb82998.jpg", "non-flower: building/object",
     "exclude", "window and bicycle; no tulip"),
    ("tulips/13910131718_731353d84c_n.jpg", "non-flower: person/event", "exclude",
     "black-and-white people; no tulip"),
    ("roses/5193918046_d44e4fcd75_m.jpg", "non-flower: decoration", "exclude",
     "holiday candles/decor; no rose"),
    ("roses/180613732_3a7aba0b80_n.jpg", "non-flower: landscape/animal",
     "exclude", "forest and ducks; no rose"),
    ("roses/319298955_0c72bd36bf.jpg", "non-flower: food", "exclude",
     "chocolate cake with decorative roses; no real flower"),
    ("tulips/7064813645_f7f48fb527.jpg", "non-flower: vehicle/field", "exclude",
     "tractor in a field; no tulip"),
    ("sunflowers/9381481549_5a5d503e42_n.jpg", "non-flower: plants", "exclude",
     "green seedlings, no sunflower bloom"),
    ("roses/16334786972_1b3e71cab8_m.jpg", "non-flower: illustration/decoration",
     "exclude", "vintage seed catalogue page; no real rose"),
    ("roses/2568105249_15720d081f_n.jpg", "non-flower: interior", "exclude",
     "interior room with wedding decor; no rose"),
    ("tulips/14487705209_ea723109e1_m.jpg", "non-flower: illustration/decoration",
     "exclude", "tulip illustration; no real flower"),
    ("dandelion/4573204385_9b71e96b35_m.jpg", "non-flower: plants", "exclude",
     "grass blades with dew; no dandelion"),
    ("roses/6069602140_866eecf7c2_m.jpg", "non-flower: illustration/decoration",
     "exclude", "tattoo artwork; no real rose"),
    ("sunflowers/7270375648_79f0caef42_n.jpg", "non-flower: light art",
     "exclude", "abstract sunflower-shaped light installation; no real flower"),
    # species-doubtful but flower present: keep
    ("roses/15174615529_144ae28bdb_n.jpg", "species mismatch (flower present)",
     "keep", "portulaca-like flower; species doubtful, flower present"),
    ("roses/5212885371_fe27c406a2_n.jpg", "species mismatch (flower present)",
     "keep", "orange gerbera-like flower; species doubtful, flower present"),
    ("tulips/4546299243_23cd58eb43.jpg", "species mismatch (flower present)",
     "keep", "cherry blossom path; flowers present, species doubtful"),
    ("tulips/5674707464_dc18de05b1.jpg", "species mismatch (flower present)",
     "keep", "lavender field; flowers present, species doubtful"),
]


def main() -> None:
    data_dir = Path("flower_photos")
    splits_dir = Path("splits")
    clean_dir = Path("splits_clean")
    out_dir = Path("reports/label_audit")

    split_of: dict[str, str] = {}
    split_rows: dict[str, list[dict]] = {}
    for split in SPLITS:
        with (splits_dir / f"{split}.csv").open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        split_rows[split] = rows
        for row in rows:
            if row["path"] in split_of:
                raise ValueError(f"duplicate path across splits: {row['path']}")
            split_of[row["path"]] = split

    seen = set()
    cleaning_rows = []
    missing = []
    for path, issue, action, evidence in DECISIONS:
        if path in seen:
            raise ValueError(f"duplicate decision: {path}")
        seen.add(path)
        if path not in split_of:
            missing.append(f"{path} (not in splits)")
            continue
        if not (data_dir / path).is_file():
            missing.append(f"{path} (file missing)")
            continue
        cleaning_rows.append({
            "path": path,
            "split": split_of[path],
            "issue": issue,
            "action": action,
            "evidence": evidence,
        })
    if missing:
        raise FileNotFoundError("invalid decision paths:\n" + "\n".join(missing))

    exclude = {r["path"] for r in cleaning_rows if r["action"] == "exclude"}
    clean_dir.mkdir(exist_ok=True)
    summary = {"policy": "exclude only images with no flower of any kind; "
                         "keep species-doubtful flowers",
               "before": {}, "after": {}, "excluded": {}}
    for split in SPLITS:
        rows = split_rows[split]
        kept = [r for r in rows if r["path"] not in exclude]
        with (clean_dir / f"{split}.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(kept)
        summary["before"][split] = len(rows)
        summary["after"][split] = len(kept)
        summary["excluded"][split] = len(rows) - len(kept)
        summary[f"classes_before_{split}"] = dict(Counter(r["label"] for r in rows))
        summary[f"classes_after_{split}"] = dict(Counter(r["label"] for r in kept))
    (clean_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (out_dir / "cleaning_list.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "split", "issue",
                                                "action", "evidence"])
        writer.writeheader()
        writer.writerows(cleaning_rows)

    excluded = [r for r in cleaning_rows if r["action"] == "exclude"]
    kept = [r for r in cleaning_rows if r["action"] == "keep"]
    lines = ["# 标签清洗记录\n",
             "- 标准：图片是否包含符合类别定义的真实花卉；",
             "- 处理：无任何花卉/明确无关的图片排除；花卉种类存疑的保留。\n",
             "## 汇总\n",
             "| split | 原始 | 排除 | 清洗后 |",
             "|---|---:|---:|---:|"]
    for split in SPLITS:
        lines.append(f"| {split} | {summary['before'][split]} | "
                     f"{summary['excluded'][split]} | {summary['after'][split]} |")
    lines += [f"| **合计** | **{sum(summary['before'].values())}** | "
              f"**{sum(summary['excluded'].values())}** | "
              f"**{sum(summary['after'].values())}** |", "",
              "## 已排除样本\n",
              "| 路径 | split | 问题 | 依据 |",
              "|---|---|---|---|"]
    for r in excluded:
        lines.append(f"| `{r['path']}` | {r['split']} | {r['issue']} | {r['evidence']} |")
    lines += ["", "## 保留的物种存疑样本\n",
              "| 路径 | split | 问题 | 依据 |",
              "|---|---|---|---|"]
    for r in kept:
        lines.append(f"| `{r['path']}` | {r['split']} | {r['issue']} | {r['evidence']} |")
    lines += ["", "完整机器可读清单见 `cleaning_list.csv`；清洗后的划分在 `splits_clean/`。", ""]
    (out_dir / "cleaning_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"excluded {len(excluded)} images, kept {len(kept)} doubtful flowers")
    for split in SPLITS:
        print(f"  {split}: {summary['before'][split]} -> {summary['after'][split]}")
    print(f"wrote {out_dir / 'cleaning_list.csv'}")
    print(f"wrote {clean_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
