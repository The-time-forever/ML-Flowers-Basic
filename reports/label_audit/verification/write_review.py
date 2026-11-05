"""Write review findings without changing cleaning decisions or manifests."""

import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


findings = [
    ("daisy/5700781400_65761f3fce.jpg", "restore", "A white ray-flowered head with a yellow centre is clearly visible; original title: Daisy on the rocks."),
    ("dandelion/14648777167_1d92d403c9_n.jpg", "restore", "Yellow flowers are visible throughout the field; original title: Blinking Dandelions. The previous no-flower rationale is unsupported."),
    ("dandelion/4573204385_9b71e96b35_m.jpg", "restore", "Macro photograph of seed parachutes with water drops, not grass blades; consistent with seed-head photographs retained elsewhere. Original title: Liquid Sun Seed."),
    ("sunflowers/9381481549_5a5d503e42_n.jpg", "restore", "A small yellow flower head is visible on the right; original title: Sunflowers Wk 7. The previous no-bloom rationale is unsupported."),
    ("roses/16424992340_c1d9eb72b4.jpg", "exclude", "Waterfront and boat; no visible rose. Original title: Rays of Sunlight."),
    ("roses/4248222578_b4d5868b32.jpg", "exclude", "Marching brass band. Original title: OSU Tuba Line."),
    ("roses/8590442797_07fa2141c0_n.jpg", "exclude", "Retail sign reading Roses; no flower. Original title: Roses in Akron, Ohio."),
    ("roses/16316557109_7fc55c1cbc_m.jpg", "exclude", "Botanical catalogue illustration; exclude under the existing real-flower-photo policy, consistent with other excluded illustrations."),
    ("tulips/16702188449_3dacce90b2_m.jpg", "exclude", "Supertree structure and foliage; no visible tulip. Original title: Supertree."),
    ("tulips/16862374316_4135908d4c_m.jpg", "exclude", "Decorative windmill; no visible tulip. Original title: Windmill."),
    ("tulips/17202535346_ab828e779b.jpg", "exclude", "Tree canopy. Original title: Split Leaf Japanese maple trees."),
    ("tulips/6948239566_0ac0a124ee_n.jpg", "exclude", "Printed mascot design sheet; no flower. Original title: Kurogawa Onsen."),
    ("tulips/7070694881_e9a331fa4e_n.jpg", "exclude", "Tree trunks and canopy; no visible tulip. Original Flickr oEmbed returned 404; decision is based on the local image."),
    ("roses/2408236801_f43c6bcff2.jpg", "exclude", "Hanging teacup ornaments; no real flower. Original title: Tea Room."),
    ("roses/7211616670_2d49ecb3a5_m.jpg", "exclude", "Cupcake tower; flower-like icing decorations. Original title: Love and lace wedding cupcake tower."),
    ("tulips/5687705933_55a8c2dbac.jpg", "exclude", "Tree branches viewed through a window; no visible tulip. Original title: Botanic garden - Cluj-Napoca."),
    ("tulips/15275190769_0ed7bbf490.jpg", "exclude", "Green wooden building and bicycle; no visible tulip. Original title: Holland."),
    ("sunflowers/418056361_1dfac1c151_n.jpg", "exclude", "Photograph of a sunflower painting, not a living flower; exclude under the existing real-flower-photo policy."),
    ("roses/475936554_a2b38aaa8e.jpg", "policy_review", "Rose stem, thorns and leaves with droplets; original title: Rose Thorns. Decide explicitly whether vegetative rose parts are in scope."),
    ("tulips/7064813645_f7f48fb527.jpg", "review", "Tractor in front of blooming blue flower rows. Existing no-flower statement is inaccurate; the visible flowers do not establish a tulip label."),
    ("roses/2568105249_15720d081f_n.jpg", "review", "Interior with several floral arrangements. Whether these are real or artificial cannot be established confidently from this image."),
]

sources = json.loads((OUT / "source_checks.json").read_text(encoding="utf-8"))
source_of = {row["path"]: row for row in sources}
original = {
    row["path"]: row
    for split in ("train", "val", "test")
    for row in read_csv(f"splits/{split}.csv")
}
decisions = {row["path"]: row for row in read_csv("reports/label_audit/cleaning_list.csv")}
rows = []
for path, recommendation, evidence in findings:
    row = original[path]
    rows.append({
        "path": path,
        "split": row["split"],
        "current_action": decisions.get(path, {}).get("action", "keep_unlisted"),
        "recommendation": recommendation,
        "evidence": evidence,
        "source_url": source_of.get(path, {}).get("url", ""),
        "source_title": source_of.get(path, {}).get("title", ""),
    })
with (OUT / "review_findings.csv").open("w", newline="", encoding="utf-8-sig") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

reviewed = set(decisions)
for name in ("retained_objects", "retained_lowconf"):
    reviewed.update(row["path"] for row in json.loads((OUT / f"{name}_index.json").read_text()))
print("Unique paths visually screened:", len(reviewed))
print("Recommendations:", len(rows), "; source lookups:", len(sources))
