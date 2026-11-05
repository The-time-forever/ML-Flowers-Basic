# clean 标签清洗交叉复核

> 处理状态：本报告记录的建议已应用到 `clean-v2`。恢复 4 张误排除样本，补充排除 14 张遗漏样本；另外 2 张依据不足的原排除项改为保留待复核。当前 train/val/test 为 2531/540/545 张，共 3616 张。当前决策见上一级 `cleaning_list.csv`，校验见 `clean_v2_integrity.json`。下文保留修订前的审查发现。

## 结论

当前 clean 尚未清洗完整，同时存在误排除与遗漏。已完成第一轮无关样本筛查；应先修订复核清单，再生成下一版划分。此次只新增复核产物，没有修改 cleaning_list.csv、clean_labels.py、splits_clean 或原始图片。

## 审查范围与方法

- 阅读现有清洗报告、49 条决策（42 exclude、7 keep）、清洗脚本、置信度筛查和 ImageNet 物体筛查实现。
- 逐张缩略图筛查 49 条决策、226 个保留集物体筛查命中、120 个保留集低置信度样本；去重后共 383 张。对关键争议和新增排除候选放大查看本地原图。
- 根据原始 LICENSE.txt 的作者和 Flickr 链接访问 Flickr 官方 oEmbed 元数据：25 次查询，24 次成功，1 次 404。返回的作者/标题保存在 source_checks.json。标题用于交叉支持图片内容判断，泛化地名、编号标题不用于确认物种。
- 网络搜索对部分照片仅返回许可证或地点信息；7 个既有物种存疑样本的植物学真值仍未解决。
- 本次属于候选样本定向复核，未对全部 3624 个保留样本逐张做物种鉴定；数量不能外推为总体错标率。

## 划分完整性

- train: 2564 → 2529；val: 551 → 547；test: 551 → 548。
- 三份 clean CSV 都严格等于原 CSV 按 42 个 exclude 路径过滤的结果，其他字段及原归属保持一致。
- 保留路径唯一、对应文件均存在；跨 split cluster_id 数量为 0。完整结果见 integrity.json。

## 建议恢复 / 撤销原排除依据（4 张）

蒲公英种子头在现有数据中大量保留，局部种子特写应使用同一标准。向日葵样本右侧存在黄色花头，原先“只有幼苗，无花”的依据与图像不符。

| 路径 | split | 本次依据 | 原作者来源 |
|---|---|---|---|
| `daisy/5700781400_65761f3fce.jpg` | train | A white ray-flowered head with a yellow centre is clearly visible; original title: Daisy on the rocks. | [Daisy on the rocks](https://www.flickr.com/photos/viatorius/5700781400/) |
| `dandelion/14648777167_1d92d403c9_n.jpg` | train | Yellow flowers are visible throughout the field; original title: Blinking Dandelions. The previous no-flower rationale is unsupported. | [Blinking Dandelions](https://www.flickr.com/photos/winterpearl/14648777167/) |
| `dandelion/4573204385_9b71e96b35_m.jpg` | train | Macro photograph of seed parachutes with water drops, not grass blades; consistent with seed-head photographs retained elsewhere. Original title: Liquid Sun Seed. | [Liquid Sun Seed](https://www.flickr.com/photos/ladydragonflyherworld/4573204385/) |
| `sunflowers/9381481549_5a5d503e42_n.jpg` | train | A small yellow flower head is visible on the right; original title: Sunflowers Wk 7. The previous no-bloom rationale is unsupported. | [Sunflowers Wk 7](https://www.flickr.com/photos/myeral/9381481549/) |

## 建议补入排除清单（14 张）

| 路径 | split | 本次依据 | 原作者来源 |
|---|---|---|---|
| `roses/16424992340_c1d9eb72b4.jpg` | train | Waterfront and boat; no visible rose. Original title: Rays of Sunlight. | [Rays of Sunlight](https://www.flickr.com/photos/dvanzuijlekom/16424992340/) |
| `roses/4248222578_b4d5868b32.jpg` | train | Marching brass band. Original title: OSU Tuba Line. | [OSU Tuba Line](https://www.flickr.com/photos/prayitnophotography/4248222578/) |
| `roses/8590442797_07fa2141c0_n.jpg` | train | Retail sign reading Roses; no flower. Original title: Roses in Akron, Ohio. | [Roses in Akron, Ohio](https://www.flickr.com/photos/fanofretail/8590442797/) |
| `roses/16316557109_7fc55c1cbc_m.jpg` | train | Botanical catalogue illustration; exclude under the existing real-flower-photo policy, consistent with other excluded illustrations. | [n37_w1150](https://www.flickr.com/photos/biodivlibrary/16316557109/) |
| `tulips/16702188449_3dacce90b2_m.jpg` | val | Supertree structure and foliage; no visible tulip. Original title: Supertree. | [Supertree](https://www.flickr.com/photos/ahmed_xp/16702188449/) |
| `tulips/16862374316_4135908d4c_m.jpg` | val | Decorative windmill; no visible tulip. Original title: Windmill. | [Windmill](https://www.flickr.com/photos/ahmed_xp/16862374316/) |
| `tulips/17202535346_ab828e779b.jpg` | val | Tree canopy. Original title: Split Leaf Japanese maple trees. | [Split Leaf Japanese maple trees](https://www.flickr.com/photos/55229469@N07/17202535346/) |
| `tulips/6948239566_0ac0a124ee_n.jpg` | val | Printed mascot design sheet; no flower. Original title: Kurogawa Onsen. | [Kurogawa Onsen 黑川溫泉](https://www.flickr.com/photos/14541393@N03/6948239566/) |
| `tulips/7070694881_e9a331fa4e_n.jpg` | val | Tree trunks and canopy; no visible tulip. Original Flickr oEmbed returned 404; decision is based on the local image. | [原作者页面](https://www.flickr.com/photos/seanfx/7070694881/) |
| `roses/2408236801_f43c6bcff2.jpg` | val | Hanging teacup ornaments; no real flower. Original title: Tea Room. | [Tea Room](https://www.flickr.com/photos/sherrysrosecottage/2408236801/) |
| `roses/7211616670_2d49ecb3a5_m.jpg` | val | Cupcake tower; flower-like icing decorations. Original title: Love and lace wedding cupcake tower. | [Love and lace wedding cupcake tower](https://www.flickr.com/photos/kristinausk/7211616670/) |
| `tulips/5687705933_55a8c2dbac.jpg` | test | Tree branches viewed through a window; no visible tulip. Original title: Botanic garden - Cluj-Napoca. | [Botanic garden - Cluj-Napoca](https://www.flickr.com/photos/bortescristian/5687705933/) |
| `tulips/15275190769_0ed7bbf490.jpg` | test | Green wooden building and bicycle; no visible tulip. Original title: Holland. | [Holland](https://www.flickr.com/photos/phalaenopsisaphrodite523/15275190769/) |
| `sunflowers/418056361_1dfac1c151_n.jpg` | test | Photograph of a sunflower painting, not a living flower; exclude under the existing real-flower-photo policy. | [Girasoli, close close up](https://www.flickr.com/photos/giopuo/418056361/) |

## 其他需要复核的现有排除项

| 路径 | split | 本次依据 | 原作者来源 |
|---|---|---|---|
| `tulips/7064813645_f7f48fb527.jpg` | train | Tractor in front of blooming blue flower rows. Existing no-flower statement is inaccurate; the visible flowers do not establish a tulip label. | [ALG_8321](https://www.flickr.com/photos/nikontino/7064813645/) |
| `roses/2568105249_15720d081f_n.jpg` | train | Interior with several floral arrangements. Whether these are real or artificial cannot be established confidently from this image. | [A corner of my family room.](https://www.flickr.com/photos/sherrysrosecottage/2568105249/) |

## 需要明确的范围边界

| 路径 | split | 本次依据 | 原作者来源 |
|---|---|---|---|
| `roses/475936554_a2b38aaa8e.jpg` | test | Rose stem, thorns and leaves with droplets; original title: Rose Thorns. Decide explicitly whether vegetative rose parts are in scope. | [Rose Thorns](https://www.flickr.com/photos/peasap/475936554/) |

## 执行建议

1. 优先恢复上述 4 项，撤销错误的“无花”判定；对提出的 14 项按现有真实花卉照片标准补充排除记录。
2. 拖拉机图实际包含大片花田，室内图包含花卉状摆设，均应纠正现有文字依据；确认是否符合类别/真实性标准后决定保留与否。
3. 明确非开花期植株、种子头、绘画、插画、人工花及混合花束的处理标准；种子头应和现有蒲公英样本一致。
4. 7 个保留的物种疑似项继续标记为待鉴定，不依据分类模型输出直接改标签。
5. 所有决定应先写入统一清单，再同步 clean_labels.py 中硬编码的 DECISIONS，避免重新运行脚本覆盖人工修订。保持原 split 归属，保存 clean 版本和排除清单。
