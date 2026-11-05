# 标签清洗记录

- 版本：`clean-v3`。
- 唯一决策来源：`reports/label_audit/cleaning_list.csv`；脚本直接读取该清单。
- 排除：确定无关、目标类别明显不符的照片，以及绘画、插画、食品装饰等非真实花卉图像。
- 保留：目标花卉、种子头及可辨认的植株枝叶；混合场景中可见目标的保留。真实性证据不足时保守保留并记录限制。
- 原始图片及原始划分保持不变；`review` 样本继续留在 clean 中。

## 汇总

| split | 原始 | 排除 | 清洗后 |
|---|---:|---:|---:|
| train | 2564 | 37 | 2527 |
| val | 551 | 13 | 538 |
| test | 551 | 7 | 544 |
| **合计** | **3666** | **57** | **3609** |

## 前轮修订（clean-v2）

- 恢复 4 张被误排除的花卉或种子特写。
- 新增排除 14 张经图片及来源复核确认的遗漏样本。
- 拖拉机花田、室内花卉摆设 2 张撤销确定排除，改为保留待复核。
- 当时保留 7 张物种存疑图片及 3 张范围/真实性待复核图片。
- 初版清单、报告、脚本和划分备份在 `verification/clean_v1_snapshot/`。
- 本轮定向复核范围为 383 张候选图片；其余图片尚未完成逐张物种鉴定。

## 本轮修订（clean-v3）

- 对剩余 10 张逐项复核：7 张排除、3 张保留，无剩余 review 动作。
- 排除：6 张目标类别明显不符的花卉图和 1 张拖拉机/非郁金香花田。
- 保留：树下可见郁金香的混合花园、原作者确认的玫瑰枝叶特写、正常的室内白色玫瑰花束。
- 室内白色玫瑰图经用户确认按正常样本保留，已移除存疑标记。
- 逐项依据见 `verification/remaining10_decisions.csv`；原作者网页、元数据及较大图片均在 verification 中留档。
- clean-v2 备份位于 `verification/clean_v2_snapshot/`。

## 已排除样本

| 路径 | split | 问题 | 依据 | 来源 |
|---|---|---|---|---|
| `roses/8462246855_1bdfee7478.jpg` | val | non-flower: vehicle | photo of a white Hummer; no rose | 本地图像复核 |
| `roses/17953368844_be3d18cf30_m.jpg` | val | non-flower: building | gothic ceiling vault; no rose | 本地图像复核 |
| `roses/15419696882_9394168a10_n.jpg` | val | non-flower: landscape | sunset over water; no rose | 本地图像复核 |
| `roses/4256169180_55df2048a0.jpg` | val | non-flower: person/event | rodeo performer on horseback; no rose | 本地图像复核 |
| `roses/8775267816_726ddc6d92_n.jpg` | val | class mismatch: non-rose flower | The enlarged original shows numerous narrow radial florets in a composite flower head, inconsistent with a rose. Exact taxon is not assigned; exclude from roses. Original title: 009 pink star burst. | [原作者](https://www.flickr.com/photos/rosalee333/8775267816/) |
| `tulips/13513644515_a51470b899.jpg` | val | class mismatch: hyacinth field | Enlarged original shows dense spikes of many small flowers and strap-like leaves, consistent with hyacinths rather than tulips. Previous daffodil description corrected. Exclude from tulips. | [原作者](https://www.flickr.com/photos/nikontino/13513644515/) |
| `roses/12323085443_8ac0cdb713_n.jpg` | test | non-flower: person/event | horse-drawn parade carriage; no rose | 本地图像复核 |
| `tulips/16862422576_5226e8d1d0.jpg` | test | non-flower: building/landscape | Gardens by the Bay Supertrees; no tulip | 本地图像复核 |
| `tulips/15275199229_962387f24d.jpg` | test | non-flower: building/landscape | windmills on water; no tulip | 本地图像复核 |
| `tulips/13513851673_9d813dc7b0.jpg` | test | class mismatch: hyacinth field | Enlarged original shows dense blue and pink hyacinth flower spikes; no identifiable tulip target. Exclude from tulips. | [原作者](https://www.flickr.com/photos/nikontino/13513851673/) |
| `sunflowers/7176729016_d73ff2211e.jpg` | train | non-flower: person/event | outdoor market stalls; no sunflower | 本地图像复核 |
| `tulips/17907238905_1ae121f8d9_m.jpg` | train | non-flower: building | war memorial; no tulip | 本地图像复核 |
| `roses/5148639829_781eb7d346.jpg` | train | non-flower: landscape | field/fence landscape; no rose | 本地图像复核 |
| `tulips/3511776685_3635087b12_n.jpg` | train | non-flower: food | strawberries; no tulip | 本地图像复核 |
| `tulips/4522153453_06437ca3af_m.jpg` | train | non-flower: person/event | person painting outdoors; no tulip | 本地图像复核 |
| `tulips/14116780333_7836f4448c.jpg` | train | non-flower: object | water fountain; no tulip | 本地图像复核 |
| `tulips/6325571510_7544b27e57_n.jpg` | train | non-flower: object | teacup and biscuits; no tulip | 本地图像复核 |
| `tulips/15275144259_f9a18ec9cb.jpg` | train | non-flower: building | steel arch bridge; no tulip | 本地图像复核 |
| `roses/3921794817_276eb4386b.jpg` | train | non-flower: interior | bedroom interior; no rose | 本地图像复核 |
| `roses/2863863372_605e29c03e_m.jpg` | train | non-flower: building | brick pub building; no rose | 本地图像复核 |
| `roses/15750320284_22ef21c682.jpg` | train | non-flower: person/event | marching band; no rose | 本地图像复核 |
| `daisy/1355787476_32e9f2a30b.jpg` | train | non-flower: object | hand holding a water drop; no daisy | 本地图像复核 |
| `roses/12450781274_eb78723921.jpg` | train | non-flower: vehicle | tram street scene; no rose | 本地图像复核 |
| `roses/4292443009_3a2831b0b9_m.jpg` | train | non-flower: person/event | rodeo performer; no rose | 本地图像复核 |
| `roses/12045735155_42547ce4e9_n.jpg` | train | non-flower: person/event | marching band; no rose | 本地图像复核 |
| `roses/3655527028_0fab2b547d_n.jpg` | train | non-flower: interior/object | decorative china cabinet; no living rose | 本地图像复核 |
| `tulips/14097328354_4f1469a170.jpg` | train | non-flower: person/event | person with camera; no tulip | 本地图像复核 |
| `tulips/13910126337_53faf1d214_n.jpg` | train | non-flower: landscape | branches over water; no tulip | 本地图像复核 |
| `roses/9406573080_60eab9278e_n.jpg` | train | non-flower: person/event | crowd at a festival; no rose | 本地图像复核 |
| `roses/2364976562_a184463083_m.jpg` | train | non-flower: object | dolls; no rose | 本地图像复核 |
| `tulips/15275504998_ca9eb82998.jpg` | train | non-flower: building/object | window and bicycle; no tulip | 本地图像复核 |
| `tulips/13910131718_731353d84c_n.jpg` | train | non-flower: person/event | black-and-white people; no tulip | 本地图像复核 |
| `roses/5193918046_d44e4fcd75_m.jpg` | train | non-flower: decoration | holiday candles/decor; no rose | 本地图像复核 |
| `roses/180613732_3a7aba0b80_n.jpg` | train | non-flower: landscape/animal | forest and ducks; no rose | 本地图像复核 |
| `roses/319298955_0c72bd36bf.jpg` | train | non-flower: food | chocolate cake with decorative roses; no real flower | 本地图像复核 |
| `tulips/7064813645_f7f48fb527.jpg` | train | class mismatch: tractor with non-tulip flower rows | Photograph contains flowers, but the visible dense blue flower rows are consistent with hyacinths; no identifiable tulip target. Exclude from tulips. Replaces the incorrect no-flower rationale. | [原作者](https://www.flickr.com/photos/nikontino/7064813645/) |
| `roses/16334786972_1b3e71cab8_m.jpg` | train | non-flower: illustration/decoration | vintage seed catalogue page; no real rose | 本地图像复核 |
| `tulips/14487705209_ea723109e1_m.jpg` | train | non-flower: illustration/decoration | tulip illustration; no real flower | 本地图像复核 |
| `roses/6069602140_866eecf7c2_m.jpg` | train | non-flower: illustration/decoration | tattoo artwork; no real rose | 本地图像复核 |
| `sunflowers/7270375648_79f0caef42_n.jpg` | train | non-flower: light art | abstract sunflower-shaped light installation; no real flower | 本地图像复核 |
| `roses/15174615529_144ae28bdb_n.jpg` | train | class mismatch: clover image with decorative overlay | Original author title is Pink clover flowers. Main photograph contains small pink non-rose flowers; corner rose is a graphic overlay. Exclude from roses; no exact botanical relabelling. | [原作者](https://www.flickr.com/photos/82066314@N06/15174615529/) |
| `roses/5212885371_fe27c406a2_n.jpg` | train | class mismatch: gerbera-like composite flower heads | Local original clearly shows daisy-like composite heads with central discs, consistent with gerbera, rather than roses. Original Flickr page returned 404. Exclude from roses without relabelling to daisy. | [原作者](https://www.flickr.com/photos/surfergirl30/5212885371/) |
| `tulips/5674707464_dc18de05b1.jpg` | train | class mismatch: blue spike-flowered border | Larger original shows a continuous border of many small flowers along elongated blue spikes, not tulip flowers. Exact taxon is not assigned. Generic author tag tulips is insufficient to establish the image label. | [原作者](https://www.flickr.com/photos/gnuckx/5674707464/) |
| `roses/16424992340_c1d9eb72b4.jpg` | train | irrelevant image or non-photographic depiction | Waterfront and boat; no visible rose. Original title: Rays of Sunlight. | [原作者](https://www.flickr.com/photos/dvanzuijlekom/16424992340/) |
| `roses/4248222578_b4d5868b32.jpg` | train | irrelevant image or non-photographic depiction | Marching brass band. Original title: OSU Tuba Line. | [原作者](https://www.flickr.com/photos/prayitnophotography/4248222578/) |
| `roses/8590442797_07fa2141c0_n.jpg` | train | irrelevant image or non-photographic depiction | Retail sign reading Roses; no flower. Original title: Roses in Akron, Ohio. | [原作者](https://www.flickr.com/photos/fanofretail/8590442797/) |
| `roses/16316557109_7fc55c1cbc_m.jpg` | train | irrelevant image or non-photographic depiction | Botanical catalogue illustration; exclude under the existing real-flower-photo policy, consistent with other excluded illustrations. | [原作者](https://www.flickr.com/photos/biodivlibrary/16316557109/) |
| `tulips/16702188449_3dacce90b2_m.jpg` | val | irrelevant image or non-photographic depiction | Supertree structure and foliage; no visible tulip. Original title: Supertree. | [原作者](https://www.flickr.com/photos/ahmed_xp/16702188449/) |
| `tulips/16862374316_4135908d4c_m.jpg` | val | irrelevant image or non-photographic depiction | Decorative windmill; no visible tulip. Original title: Windmill. | [原作者](https://www.flickr.com/photos/ahmed_xp/16862374316/) |
| `tulips/17202535346_ab828e779b.jpg` | val | irrelevant image or non-photographic depiction | Tree canopy. Original title: Split Leaf Japanese maple trees. | [原作者](https://www.flickr.com/photos/55229469@N07/17202535346/) |
| `tulips/6948239566_0ac0a124ee_n.jpg` | val | irrelevant image or non-photographic depiction | Printed mascot design sheet; no flower. Original title: Kurogawa Onsen. | [原作者](https://www.flickr.com/photos/14541393@N03/6948239566/) |
| `tulips/7070694881_e9a331fa4e_n.jpg` | val | irrelevant image or non-photographic depiction | Tree trunks and canopy; no visible tulip. Original Flickr oEmbed returned 404; decision is based on the local image. | [原作者](https://www.flickr.com/photos/seanfx/7070694881/) |
| `roses/2408236801_f43c6bcff2.jpg` | val | irrelevant image or non-photographic depiction | Hanging teacup ornaments; no real flower. Original title: Tea Room. | [原作者](https://www.flickr.com/photos/sherrysrosecottage/2408236801/) |
| `roses/7211616670_2d49ecb3a5_m.jpg` | val | irrelevant image or non-photographic depiction | Cupcake tower; flower-like icing decorations. Original title: Love and lace wedding cupcake tower. | [原作者](https://www.flickr.com/photos/kristinausk/7211616670/) |
| `tulips/5687705933_55a8c2dbac.jpg` | test | irrelevant image or non-photographic depiction | Tree branches viewed through a window; no visible tulip. Original title: Botanic garden - Cluj-Napoca. | [原作者](https://www.flickr.com/photos/bortescristian/5687705933/) |
| `tulips/15275190769_0ed7bbf490.jpg` | test | irrelevant image or non-photographic depiction | Green wooden building and bicycle; no visible tulip. Original title: Holland. | [原作者](https://www.flickr.com/photos/phalaenopsisaphrodite523/15275190769/) |
| `sunflowers/418056361_1dfac1c151_n.jpg` | test | irrelevant image or non-photographic depiction | Photograph of a sunflower painting, not a living flower; exclude under the existing real-flower-photo policy. | [原作者](https://www.flickr.com/photos/giopuo/418056361/) |

## 已复核保留样本

| 路径 | split | 问题 | 依据 | 来源 |
|---|---|---|---|---|
| `daisy/5700781400_65761f3fce.jpg` | train | confirmed target or seed-head present | A white ray-flowered head with a yellow centre is clearly visible; original title: Daisy on the rocks. | [原作者](https://www.flickr.com/photos/viatorius/5700781400/) |
| `dandelion/14648777167_1d92d403c9_n.jpg` | train | confirmed target or seed-head present | Yellow flowers are visible throughout the field; original title: Blinking Dandelions. The previous no-flower rationale is unsupported. | [原作者](https://www.flickr.com/photos/winterpearl/14648777167/) |
| `sunflowers/9381481549_5a5d503e42_n.jpg` | train | confirmed target or seed-head present | A small yellow flower head is visible on the right; original title: Sunflowers Wk 7. The previous no-bloom rationale is unsupported. | [原作者](https://www.flickr.com/photos/myeral/9381481549/) |
| `roses/2568105249_15720d081f_n.jpg` | train | normal rose sample | White rose arrangement visible in the room; author tags include roses and white. User confirmed this as a normal rose sample; retain without a doubtful flag. | [原作者](https://www.flickr.com/photos/sherrysrosecottage/2568105249/) |
| `dandelion/4573204385_9b71e96b35_m.jpg` | train | confirmed target or seed-head present | Macro photograph of seed parachutes with water drops, not grass blades; consistent with seed-head photographs retained elsewhere. Original title: Liquid Sun Seed. | [原作者](https://www.flickr.com/photos/ladydragonflyherworld/4573204385/) |
| `tulips/4546299243_23cd58eb43.jpg` | train | reviewed: target present in mixed garden scene | Larger author image reveals orange cup-shaped tulips in the lower-left flower bed under the tree. Original author keywords include tulips. Keep original label; earlier cherry-blossom-only description was incomplete. | [原作者](https://www.flickr.com/photos/andrea_44/4546299243/) |
| `roses/475936554_a2b38aaa8e.jpg` | test | reviewed: identifiable vegetative rose parts in scope | Original title Rose Thorns and author tags roses, leaves and stem agree with the local rose stem/leaf photograph. Retain as a vegetative plant detail, consistent with allowing non-bloom stages; authenticity of a flower is not required for identifiable plant parts. | [原作者](https://www.flickr.com/photos/peasap/475936554/) |

## 待复核样本（当前保留）

| 路径 | split | 问题 | 依据 | 来源 |
|---|---|---|---|---|
