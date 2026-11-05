# 标签清洗记录

- 标准：图片是否包含符合类别定义的真实花卉；
- 处理：无任何花卉/明确无关的图片排除；花卉种类存疑的保留。

## 汇总

| split | 原始 | 排除 | 清洗后 |
|---|---:|---:|---:|
| train | 2564 | 35 | 2529 |
| val | 551 | 4 | 547 |
| test | 551 | 3 | 548 |
| **合计** | **3666** | **42** | **3624** |

## 已排除样本

| 路径 | split | 问题 | 依据 |
|---|---|---|---|
| `roses/8462246855_1bdfee7478.jpg` | val | non-flower: vehicle | photo of a white Hummer; no rose |
| `roses/17953368844_be3d18cf30_m.jpg` | val | non-flower: building | gothic ceiling vault; no rose |
| `roses/15419696882_9394168a10_n.jpg` | val | non-flower: landscape | sunset over water; no rose |
| `roses/4256169180_55df2048a0.jpg` | val | non-flower: person/event | rodeo performer on horseback; no rose |
| `roses/12323085443_8ac0cdb713_n.jpg` | test | non-flower: person/event | horse-drawn parade carriage; no rose |
| `tulips/16862422576_5226e8d1d0.jpg` | test | non-flower: building/landscape | Gardens by the Bay Supertrees; no tulip |
| `tulips/15275199229_962387f24d.jpg` | test | non-flower: building/landscape | windmills on water; no tulip |
| `daisy/5700781400_65761f3fce.jpg` | train | non-flower: object | tree bark texture; no flower |
| `sunflowers/7176729016_d73ff2211e.jpg` | train | non-flower: person/event | outdoor market stalls; no sunflower |
| `tulips/17907238905_1ae121f8d9_m.jpg` | train | non-flower: building | war memorial; no tulip |
| `roses/5148639829_781eb7d346.jpg` | train | non-flower: landscape | field/fence landscape; no rose |
| `tulips/3511776685_3635087b12_n.jpg` | train | non-flower: food | strawberries; no tulip |
| `tulips/4522153453_06437ca3af_m.jpg` | train | non-flower: person/event | person painting outdoors; no tulip |
| `tulips/14116780333_7836f4448c.jpg` | train | non-flower: object | water fountain; no tulip |
| `tulips/6325571510_7544b27e57_n.jpg` | train | non-flower: object | teacup and biscuits; no tulip |
| `tulips/15275144259_f9a18ec9cb.jpg` | train | non-flower: building | steel arch bridge; no tulip |
| `roses/3921794817_276eb4386b.jpg` | train | non-flower: interior | bedroom interior; no rose |
| `roses/2863863372_605e29c03e_m.jpg` | train | non-flower: building | brick pub building; no rose |
| `dandelion/14648777167_1d92d403c9_n.jpg` | train | non-flower: landscape | green field and tree; no dandelion |
| `roses/15750320284_22ef21c682.jpg` | train | non-flower: person/event | marching band; no rose |
| `daisy/1355787476_32e9f2a30b.jpg` | train | non-flower: object | hand holding a water drop; no daisy |
| `roses/12450781274_eb78723921.jpg` | train | non-flower: vehicle | tram street scene; no rose |
| `roses/4292443009_3a2831b0b9_m.jpg` | train | non-flower: person/event | rodeo performer; no rose |
| `roses/12045735155_42547ce4e9_n.jpg` | train | non-flower: person/event | marching band; no rose |
| `roses/3655527028_0fab2b547d_n.jpg` | train | non-flower: interior/object | decorative china cabinet; no living rose |
| `tulips/14097328354_4f1469a170.jpg` | train | non-flower: person/event | person with camera; no tulip |
| `tulips/13910126337_53faf1d214_n.jpg` | train | non-flower: landscape | branches over water; no tulip |
| `roses/9406573080_60eab9278e_n.jpg` | train | non-flower: person/event | crowd at a festival; no rose |
| `roses/2364976562_a184463083_m.jpg` | train | non-flower: object | dolls; no rose |
| `tulips/15275504998_ca9eb82998.jpg` | train | non-flower: building/object | window and bicycle; no tulip |
| `tulips/13910131718_731353d84c_n.jpg` | train | non-flower: person/event | black-and-white people; no tulip |
| `roses/5193918046_d44e4fcd75_m.jpg` | train | non-flower: decoration | holiday candles/decor; no rose |
| `roses/180613732_3a7aba0b80_n.jpg` | train | non-flower: landscape/animal | forest and ducks; no rose |
| `roses/319298955_0c72bd36bf.jpg` | train | non-flower: food | chocolate cake with decorative roses; no real flower |
| `tulips/7064813645_f7f48fb527.jpg` | train | non-flower: vehicle/field | tractor in a field; no tulip |
| `sunflowers/9381481549_5a5d503e42_n.jpg` | train | non-flower: plants | green seedlings, no sunflower bloom |
| `roses/16334786972_1b3e71cab8_m.jpg` | train | non-flower: illustration/decoration | vintage seed catalogue page; no real rose |
| `roses/2568105249_15720d081f_n.jpg` | train | non-flower: interior | interior room with wedding decor; no rose |
| `tulips/14487705209_ea723109e1_m.jpg` | train | non-flower: illustration/decoration | tulip illustration; no real flower |
| `dandelion/4573204385_9b71e96b35_m.jpg` | train | non-flower: plants | grass blades with dew; no dandelion |
| `roses/6069602140_866eecf7c2_m.jpg` | train | non-flower: illustration/decoration | tattoo artwork; no real rose |
| `sunflowers/7270375648_79f0caef42_n.jpg` | train | non-flower: light art | abstract sunflower-shaped light installation; no real flower |

## 保留的物种存疑样本

| 路径 | split | 问题 | 依据 |
|---|---|---|---|
| `roses/8775267816_726ddc6d92_n.jpg` | val | species mismatch (flower present) | pink gerbera-like flower; species doubtful, flower present |
| `tulips/13513644515_a51470b899.jpg` | val | species mismatch (flower present) | daffodil field; species doubtful, flowers present |
| `tulips/13513851673_9d813dc7b0.jpg` | test | species mismatch (flower present) | hyacinth field with barn; flowers present, species doubtful |
| `roses/15174615529_144ae28bdb_n.jpg` | train | species mismatch (flower present) | portulaca-like flower; species doubtful, flower present |
| `roses/5212885371_fe27c406a2_n.jpg` | train | species mismatch (flower present) | orange gerbera-like flower; species doubtful, flower present |
| `tulips/4546299243_23cd58eb43.jpg` | train | species mismatch (flower present) | cherry blossom path; flowers present, species doubtful |
| `tulips/5674707464_dc18de05b1.jpg` | train | species mismatch (flower present) | lavender field; flowers present, species doubtful |

完整机器可读清单见 `cleaning_list.csv`；清洗后的划分在 `splits_clean/`。
