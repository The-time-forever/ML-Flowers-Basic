# 数据划分报告

- 数据集：`flower_photos`
- 可用图片：**3670** 张；使用 **3666** 张，剔除 **4** 张
- 划分比例：train / val / test = 70% / 15% / 15%
- 随机种子：`42`
- 近重复阈值：dHash 汉明距离 ≤ 4（仅用于同 split 分组，不删除）
- 重复簇：3652 个，其中多图簇 14 个；**跨 split 的簇：0 个**

## 各类别分布

| 类别 | train | val | test | 合计 |
|---|---:|---:|---:|---:|
| daisy | 443 | 95 | 95 | 633 |
| dandelion | 628 | 135 | 135 | 898 |
| roses | 448 | 96 | 96 | 640 |
| sunflowers | 487 | 105 | 105 | 697 |
| tulips | 558 | 120 | 120 | 798 |
| **合计** | **2564** | **551** | **551** | **3666** |
| 占比 | 69.9% | 15.0% | 15.0% | 100% |

每个类别的 val/test 数量按比例四舍五入，train 为剩余部分，因此各类别占比最大偏差很小。

## 去重与泄漏防护

- 完全重复（MD5 相同）：3 组，涉及 6 张；跨类别 1 组
  - `roses/15922772266_1167a06620.jpg`; `tulips/15922772266_1167a06620.jpg`
- 处理方式：完全重复每组只保留 1 张；跨类别冲突整组剔除
- 因标签冲突剔除 **2** 张；因完全重复剔除 **2** 张
- 近重复（非完全同一文件）：保留，14 个多图簇整体分入同一 split
- 校验结果：**0** 个簇跨越多个 split（应为 0）

### 已剔除的图片

| 路径 | 原因 |
|---|---|
| `roses/15922772266_1167a06620.jpg` | cross-class duplicate (label conflict) |
| `sunflowers/15066430311_fb57fa92b0_m.jpg` | exact duplicate of `sunflowers/14889392928_9742aed45b_m.jpg` |
| `sunflowers/15072973261_73e2912ef2_n.jpg` | exact duplicate of `sunflowers/15069459615_7e0fd61914_n.jpg` |
| `tulips/15922772266_1167a06620.jpg` | cross-class duplicate (label conflict) |

跨类别冲突样本无法确定唯一标签，直接剔除；完全重复只保留分辨率/文件最大的一张。

## 产物

- `train.csv` / `val.csv` / `test.csv`：划分清单（`path,label,split,cluster_id,cluster_size`）
- `summary.json`：机器可读的划分统计
- `split_report.md`：本报告
