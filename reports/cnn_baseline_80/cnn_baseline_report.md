# CNN Baseline 训练报告

- 模型：`SimpleCNN`（4 个 conv block，Global Average Pooling，**1,174,501** 可训练参数）
- 输入：224×224，ImageNet 归一化；训练增强：RandomResizedCrop(0.65-1.0)、水平翻转、ColorJitter
- 优化：AdamW lr=0.001, weight_decay=0.0001, cosine schedule, label smoothing=0.1, AMP=True
- 训练：80 epochs（early stopping patience=15），batch size=32，seed=42
- 诊断指标：每轮训练后用**验证预处理**（无增强、dropout off）再评估一次训练集，得到可与 val 直接比较的 `train_eval_acc`

## 结果

| 指标 | 数值 |
|---|---:|
| 最佳 epoch | 75 |
| 最佳验证准确率 | 0.8766 |
| 最佳 epoch 训练集准确率（无增强） | 0.9587 |
| 验证 - 训练差距 | -0.0821 |
| 测试准确率 | **0.8475** |
| 测试 Macro-F1 | **0.8451** |
| 测试错误数 | 84 / 551 |

**拟合诊断**：验证集比无增强训练集低 8.2 个百分点，存在一定过拟合；可加强增强、提高 weight decay、使用 EMA 或增加数据。

## 各类别指标（test）

| 类别 | precision | recall | f1-score | support |
|---|---:|---:|---:|---:|
| daisy | 0.820 | 0.863 | 0.841 | 95 |
| dandelion | 0.862 | 0.926 | 0.893 | 135 |
| roses | 0.762 | 0.802 | 0.782 | 96 |
| sunflowers | 0.958 | 0.867 | 0.910 | 105 |
| tulips | 0.836 | 0.767 | 0.800 | 120 |
| **macro avg** | **0.848** | **0.845** | **0.845** | **551** |

## 训练曲线

![训练曲线](figures/10_training_curves.png)

## 混淆矩阵

![混淆矩阵](figures/11_confusion_matrix.png)

## 高置信度错误样本

![错误样本](figures/12_misclassified.png)

## 说明

- 这是**从零训练**的小型 CNN，不使用预训练权重，用于建立基线。
- 若验证/测试差距较大，说明过拟合，可加强增强、加大 weight decay 或改用预训练模型。
- 混淆矩阵可用于判断类别间的视觉相似性（如 daisy/tulips、dandelion/sunflowers）。
