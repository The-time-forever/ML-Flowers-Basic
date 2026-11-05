# 增强实验对比

| experiment | best ep | val acc | test acc | macro F1 | clean train | roses F1 (val) | tulips F1 (val) | roses→tulips | tulips→roses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exp_convnext | 5 | 0.9492 | 0.9673 | 0.9665 | 0.9984 | 0.898 | 0.914 | 9 | 7 |
| exp_efficientnet | 5 | 0.9456 | 0.9619 | 0.9615 | 0.9984 | 0.904 | 0.923 | 9 | 5 |
| exp_mobilenet | 7 | 0.9256 | 0.9256 | 0.9239 | 0.9980 | 0.882 | 0.888 | 4 | 9 |
| exp_residual_se | 67 | 0.8911 | 0.8639 | 0.8597 | 0.9809 | 0.830 | 0.854 | 10 | 10 |

roses/tulips 的 F1 与混淆数来自验证集（`analyze_errors.py`）；test 指标仅供最终参考。clean train 为最佳 epoch 的无增强训练集准确率。
