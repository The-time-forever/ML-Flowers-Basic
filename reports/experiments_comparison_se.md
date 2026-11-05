# 增强实验对比

| experiment | best ep | val acc | test acc | macro F1 | clean train | roses F1 (val) | tulips F1 (val) | roses→tulips | tulips→roses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cnn_baseline_80 | 75 | 0.8766 | 0.8476 | 0.8451 | 0.9587 | 0.826 | 0.831 | 8 | 13 |
| exp_residual | 65 | 0.8802 | 0.8512 | 0.8494 | 0.9700 | 0.844 | 0.838 | 8 | 11 |
| exp_residual_deep | 31 | 0.8748 | 0.8548 | 0.8527 | 0.9501 | 0.824 | 0.804 | 8 | 18 |
| exp_residual_se | 67 | 0.8911 | 0.8639 | 0.8597 | 0.9809 | 0.830 | 0.854 | 10 | 10 |

roses/tulips 的 F1 与混淆数来自验证集（`analyze_errors.py`）；test 指标仅供最终参考。clean train 为最佳 epoch 的无增强训练集准确率。
