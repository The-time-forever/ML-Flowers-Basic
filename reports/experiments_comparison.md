# 增强实验对比

| experiment | best ep | val acc | test acc | macro F1 | clean train | roses F1 (val) | tulips F1 (val) | roses→tulips | tulips→roses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cnn_baseline_80 | 75 | 0.8766 | 0.8476 | 0.8451 | 0.9587 | 0.826 | 0.831 | 8 | 13 |
| exp_crop08 | 77 | 0.8693 | 0.8476 | 0.8456 | 0.9583 | 0.786 | 0.835 | 9 | 12 |
| exp_hue0 | 69 | 0.8784 | 0.8566 | 0.8525 | 0.9723 | 0.847 | 0.849 | 9 | 12 |
| exp_rot20 | 76 | 0.8711 | 0.8512 | 0.8488 | 0.9411 | 0.836 | 0.817 | 6 | 14 |

roses/tulips 的 F1 与混淆数来自验证集（`analyze_errors.py`）；test 指标仅供最终参考。clean train 为最佳 epoch 的无增强训练集准确率。
