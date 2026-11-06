# 设计还是迁移？小规模花卉分类中网络设计与预训练集成的再审视

- `main.pdf`：最终报告（17 页，含中英文摘要、关键词、11 幅图、13 张表和 5 个附录）
- `main.tex`：LaTeX 源码（XeLaTeX + ctex + pgfplots + TikZ）
- 正文结构：引言 → 数据集与数据处理 → CNN 基线与模型改进 → 基于 ImageNet 预训练的花卉分类 → 双模型集成分类 → 标签清洗与误差分析 → 讨论与局限 → 结论与展望
- 附录结构：A 预处理与数据增强示例，B 实验环境与训练配置，C 数据增强对照结果，D 移除图像清单，E 双模型集成的逐类指标
- 第 6 节分别比较同一集成模型在测试集清洗前后的表现，以及用原训练集和清洗后训练集训练的集成模型在同一测试集上的表现。
- `figures/*.dat`：训练曲线、短边累积分布、逐类 F1、混淆矩阵等图表所用的数据。
- `figures/preprocess_examples.jpg`、`figures/label_noise_examples.jpg`、`figures/architecture_input.jpg`：真实图像拼图与核心图输入示例
- `figures/` 中的 `.dat` 与两张拼图均可由 `python src/make_report_figures.py` 重新生成

重新编译：

```powershell
cd reports/final_report
latexmk -xelatex -interaction=nonstopmode main.tex
```

若只更新图表数据，先在仓库根目录运行：

```powershell
python src/make_report_figures.py
```

依赖：TeX Live 2025（ctex、pgfplots、booktabs、TikZ），Windows 中文字体（SimSun / 宋体）。报告中的数字来自 `reports/` 下的实验记录与 `reports/tables/final_ensemble_metrics.json`；清洗决策与证据见 `reports/label_audit/`。检查点保存在 `runs/`，不纳入 Git。
