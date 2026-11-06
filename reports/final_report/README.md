# 论文：花间识微

- `main.pdf`：编译后的单栏论文（20 页，含中英文摘要、关键词、14 幅图、13 张表与 3 个附录）
- `main.tex`：LaTeX 源码（XeLaTeX + ctex + pgfplots + TikZ）
- 正文结构：引言 → 模型设计 → 数据与实验设置 → 实验结果与分析 → 结论
- `figures/*.dat`：矢量图使用的数值表（训练曲线、短边累积分布、逐类 F1、混淆矩阵、标签审计统计）
- `figures/preprocess_examples.jpg`、`figures/label_noise_examples.jpg`、`figures/architecture_input.jpg`：真实图像拼图与核心图输入示例
- `figures/` 中的 `.dat` 与两张拼图均可由 `python src/make_report_figures.py` 重新生成

论文正文不包含本地路径或工程文件名；实验配置、超参数与完整实验结果见附录。

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

