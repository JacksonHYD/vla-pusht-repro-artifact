# 论文源文件

`main.tex` + `figs/*.pdf` + 编译好的 `main.pdf`。

## 怎么编译

本地（MiKTeX 或 TeX Live 都行），连编两遍让交叉引用正确：

```bash
pdflatex main.tex && pdflatex main.tex
```

或者在 Overleaf：New Project → Upload，**整个 `paper/` 目录一起传（含 `figs/`）**，然后 Recompile。

依赖包全是 TeX Live 自带：`geometry`、`amsmath`、`amssymb`、`graphicx`、`caption`、`booktabs`、`array`、`hyperref`、`xcolor`、`fontenc`、`newtxtext`、`newtxmath`。

## 当前版本

- 标题：**What a Success Rate Measures: Configuration, Compute and Sampling in a Robot Learning Benchmark**
- 10 页 / 4 图 / 3 表 / 18 条参考文献
- 统一口径：200 局、300 步、eval `seed=1000`、batch 22 同步环境

## 正在进行的更新

2×2 的每一格正在补到 **3 个训练种子**（`train_seed_block.sh` / `eval_seed_block.sh`）。
跑完后，摘要、表 1、表 4、图 2 和 §4.1 的效应量会用**种子均值**替换现在的单点估计
（例如 baseline 33.0% → 约 37%，crop 效应 +22.0 → 约 +22pp，GN 效应 −4.5 → 约 −11.8pp）。
届时 `main.tex` 与 `main.pdf` 会再同步一次，中间的版本请不要引用。