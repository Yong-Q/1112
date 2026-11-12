# 重加权方差与详细推导（2026-10-27，独立新增）

> **历史版本入口。** 当前请读[修订PDF](REWEIGHT_VARIANCE_REVISED_2026-10-27.pdf)及[报告索引](README.md)。本页保留501帧复算数值；旧PDF未改动，旧LaTeX文件已作为修订源码的兼容入口。

- [6 页 PDF](REWEIGHT_VARIANCE_2026-10-27.pdf)
- [LaTeX 源码](REWEIGHT_VARIANCE_2026-10-27.tex)
- [完整复算结果](../results/reweight_variance_20261027.json)
- [数值复算程序](../../scripts/report/reweight_variance_20261027.py)

原有三份 PDF 未修改。此报告针对上传的 501 帧 / 1152 原子数据，补充总体与样本定义、常数偏移、相对方差与 ESS 的关系，以及相关轨迹和 g/c1/c2 的适用条件。不能将此前报告中的早期 6 帧结果与这里的 501 帧结果混用。

## 核心数值

全部经验方差采用分母 n，n = 501，T = 300 K：

| 定义 | 实测值 |
|---|---:|
| CV²(q) = Var(q)/mean(q)² | 212.311856 |
| Var(R), R_i = n w_i，mean(R)=1 | 212.311856 |
| Var(w), sum(w)=1 | 0.000845860599 |
| Var(a), a_i = exp[-β(ΔU_i−min ΔU)] | 0.002381360301 |
| Kish ESS | 2.348673951 |
| ESS/n | 0.00468797196 |

原始 q 的绝对方差依赖能量参考常数，表中 a 明确采用最小能量差为参考。相对方差不受整体参考能平移影响。这些是经验权重诊断，不是总体方差或物理观测量的误差条。

## 复现与保留结果

Python 程序只用标准库，读取原始 CSV，并以 `open('x')` 写入 JSON 与数值 TeX 文件：已有文件时主动拒绝覆盖。若需重新运行，应先复制整个工作目录并在副本内另设输出路径，保留原始结果。

当前请通过 `validation/make_reweight_variance_pdf.py` 构建修订版；不要以默认旧文件名直接编译旧入口，避免覆盖历史 PDF。原数值源文件为源码引用 `docs/results/reweight_variance_20261027_numbers.tex`。当前字体为 `/home/qiuyong/fonts/arial.ttf` 及 `1611458310630572.ttf`（微软雅黑）；其他服务器请在源码副本中改为当地可用的拉丁和 CJK 字体。将 `-output-directory` 指向新的构建目录，避免覆盖已有 PDF。

本次数值复算 Slurm 109819 成功；其中 PDF 阶段因节点缺少 fontconfig 失败。CPU 排版改放 gpu 分区（只申请 1 CPU，未申请 GPU），109821 因字体名解析失败；改用明确的字体文件路径后，109823 成功生成 6 页 PDF。失败日志保留在本地 audits 目录。旧 PDF 的 SHA256 前后全部一致。

文献 DOI 与开放全文链接见 PDF 末尾。当前无需额外下载论文；若进一步核查补充材料，另列缺失文件。
