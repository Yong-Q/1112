# 参考论文

三篇核心参考，对应"从关联函数/密度场学习 c1、c2"的三条路线。

| 文件 | 论文 | 路线 |
|---|---|---|
| `PRL_2025_Dijkman_pair-correlation.pdf` | Dijkman et al., **Phys. Rev. Lett. 134, 056103 (2025)** — *Learning neural free-energy functionals with pair-correlation matching*（arXiv:2403.15007） | 均相 bulk 的 **g(r) 反演 c2**（pair-correlation matching），无外场、无 c1 场 |
| `PNAS_2026_Bui-Cox_mlip-neural-dft.pdf` | Bui & Cox, **PNAS (2026)** — *A unified machine learning framework for ab initio multiscale modeling of liquids*（arXiv:2603.20493） | 外场/受限下的**密度剖面监督 c1[n]**，c2 由自动微分得到 |
| `JCTC_2025_Yang_operator-learning-cDFT.pdf` | Yang, Pan, Sun, Wu, **J. Chem. Theory Comput. 21, 5905–5915 (2025)** — *High-Dimensional Operator Learning for Molecular Density Functional Theory*（DOI 10.1021/acs.jctc.5c00484；arXiv:2411.03698） | 3D 密度场的**高维算子学习 c1**（球谐展开表示），c1 单独监督 |

## 与本项目的关系

- **PRL**：本次密度扫描（T = 1.55 Tc、ρ/ρc = 0.2–2.2、六物种各 10 点）就是按它的均相 bulk 口径设计的，产出 c2(r; ρ, T)。
- **PNAS**：如果要训 c1(ξ(R)) 泛函，需要补"外场/受限非均相剖面"数据；c2 建议由 c1 自动微分得到，保证变分一致。
- **JCTC**：给了 3D 密度场算子表示的具体做法（球谐展开 + 高维算子），可作为 c1 网络的架构参考。

## 说明

- 这里放的是出版社版本 PDF，仅供组内阅读。本仓库若公开，建议改用 arXiv/OA 版本：
  PRL arXiv:2403.15007、JCTC arXiv:2411.03698、PNAS arXiv:2603.20493。
- 补充材料（SI）未收入，原文件在
  `bza/01_PRL-Dijkman-neural-free-energy/paper/PRL-134-056103-Supplemental.pdf`、
  `bza/02_JCTC-Yang-operator-learning-cDFT/paper/ct5c00484_si_001.pdf`、
  `bza/03_PNAS-Bui-Cox-mlip-neural-dft/paper/pnas.2610049123.sapp.pdf`。
