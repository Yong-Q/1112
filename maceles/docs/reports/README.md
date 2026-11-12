# 重加权与c1/c2报告：当前阅读入口

## 六分子 MACELES 均相采样预算（2026-10-30）

[6页PDF](SIX_MOLECULE_MACELES_PRL_PLAN_20261030/report.pdf) · [密度清单与时间说明](SIX_MOLECULE_MACELES_PRL_PLAN_20261030/README.md)。每盒3000原子，每条1 ns；每种10点首批、20点扩展，共60/120条。包括PRL约化量核对、明确标注的真实分子换算假设、完整20×6密度表和逐条建盒CSV。MACELES正式192通道模型的修复后速度尚待补测，已与旧128通道试验模型的速度参照分开。该预算不采用旧OFF23或Plan D的吞吐，也没有启动MD。

## 当前结论（2026-10-27修订）

**给定501帧CSV的OFF→POLAR权重复算正确，但实际源势实现一致性及目标统计性质尚未验收。** 学生1152原子评分来自OFF23旧轨迹，不能判定学生自采样的重加权性能；撤回此前生产尺寸不可用的结论。高斯模型的10^41帧外推不是真实算量预测。

建议依次阅读：

1. **[12页扩展PDF：重加权估计器与实际观测量](REWEIGHT_ESTIMATOR_EXTENDED_2026-10-28.pdf)**：包括方差推导、每帧 A(x)、分子分母不同采样频率、分部积分的力密度表达式和验证条件。[扩展版LaTeX源码](REWEIGHT_ESTIMATOR_EXTENDED_2026-10-28.tex)。

   此为理论及计算方案；力采样和多频率估计器尚未运行。原[8页修订版](REWEIGHT_VARIANCE_REVISED_2026-10-27.pdf)保留。
2. **[七篇原文逐式核对](FORMULA_CROSSCHECK_PRIMARY_PDFS_2026-10-27.md)**：原PDF页码、公式区别和2019式(18)符号核验。
3. **[学生评分报告](../ESS_STUDENT_TO_POLAR_2026-10-27.md)**：保留原数值，区分OFF旧帧、学生短轨迹及池化验证集。
4. **[源势能量与空间审计](../results/20261027_off_polar_audit/README.md)**、[TODO](../TODO_OFF_POLAR_2026-10-27.md)：同帧能量一致性、分层、分子密度与c1/c2提取资格。
5. **[审计修订](REWEIGHT_VARIANCE_AUDIT_2026-10-27.md)**、[文献对照修订](REWEIGHT_VARIANCE_LITERATURE_2026-10-27.md)：撤回哪些旧判断、哪些数值仍成立。

## 三类未完成项

- 实际MD引擎与PyTorch OFF重算能量，去常数后差异标准差约0.030eV；尚未定位全部原因。
- 学生势需要来源匹配的平衡生产轨迹；19帧、0.2ps不能认证结构统计。
- 目标c1/c2需要独立目标参照、观测量误差及适用理论验证。解析OZ测试通过不等于分层真实水数据可直接套均匀体相OZ。

## 历史PDF：保留，不作为当前结论入口

以下文件保留原字节用于追溯，其中早期样本、温度或失败结论可能已被修订：

- [RESEARCH_REPORT.pdf](RESEARCH_REPORT.pdf)
- [EXPERIMENT_REPORT.pdf](EXPERIMENT_REPORT.pdf)
- [THEORY_REPORT.pdf](THEORY_REPORT.pdf)
- [原六页方差报告](REWEIGHT_VARIANCE_2026-10-27.pdf)
- [旧四组数据附录](../REWEIGHT_VARIANCE_DATA_2026-10-27.pdf)

旧版本不能单独用于引用“生产尺寸失败”“更多帧必然无效”“高斯模型确定真实所需帧数”或学生系综ESS等结论。修订版采用新PDF文件名，原PDF没有覆盖。

## 复现

[兼容构建入口](../../validation/make_reweight_variance_pdf.py)已更新为修订版XeLaTeX构建器，默认写新文件名，已有文件时拒绝覆盖。新源码为[REWEIGHT_VARIANCE_REVISED_2026-10-27.tex](REWEIGHT_VARIANCE_REVISED_2026-10-27.tex)，从仓库根目录读取保存的数值TeX。需要XeLaTeX及源码中配置的字体；请在计算节点运行。构建日志保留在本地audits目录，不上传环境缓存或作业脚本。

## 扩展版构建

从仓库根目录用 XeLaTeX 对 `REWEIGHT_ESTIMATOR_EXTENDED_2026-10-28.tex` 编译两遍；源码读取 `docs/results/reweight_variance_20261027_numbers.tex`。将 `-output-directory` 指向新的构建目录，保留历史PDF。源码配置了当前服务器的字体路径，其他服务器可在源码副本中换成本地可用的拉丁与CJK字体。
