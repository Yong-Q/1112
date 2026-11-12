# 501帧独立核查证据

基线输入来自仓库提交`d89ba5f`。本目录是独立复算／诊断输出，不覆盖原始结果，不是合格的目标势c₁/c₂训练标签。

|文件|说明|
|---|---|
|energy_ess_audit.json|501帧ESS、权重集中、温度、源引擎能量差、预定时间截断敏感性及输入哈希|
|independent_weights.csv|全部501个独立复算权重；CSV使用标准CRLF行尾，不更改数值精度|
|spatial_audit.json|完整轨迹的O位点空间密度与RDF复算比较|
|corrected_rdf_diagnostic.json|修复配对计数后的RDF诊断；低ESS、有限尺寸与非均匀性仍存在|
|regression_tests.json|修复后15个无参数测试函数逐项直接执行通过；不代表完整生产统计验收|
|ESS_and_density.pdf/png|所有帧权重与z向O位点密度；无独立目标MD参照，不给出虚构置信区间|

能量脚本输入哈希保存的是修复前已归档版本；后续本地RDF修复不改变输入能量表和ESS。O位点密度不是完整分子COM／取向标签。空间不均匀性与液体薄层、低密度区相符，本图不单独证明相平衡。

数值复现命令见[最新TODO](../../TODO_OFF_POLAR_2026-10-27.md)。移植后的两个公开复算脚本已通过Slurm109541，分别重现能量与结构结果。图可用以下命令输出到新目录：

```bash
python scripts/report/make_off_polar_audit_figure.py --outdir runs/audit501-figure-new
```

模型权重、长程实现和源引擎力的一致性没有在本次重新推理，尚需完成TODO中的单点核查。
