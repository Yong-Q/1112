# 首页诊断图 B：增加经典 SPC 水参考

在原有 OFF23 与重加权估计曲线上，叠加绿色 `SPC/PPPM time average` 曲线。三条曲线均使用氧原子位点密度、384 个水、300 K、同一周期盒和40个 z 分箱，采用各自全部501个生产帧（0–500 ps）的时间等权平均；未平滑、未按分层中心平移。

SPC采用标准三点参数与PPPM长程静电，不是SPC/E。SPC先100 ps平衡再500 ps生产；OFF23原先20 ps平衡再500 ps生产。它们是相同N/V/T的不同势函数参考，经典线不能当作POLAR真实密度。橙色虚线仍由同一OFF轨迹做目标重加权得到。

- [PNG预览](key_diagnostics_with_SPC.png) · [高清PDF](key_diagnostics_with_SPC.pdf) · [可编辑SVG](key_diagnostics_with_SPC.svg)
- [SPC氧位点密度CSV](SPC_O_z_density.csv) · [定义与原始轨迹哈希](metadata.json)
- [SPC参数、输入和长轨迹实耗](https://github.com/starlessinred-interns/qiuyong/tree/main/report/homogeneous_sampling_20261029_v4/evidence/spc_water_reference_20261029)

SPC曲线沿盒子积分为384；新图峰值约0.03302 Å⁻³。该数使用氧位点和全部501帧，与报告中质心、t≥20 ps的峰值口径不同，不能混用。

从随仓小CSV重画，不需要原始大轨迹或新MD：

```bash
python scripts/report/make_diagnostic_with_spc.py \
  --spc-profile docs/results/20261029_spc_readme_overlay/SPC_O_z_density.csv \
  --outdir runs/readme-spc-new
```

原图和数据保留在历史目录；首页指向此更新版本。
