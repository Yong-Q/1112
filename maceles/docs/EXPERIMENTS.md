# 实验报告：MACE-OFF → MACE-POLAR 的重加权与 c1/c2

> **后续检查说明（2026-10-23）：** 原基线RDF体积错误已由远端865f203修复并重算部分结果；Widom分母、径向网格、温度约定及统计解释仍需验收。见[检查记录](AUDIT_2026-10-23.md)与[待完成计划](superpowers/plans/2026-10-23-polar-c1c2-validation.md)。新的RMS不能在剩余问题解决前当作认证标签精度。

**日期：** 2026-10-23  
**状态：** 探索性实验，**不是**生产级 c1/c2 结果。  
**结论先行：** 1,000+ 原子的直接重加权 **不可行**（ESS≈1）；96 原子也只是 marginal（ESS≈10%）。c2 的 OZ 反演本身已验证；问题在采样/重叠，不在公式。

---

## 1. 目的

判断能否用便宜的 MACE-OFF 轨迹，加上逐帧 MACE-POLAR 单点能量，通过 Boltzmann 重加权得到 MACE-POLAR 的：

- **c2(r)**：二体直接相关函数（由 \(g(r)\)+OZ）；
- **c1**：一体直接相关函数（均匀体相为常数；非均匀为剖面）。

同时记录当前流程的问题，供老师判断下一步。

## 2. 体系与设置

| 项目 | 设置 |
|---|---|
| 体系 | 水，32 / 384 个分子（96 / 1,152 原子） |
| 盒子 | 96 原子：11.696 Å；1,152 原子：23.39×23.39×35.09 Å |
| 密度 | 0.02 molecules/Å³（约 0.6 g/cm³） |
| 温度 | 300 K |
| 源模型 | **原始 MACE-OFF23 medium**（Symmetrix FP32 引擎 JSON + JIT，不是蒸馏学生） |
| 目标模型 | **MACE-POLAR-1-M**，FP64（PyTorch 单点） |
| 轨迹 | 96 原子 5 ps / 10 ps；1,152 原子 5 ps（6 帧，24 min；1 ns 未完成，500 ps 生产已启动） |
| 时间步 | 0.5 fs |
| 帧间隔 | 0.1 ps 或 1 ps |

> 注意：0.02 molecules/Å³ 在 300 K 接近水的两相区，不是稳定单相液体；这是当前实验的一个物理问题。

## 3. 结果

### 3.1 两力场单点差异（96 原子，81 帧/套，10 ps）

| 量 | MACE-OFF | MACE-POLAR |
|---|---:|---:|
| \(c_1^{\rm ex}\)（绝对值，未对齐） | 80504.257 | 80456.478 |
| \(c_1\) 差 | — | **−47.78**（与 \(-βΔE\) 一致） |
| \(c_2(r)\) 范围（修正后） | [−4.0, 3.1] | [−3.5, 3.1] |

- \(c_1\) 的绝对值被能量参考主导，必须对齐零点；差值 47.78 与 416.6 meV/atom 的偏移一致。
- 修正 RDF 归一化后，两套 \(c_2\) 很接近（RMS 0.20），96 原子/10 ps 的噪声和有限尺寸仍占主导，**不能当作真实结构差**。
- 旧版 \(c_2\) 范围 [−51.2, 226.2] / [−293.3, 1336.8] 是球壳体积归一化 bug 的伪影，已作废；见 §3.5。

### 3.2 重加权诊断（96 原子，36 帧）

| 量 | 数值 |
|---|---:|
| ΔE/atom 偏移 | 416.62 meV/atom |
| ΔE/atom 涨落 std | 1.166 meV/atom |
| \(\sigma_\beta=\mathrm{std}(\beta\Delta E)\) | **3.89** |
| ESS | **3.63 / 36 = 10.1%** |
| 判定 | **marginal** |
| 线性 slope / intercept | 0.9452 / −3606.84 eV |
| \(R^2\) / 残差 | 0.9866 / 1.03 meV/atom |
| 力 MAE | 104.7 meV/Å |

### 3.3 重加权 c2 vs 直接 MACE-POLAR c2（81 帧）

| 量 | 数值 |
|---|---:|
| \(\sigma_\beta\) | 3.31 |
| ESS | **2.09 / 81 = 2.6%** |
| c2 直接 POLAR vs OFF 的 RMS（修正后） | 0.196 |
| c2 直接 POLAR vs 重加权的 RMS（修正后） | 0.484 |
| 最大差（修正后） | 1.294 |

**重加权没有改善一致性，反而略差。** 但 ESS 只有 2.6%，这个比较仍受噪声限制，不能证明重加权理论错误——只说明当前数据不够。旧版的 RMS 115.8 / 121.0 来自 RDF 归一化 bug，已作废。

### 3.4 1,152 原子：5 ps / 6 帧早期检查

运行时间 **24 min 17 s**（06:08:28–06:32:45），10,000 步 × 0.5 fs = **5 ps**；dump 每 2,000 步（1 ps）一帧，共 **6 帧**。这段属于输入中的 `run 200000`（100 ps 平衡段），**不是生产段**；`run 2000000` 从未开始。

用 `scripts/monitor_ess.py` 实时读取 LAMMPS dump、逐帧算 MACE-POLAR 单点：

| step | ΔE/atom (meV) | \(\sigma_\beta\) | ESS |
|---:|---:|---:|---:|
| 0 | 410.254 | 0.00 | 1/1 |
| 2000 | 414.291 | 127.21 | 1/2 |
| 4000 | 414.857 | 111.87 | 1/3 |
| 6000 | 415.245 | 102.74 | 1/4 |
| 8000 | 415.837 | 98.98 | 1/5 |
| 10000 | 415.806 | 93.83 | 1/6 |

verdict：**collapse（早期判据）**。  
6 帧不足以给精确 ESS；但现有帧的 \(\beta\Delta V\) 跨度约 180 kT，ESS≈1 的量级不会因再加几帧翻转。按实测 6.9 步/s 外推，1 ns（2,000,000 步）约需 80 h，因此在 5 ps 处停止。**500 ps 生产运行已在后台启动**（见 §7）。

### 3.5 RDF 归一化修正（2026-10-23）

检查图表时发现 `OffPolarReweighter.rdf` 把球壳体积写成
\(4\pi/3\,(r_{\rm hi}^2-r_{\rm lo}^2)\)，正确应为
\(4\pi/3\,(r_{\rm hi}^3-r_{\rm lo}^3)\)。这使 \(g(r)\) 被放大约 \(3r/2\)
（\(r=2.85\) Å 时 4.3 倍），c2 的 OZ 反演随之失真。

| 量 | 修正前 | 修正后 |
|---|---:|---:|
| \(g_{OO}\) 峰值 | 16.2 | 3.8 |
| MACE-OFF c2 范围 | [−51.2, 226.2] | [−4.0, 3.1] |
| MACE-POLAR c2 范围 | [−293.3, 1336.8] | [−3.5, 3.1] |
| 直接 POLAR vs OFF c2 RMS | 115.8 | 0.196 |

修正后 c2 曲线物理合理（第一峰在 2.85 Å），两力场 c2 很接近；重加权
（ESS 2.6%）仍未改善。新增两个单测：理想气体 \(g=1\)、配位数与原始对数一致。
存储的 c2 JSON 由 `validation/repair_rdf_normalization.py` 解析修正
（直方图本身正确，g 是精确线性像）。

### 3.6 1,152 原子 500 ps 生产完成（2026-10-25）

500 ps 生产段已跑完：500,000 步 × 1 fs，wall time **19:54:04**，dump 每 1,000 步 = 1 ps，共 **501 帧**。

| 量 | 96 原子 / 36 帧 | **1,152 原子 / 501 帧** |
|---|---:|---:|
| ΔE/atom 均值（常数偏移） | 416.62 meV/atom | **417.42 meV/atom** |
| ΔE/atom 标准差 | 1.166 meV/atom | **0.309 meV/atom** |
| σ_β | 3.89 | **13.76** |
| ESS | 3.63 / 36 = 10.1 % | **2.35 / 501 = 0.47 %** |
| 判定 | marginal | **collapse** |
| slope / intercept | 0.9452 / −3606.84 eV | 0.9248 / −59650.78 eV |
| R² / 残差 | 0.9866 / 1.03 meV/atom | 0.9624 / 0.286 meV/atom |

**结论：500 ps 没有改善，反而给出比 96 原子更强的否定证据。**

1. 单原子涨落随体系增大而变小（1.166 → 0.309 meV/atom），但 σ_β 是广延量，涨了 3.5 倍（=√12，与 σ_β ∝ √N 预测的 1,296 原子 14.3 一致）。
2. 501 帧只剩 **2.35 个有效帧**。ESS 在固定权重宽度下会饱和，增加帧数不能救。
3. R² = 0.962 的线性拟合**没有帮助**：残差驱动的权重 σ_β 仍是 13.76，因为线性项在归一化后近似常数；slope = 0.925 主要在吸收参考能差，乘 0.925 等价于把温度改成 324 K。
4. 重加权后的 O–O RDF 只是给 OFF 的 RDF 加噪声，没有向 POLAR 结构偏移——2.35 个有效帧无法定义 80 个 bin 的 RDF。

完整记录（含四联图）：`../../symmetrix-polar/docs/results/REWEIGHT_500PS_1152_2026-10-25.md`。
诊断脚本：`../../symmetrix-polar/scripts/reweight_500ps_water1152.py`。

> 注意：该 500 ps 轨迹用 FP32 引擎（`symmetrix/mace/float32/kk`），源采样含 FP32 噪声。这不影响重加权判定（能量差在 FP64 下测），但该轨迹不宜作为 FP64 能量参考。

## 4. 已严谨验证的部分
| 验证 | 结果 |
|---|---|
| 光滑解析 c(r) 的 OZ 反演 | 最大误差 **4.9×10⁻¹³** |
| Percus–Yevick 硬球平滑段 | **2.1×10⁻¹³** |
| 理想气体 c2 | 0 |
| RDF 归一化 | 理想气体 g=1；配位数与原始对数一致 |
| 单元测试 | 14 项通过 |

所以 **OZ 公式和实现是对的**；RDF 归一化 bug 已修正；剩余问题在采样、有限尺寸和重叠。

## 5. 当前实验存在的问题

1. **盒子太小/状态点不对（96 原子）**：\(L=11.7\) Å < \(2r_{\rm cut}=12\) Å；0.02 密度在 300 K 接近两相区。
2. **轨迹太短（10 ps）**：\(g(r)\) 噪声大；修正后 c2≈[−4.8, 3.5]，旧值 [−293, 1337] 是归一化 bug 伪影。
3. **c1 绝对值不可直接比较**：被能量参考主导；Widom 只有 50/100 次插入被接受。
4. **1,152 原子只有 5 ps / 6 帧（24 min）**：\(\sigma_\beta\approx94\)，ESS=1 是早期判据；500 ps 生产运行进行中。
5. **重算能量不改变构型**：在 OFF 帧上算 POLAR 能量，\(g(r)\)/c2 仍是 OFF 的。
6. **引擎稳态约 6.3–6.9 steps/s**：10 步测试的 98 steps/s 是短程/异步计时假象；1 ns 约 80 h，500 ps 约 46 h。
7. **没有目标参照**：只有 OFF 轨迹，无法最终验证重加权 c2/c1 是否正确；500 ps 生产将提供同一体系的更长 OFF 轨迹。

## 6. 结论与建议

- **直接重加权不适合 1,000+ 原子的 MACE-OFF→MACE-POLAR**（依据 96 原子实测 + \(\sqrt N\) 标度；1,152 原子 500 ps 数据待完成）；ESS 太小。
- 96 原子只能做 **marginal** 的方法验证，不能作为生产。
- 若要 MACE-POLAR 的 c2/c1：
  1. **直接跑 MACE-POLAR** 短轨迹（100 ps；1,152 原子约 8.3 h）；
  2. 或训练 **ΔV 代理势**（能量+力）再跑 MD；
  3. 非均匀 c1(r) 需要外场下的 POLAR 轨迹 + Euler–Lagrange。
- 若要 PRL 式 c2 泛函：需要**多个密度**的 bulk c2，而不是一条轨迹重加权。

## 7. 复现命令

```bash
# 96 原子双力场 10 ps
python validation/run_forcefield_c1c2.py \
  --structure water96.xyz --off-model MACE-OFF23_medium.model \
  --polar-model MACE-POLAR-1-M.model --ps 10 --equil-ps 1 --frame-ps 0.1 \
  --n-widom 100 --outdir runs/ff_water96

# 重加权 vs 直接 c2
python validation/compare_reweighted_vs_direct.py

# 实时 ESS（任意 LAMMPS dump）
python scripts/monitor_ess.py --dump traj.lammpstrj \
  --off-model MACE-OFF23_medium.model --polar-model MACE-POLAR-1-M.model \
  --temperature 300 --outdir runs/ess_live

# 1,152 原子 500 ps 生产（后台，约 46 h；从 5 ps dump 帧重新加温，20 ps 平衡）
cd /workspace/group/shared/bza/symmetrix-polar/bench/off23_500ps_water1152_2026-10-23
env -u PYTHONPATH -u MATOS_POLAR_PROFILE \
  LD_LIBRARY_PATH=/workspace/caches/conda/envs/lammps-deepmd/lib:/opt/matos-tools/cuda/12.9/lib64:/opt/matos-tools/conda/envs/matos-compute/lib \
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  /workspace/group/shared/bza/external-repos/symmetrix-xl/symmetrix/build-kernel-r3/lmp \
  -k on g 1 -in in.production -log log.production > stdout.production 2>&1
```

## 8. 数据与图

- 96 原子重加权诊断：`examples/water96/reweight/reweight_diagnostics.png`
- 重加权 vs 直接 c2：`validation/results/forcefield_c1c2_water96/reweight_vs_direct/reweight_vs_direct.png`
- 1,152 原子 5 ps / 6 帧实时 ESS：`../../symmetrix-polar/bench/off23_1ns_water1152_2026-10-23/ess_live/ess_live.json`
- 1,152 原子 500 ps 生产（进行中）：`../../symmetrix-polar/bench/off23_500ps_water1152_2026-10-23/`
- 理论推导：`docs/THEORY.md`、`docs/reports/THEORY_REPORT.pdf`
- 公式+实验合并报告：`docs/reports/EXPERIMENT_REPORT.pdf`
