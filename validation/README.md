# 引擎安装验证（新机器照这个对）

这个目录用于**在新机器上装完 MACE-LES / Symmetrix 后，快速判断装对了没有**：
同一批坐标，PyTorch 参考实现 vs 引擎（`pair_style symmetrix/mace/float32/kk`）的
**总势能**和**每个原子的力**必须对上。

## 1. 直接可用的数据

| 文件 | 内容 |
|---|---|
| `H2O_50frames_engine_vs_pytorch.npz` | 2001 原子水盒的 50 帧：坐标、引擎能量/力、PyTorch 能量/力 |
| `summary_H2O.json` | 上面这套数据的偏差统计 |
| `existing/MACELES_ENGINE_VALIDATION_2026-10-30.md` | 1152 原子水的引擎 vs PyTorch 验证记录 |
| `existing/MACELES_FORCE_FIX_2026-10-30.md` | direct/JIT 路径 LES 电荷符号 bug 的定位与修复 |
| `existing/MACELES_Q4_VERIFY_2026-10-30.md` | 四极/电荷读出验证 |
| `existing/water1152_500ps/` | 1152 原子水 500 ps 生产跑：初始构型、501 帧总能量、输入、日志 |

## 2. 本次 50 帧对照的结果（标准答案）

2001 原子（667 H₂O），T = 986.4 K，ρ = 0.683 g/cm³，50 帧：

| 量 | 结果 |
|---|---|
| 能量最大偏差 | **0.00075 eV**（整盒） |
| 能量 RMS | 0.00046 eV |
| 每原子平均偏差 | **2.2e-4 meV/atom** |
| 力最大偏差 | **0.00099 eV/Å** |
| 力 MAE | 6.3e-5 eV/Å |
| 力相关系数 | **0.99999999** |

引擎用 float32、参考用 float64，这个量级的残差就是 float32 舍入，不能再小了。

## 3. 在新机器上怎么验

```bash
# (1) 装好引擎后，用同一批坐标跑单点，输出能量和力
#     in.check 里 read_data 用 existing/water1152_500ps/data.lammps 或你自己的构型
lmp -k on t 1 -in in.check          # thermo_modify format float %.12g
                                    # dump ... id fx fy fz + format %.10g

# (2) 用 PyTorch 参考算同样的构型
export PYTHONPATH=<mace-fork>:<les-fork>/src
python scripts/maceles_reference.py --model assets/maceles_off/MACELES-OFF_small.model \
    --eval frame.xyz --dtype float64

# (3) 对比
python scripts/check_engine.py 你的.npz
```

`scripts/check_engine.py` 会打印同样的四项统计，并给出是否通过的判定：

| 判据 | 阈值 |
|---|---|
| 能量 RMS | < 1e-2 eV |
| 力 MAE | < 1e-3 eV/Å |
| 力相关系数 | > 0.9999 |

超过阈值说明装错了（常见原因：LES patch 没打、用了 `main` 分支而不是 `polar-les`、
cubin 与实际架构不匹配、`les_norm_factor` 没导出）。

## 4. 参考数据的生成方式

`npz` 里的 `energy_engine` / `forces_engine` 来自 LAMMPS `rerun`（`thermo_modify format float %.12g`、
`dump_modify format float %.10g`）；`energy_pytorch` / `forces_pytorch` 来自
`MACELES-OFF_small.model`（TorchScript，float64，CPU）。

坐标取自 `H2O-T986.4-r00-rho0.6700` 轨迹每 40 帧取一帧（0, 40, …, 1960）。

## 5. 注意

- 引擎输出的是**整盒总势能**（extensive，eV），不是每原子平均。
- 力单位 eV/Å；dump 里写 `.10g` 才能保住精度。
- float32 引擎与 float64 参考的差异上限就是上表；若你的新机器偏差明显更大，先查 LES 是否真的启用
  （`libsymmetrix/source/les_ewald.hpp` 是否存在、`model_type` 是否为 `MACE_LES`）。
