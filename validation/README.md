# 引擎安装验证（新机器照这个对）

这个目录用于**在新机器上装完 MACE-LES / Symmetrix 后，快速判断装对了没有**：
同一批坐标，PyTorch 参考实现 vs 引擎（`pair_style symmetrix/mace/float32/kk`）的
**总势能**和**每个原子的力**必须对上。

## 1. 直接可用的数据

| 文件 | 内容 |
|---|---|
| `H2O/NH3/CO2/C2H6_50frames_engine_vs_pytorch.npz` | 四个物种各 50 帧：坐标、引擎能量/力、PyTorch 能量/力 |
| `summary.json` | 四个物种的每原子偏差统计 |
| `existing/MACELES_ENGINE_VALIDATION_2026-10-30.md` | 1152 原子水的引擎 vs PyTorch 验证记录 |
| `existing/MACELES_FORCE_FIX_2026-10-30.md` | direct/JIT 路径 LES 电荷符号 bug 的定位与修复 |
| `existing/MACELES_Q4_VERIFY_2026-10-30.md` | 四极/电荷读出验证 |
| `existing/water1152_500ps/` | 1152 原子水 500 ps 生产跑：初始构型、501 帧总能量、输入、日志 |

## 2. 50 帧对照结果（标准答案，**按每个原子**）

| 物种 | 原子数 | ΔE RMS | ΔE 最大 | ΔF MAE | ΔF 最大 | 力相关系数 |
|---|---|---|---|---|---|---|
| H2O | 2001 | **2.3e-4 meV/atom** | 3.8e-4 | **0.063 meV/Å** | 1.00 | 0.999999997 |
| NH3 | 2000 | **1.8e-3 meV/atom** | 1.9e-3 | **0.025 meV/Å** | 0.78 | 0.999999999 |
| CO2 | 2001 | **5.6e-3 meV/atom** | 5.8e-3 | **0.15 meV/Å** | 1.30 | 0.999999984 |
| C2H6 | 4000 | **7.4e-4 meV/atom** | 8.2e-4 | **0.018 meV/Å** | 0.27 | 1.000000000 |

参照量级：室温 kT ≈ 25 meV/atom，上述能量误差是 kT 的 **1e-5 ~ 2e-4**。
引擎 float32、参考 float64，这个残差就是浮点舍入的下限；CO2 偏大是因为 C=O 键力大，float32 相对误差被放大。

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
