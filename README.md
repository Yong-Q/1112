# MACELES-LES 密度扫描：代码、权重与运行记录

本仓库是一套可直接复现的 MACELES-LES（MACE + Latent Ewald Summation）CPU 密度扫描工作树。
内容包括：MD 流水线、后处理程序、LAMMPS/Symmetrix 编译材料、**模型权重**、**引擎源码快照**、
初始构型、分片清单、运行元数据和文档。

轨迹（约 8 GB）与后处理数据不在本仓库，见文末 Zenodo 记录。

---

## 1. 目录结构

```
maceles/                  主流水线（= 原 cdft/new 仓库快照）
  assets/maceles_off/      模型权重与 JIT 产物
    maceles_off_chno.json  Symmetrix/Kokkos 引擎格式（40 MB）
    MACELES-OFF_small.model PyTorch 参考 checkpoint（11 MB）
    maceles_off.cubin      sm_80 JIT 设备产物（221 KB）
    README.md              三个文件的用途与 sha256
  assets/symmetrix-xl-6c5f73b-les-src.tar.gz   含 LES 改动的完整源码快照（18 MB）
  assets/symmetrix-xl-README.md                快照说明与编译命令
  sampling/                流水线本体（les_batch / les_c1c2 / scan_analysis / ...）
  scripts/                 编译与环境脚本（build_engine.sh / setup_env.sh / ...）
  packaging/               symmpolar CLI、CUDA engine wheel、可移植 lmp-env bundle
  docs/ validation/ results/ examples/ tests/ schemas/ skills/

scan/                     本次扫描专用
  scripts/                 build_scan / make_shards / postprocess_shard / monitor / finalize
  data/<species>/*.lammps  60 个初始构型（6 分子 × 10 密度）
  manifests/<shard>/       shard 清单与 sbatch
  meta/<sim_id>/meta.json  每个状态的模型哈希、时间、速率、状态
  bench/                   低密度性能诊断的输入与日志
  *.md, *.csv, *.xlsx      密度阶梯、参数同步、实际跑的密度点

postprocessing/           水体系后处理程序（独立仓库快照）
docs/                     特征空间/后处理接口笔记
```

## 2. 不包含什么

| 内容 | 原因 | 位置 |
|---|---|---|
| MD 轨迹（46 个状态，约 5.8 GB） | 体积，已在数据仓库 | Zenodo `24123911` |
| c1/c2 与配对关联数据 | 体积 | Zenodo `24123652` |
| 运行的日志 | 清理时删除（其中 99.9% 是 Symmetrix 图重建的诊断输出） | — |
| Python 依赖包（venv、pip 包、Kokkos 源码、CUDA） | 依赖，不是本次产物 | 见 §4 |

## 3. 大文件校验

```
b0e346bbe83306c1b1f73193a54cfaba9ff140788aff71847a10cc3539937431  maceles/assets/maceles_off/maceles_off_chno.json
343d24a5fa650ecb9cd325ba133c0f156ae877567c394927ac159e17c8176c0a  maceles/assets/maceles_off/MACELES-OFF_small.model
acb7f274d5273f2cd072533d56a39d5e9831e4355d9501d57492233ebd8b26e5  maceles/assets/maceles_off/maceles_off.cubin
7b3ee1e998a0084985d2674f95c4af76cb4e681b290b960c76bef57f5b148585  maceles/assets/symmetrix-xl-6c5f73b-les-src.tar.gz
```

## 4. 编译与安装

### 4.1 LES 引擎源码

源码快照 = upstream `6c5f73b` + LES commit（superproject `ccdde6b`），**已包含 LES 改动**，
解包即用。对应线上分支：`github.com/Yong-Q/symmetrix-xl_polar` 的 `polar-les`。

```bash
mkdir -p ~/engines && tar xzf maceles/assets/symmetrix-xl-6c5f73b-les-src.tar.gz -C ~/engines/symmetrix-xl
# 关键文件应存在：libsymmetrix/source/les_ewald.hpp
```

注意：用 `main` 分支会静默丢掉 LES，必须用 `polar-les` 或上述快照。

### 4.2 Python 侧

```bash
PY=python3 maceles/scripts/setup_env.sh          # 检查
PY=python3 maceles/scripts/setup_env.sh --install # 安装 numpy scipy ase matplotlib
```

### 4.3 LAMMPS + Symmetrix 引擎

```bash
SYMMETRIX_DIR=~/engines/symmetrix-xl \
LAMMPS_DIR=~/lammps-stable_30Sep2026 \
  maceles/scripts/build_engine.sh
```

CMake 关键项：`CMAKE_CXX_STANDARD=20`、`PKG_KOKKOS=ON`、`Kokkos_ENABLE_CUDA=ON`、`CUDA_ARCH=80`。
`build_engine.sh` 会去找 `engine/polar-les/*.patch`；本仓库不带 patch（改动已在快照里），
脚本会走“无 .git → 假定 LES 已在树中”的分支，这是预期行为。

### 4.4 CUDA engine wheel（可选）

```bash
SYMMPOLAR_CUDA_ARCH=80 maceles/packaging/symmpolar-engine/build_wheel.sh
```

## 5. 跑密度扫描

```bash
# 1) 生成初始构型（已提供，可跳过）
python -m sampling.make_density_scan --out scan/data ...

# 2) 切分
python scan/scripts/make_shards.py ...

# 3) 单个 shard
WORKDIR=scan/shard_H2O_1 MANIFEST=scan/shard_H2O_1/manifest.csv \
  LMP=$BUILD_DIR/lmp python -m sampling.les_batch --run --resume
```

每个状态：100 ps 平衡（200000 步）+ 1 ns 生产（2000000 步），步长 0.5 fs，
每 1000 步写一次 dump（2001 帧）。dump 字段：`id mol type x y z fx fy fz`。

## 6. 后处理

```bash
# a) 本仓库的主分析（g(r)/取向/q/MSD/c1c2）
python -m sampling.scan_analysis --root <states> --stage all --out <out>

# b) 水体系后处理程序（周期边界、H 归属、176 维特征、OZ c2 ...）
cd postprocessing
python -m waterpost inspect --input <traj.dump.gz>
python -m waterpost analyze --input <traj.dump.gz> --output <out>
python -m waterpost rdf-only --input <traj.dump.gz> --output <out>
```

## 7. 已知问题：高温水存在质子转移

在 T = 986.4 K 的水轨迹里，H 会在 O 之间换主（H2O + H2O ⇄ H3O+ + OH- 的接力），
因此**初始 `mol` 编号不再代表完整分子**，按 `mol` 算会看到 2–26 Å 的“超长 O–H”。

- 同一时刻只有 2–8 个 O 是缺陷（少氢/多氢），其余都是完整 H2O
- 1 ns 内换过 O 的 H 比例：ρ\*=0.67 → 75.6%，0.5978 → 88%，0.3089 → 6.4%，
  0.2367 / 0.0922 → 0%
- 其他五种分子（CO2/CH4/C2H4/C2H6/NH3）全程 0 次换主
- 检查过：盒子恒定、原子数恒定、ΣF≈0、力分布光滑、压力 ~0.6 GPa 且不随时间漂移

因此后处理必须逐帧按最近 O 重新归属 H（`--topology-mode nearest --bond-cut 1.35`），
分子内量只在“当前有效 H2O”上统计。

## 8. 相关数据记录

| 记录 | 内容 |
|---|---|
| Zenodo `24123911` | 原始轨迹（46 个状态，5.82 GB） |
| Zenodo `24123652` | c1/c2 与配对关联（41 个打包，0.94 GB） |

## 9. 版本说明

- 流水线快照：`maceles/`（对应原仓库 `cdft/new`，含 2026-11-12 新增的 `sampling/pbc_local_env.py`）
- 后处理快照：`postprocessing/`（commit `35bd835`）
- 扫描记录：46/54 状态完成；缺 NH3 0.5256、CH4 0.1644、C2H4 0.5256、C2H6 0.5256/0.3811/0.3089，ρ\*=0.02 全部未跑
