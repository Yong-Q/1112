# MACELES 密度扫描：换机器快速上手

一条链路：**建盒 → 分片 → MD（逐条压缩轨迹）→ c1/c2 + 结构后处理 → 汇总图表**。

```
sampling/make_density_scan.py   6 物种 × 10 密度，刚性分子缩放 + 分级升温
        │
        ▼  density_scan.csv
sampling/split_scan.py          切成 N 个 shard 的 manifest
        │
        ▼  shard_i/attempt-01/manifest.csv
sampling/les_batch.py           LAMMPS(Kokkos/MACELES-LES) NVT → traj.dump → gzip
        │
        ▼  samples/<sim_id>/{run/traj.dump.gz, meta.json}
sampling/scan_analysis.py       c1/c2(f1,f2) + g(r)/取向/氢键/四面体序 → 汇总
        │
        ▼  analysis_scan/{scan_summary.csv, c2_<species>.npz, figures/*.png}
```

## 0.0 用打包好的引擎（不编译，最快）

`dist/maceles-lmp-env-*.tar.gz` 是 **解包即用** 的引擎包：LAMMPS + Kokkos/CUDA +
`pair_style symmetrix/mace/.../kk`、运行库（cudart/openblas/gfortran/gomp/stdc++）、
`assets/maceles_off/` 的 json+cubin、`sampling/` 全套（建盒/分片/MD/c1c2/后处理）、
示例输入和 `jobs/` 模板（veMLP 的 `q4_veMLP.yaml`、`entrypoint_scan.sh`、Slurm）。

```bash
tar xzf maceles-lmp-env-*.tar.gz && cd maceles-lmp-env-*
. ./env.sh
examples/smoke/run.sh                 # 64 水分子 500 步，A100 上约 30 s
examples/pipeline/run.sh              # 建盒→MD→c1/c2 后处理，全链路
```

节点上只需要：x86-64 + glibc ≥ 2.34、NVIDIA 驱动 ≥ 525、CUDA 12 的 `libcublas.so.12`
（`env.sh` 会自动找 `/opt/matos-tools/cuda/*/lib64`、`/usr/local/cuda*/lib64`，
也可用 `CUDA_HOME` / `MACELES_CUDA_LIB` 指定）。cublas 有 0.85 GB，所以没有打进包里。

交任务：`jobs/q4_veMLP.yaml`（把 `UserCodePath` 指向解包目录），入口是
`jobs/entrypoint_scan.sh`，它读 `MANIFEST`，用 `les_batch.py --run --resume` 跑一个 shard。

重新打包（换了 lmp 二进制之后）：

```bash
LMP=bin/lmp scripts/make_lmp_bundle.sh          # -> dist/maceles-lmp-env-*.tar.gz
scripts/make_lmp_bundle.sh --check dist/*.tar.gz
```

## 0. 依赖

- Linux + NVIDIA GPU（A100 级；Mac/CPU 只能跑后处理）
- CUDA 12.x、cmake ≥ 3.27、gcc ≥ 11
- Python 3.11：`numpy scipy ase matplotlib`（`requirements.txt`）
- LAMMPS `stable_30Sep2026` 源码 + `pair_symmetrix` 包（脚本会自动安装）
  - 需要 `git@github.com:bonan-group/symmetrix-xl.git` 的读权限


## 0.5 模型资产（两个文件，不在 git 里）

`pair_style symmetrix/mace/float32/kk ... streamed_edges direct` 必须给一个
`jit_device_artifact`，所以每次运行都要这两个文件：

| 文件 | 大小 | 是什么 |
|---|---|---|
| `maceles_off_chno.json` | 41 MB | 官方 **MACELES-OFF** 权重导出成 Symmetrix/Kokkos 引擎格式（权重 + fusion basis + `les_norm_factor`）。**不是** PyTorch 的 `.pth` |
| `maceles_off.cubin` | 226 KB | 该模型 R1 直执行算子的 CUDA 预编译产物；`streamed_edges direct` 没有它就拒绝启动 |

仓库 `.gitignore` 排除了 `models/` 和 `bench/`，所以它们属于**工作区资产**，不在 git 里。

**A. 从现有工作区拷（最快，推荐）**

```bash
rsync -av <现有工作区>/bza/symmetrix-polar/models/maceles_off_chno.json ./assets/
rsync -av <现有工作区>/bza/symmetrix-polar/bench/q4_maceles_off/attempt-03/maceles_off.cubin ./assets/
sha256sum assets/maceles_off_chno.json assets/maceles_off.cubin
# b0e346bbe83306c1b1f73193a54cfaba9ff140788aff71847a10cc3539937431
# acb7f274d5273f2cd072533d56a39d5e9831e4355d9501d57492233ebd8b26e5
export MODEL=$PWD/assets/maceles_off_chno.json CUBIN=$PWD/assets/maceles_off.cubin
```

**B. 从官方 checkpoint 重新导出 json**（需要 MACELES 的 `.pth` + 装好 symmetrix Python 包）

```bash
python symmetrix/source/symmetrix/extract_maceles.py \
    --model maceles_off.pth --output maceles_off_chno.json
```

**C. 只有 json、需要重新生成 cubin**（换了 GPU 架构，如 A100→H100 时必须重编）

```bash
python -m symmetrix.cli.prepare_jit_device_artifact \
    --model maceles_off_chno.json --precision float32 --lammps-arguments
# 输出直接可粘到 pair_style 后面；成功后会打印 artifact 路径
```

注意：cubin 是针对具体 GPU 架构编译的，跨架构（sm_80 → sm_90）会加载失败，用 C 重新生成。

## 1. 环境（1 分钟）

```bash
git clone git@github.com:Yong-Q/test.git && cd test
export PY=$HOME/miniconda3/envs/matos/bin/python      # 或任意 3.11
scripts/setup_env.sh --install                        # 检查/安装 numpy scipy ase matplotlib
```

没有 conda 时：`python3 -m venv .venv && . .venv/bin/activate && scripts/setup_env.sh --install`



### 引擎源码（fork）

MACELES/LES 引擎在 `Yong-Q/symmetrix-xl_polar` 的 `polar-les` 分支
（上游 `6c5f73b` + 一个 LES commit，公开仓库）：

```bash
git clone --recurse-submodules -b polar-les \
    git@github.com:Yong-Q/symmetrix-xl_polar.git ~/engines/symmetrix-xl
SYMMETRIX_DIR=~/engines/symmetrix-xl SYMMPOLAR_CUDA_ARCH="80;90" \
  packaging/symmpolar-engine/build_wheel.sh
```

`build_wheel.sh` 默认就从这里拉（`SYMMETRIX_REPO` / `SYMMETRIX_REF` 可覆盖）；
离线场景用 `assets/symmetrix-xl-6c5f73b-les-src.tar.gz` 快照。

## 1.5 pip 安装（可选，替代源码编译引擎）

```bash
pip install packaging/symmpolar                     # CLI: symmpolar check/smoke/scan
pip install symmpolar_engine-0.1.1+cu12.9.sm80-*.whl   # 引擎（GitHub Release / 自建）
symmpolar build-lammps --lammps-dir ~/lammps        # LAMMPS 仍需本机编一次
```

引擎 wheel 由 `packaging/symmpolar-engine/build_wheel.sh` 或
`.github/workflows/wheels.yml`（CUDA 12.9 × sm80/sm90 矩阵）产出；LAMMPS 二进制
不进 PyPI（GPL + 体积），用编译脚本或 Release 里的 `lmp`。

## 2. 编译引擎（10–20 分钟）

```bash
export SYMMETRIX_DIR=$PWD/../symmetrix-xl          # 已 clone 就指向它
export LAMMPS_DIR=$PWD/../lammps-stable_30Sep2026  # LAMMPS 源码
export CUDA_HOME=/usr/local/cuda                   # 或 /opt/matos-tools/cuda/12.9
export CUDA_ARCH=80                                # A100=80, H100=90
scripts/build_engine.sh
export LMP=$SYMMETRIX_DIR/symmetrix/build-lammps-les/lmp
```

脚本做三件事：`pair_symmetrix/install.sh` 把 pair style 装进 LAMMPS 源码、
应用 `engine/polar-les/0001-*.patch`（已应用会自动跳过）、cmake+make
（Kokkos CUDA、无 MPI）。

## 3. 冒烟测试（2 分钟，需要 GPU）

```bash
export MODEL=/path/to/maceles_off_chno.json
export CUBIN=/path/to/maceles_off.cubin
scripts/smoke_test.sh
```

64 个水分子、500 步、后处理全走一遍；最后打印 `scan_summary.csv` 并生成
`smoke/analysis_scan/figures/`。这一步过了，说明引擎、模型、后处理都通。

## 4. 正式扫描

```bash
export MODEL=... CUBIN=... LMP=...
scripts/run_density_scan.sh
```

默认：6 物种 × 10 密度（0.2–2.2 ρc，T = 1.55 Tc）、每盒 3000 原子、
100 ps 平衡 + 1 ns 生产、0.5 ps 一帧、逐条 gzip。

分阶段跑（推荐集群上分开）：

```bash
STAGE=data  scripts/run_density_scan.sh    # 只建 60 个输入盒
STAGE=split scripts/run_density_scan.sh    # 切成 4 个 shard
STAGE=run   scripts/run_density_scan.sh    # MD（可重复执行，--resume 跳过已完成）
STAGE=post  scripts/run_density_scan.sh    # 后处理 + 图表
```

可调环境变量：`OUT`（扫描根目录）、`SHARDS`、`DUMP_EVERY`（步）、
`EQUIL_STEPS`、`STEPS`、`PY`。

## 5. 输出

```
$OUT/
  data/<species>/rNN_<rho>.lammps      60 个输入盒
  density_scan.csv                     全量清单（含 T、rho、密度）
  shard_i/attempt-01/
    manifest.csv
    samples/<sim_id>/run/traj.dump.gz  轨迹（gzip）
    samples/<sim_id>/meta.json         模型/cubin/数据 sha256、计时
  analysis_scan/
    <sim_id>/c1c2/f1,f2                PRL 格式 c1/c2（rho/Vext/mu；r,c2,c2k）
    <sim_id>/analysis/profiles.npz     g(r)、键长、P2、q_tet
    scan_summary.csv                   每条一行：峰位/峰高/配位数/角度/氢键/P2/q
    c2_<species>.npz                   r 网格 + c2 矩阵 + rho/T（跨密度）
    figures/c2_<species>.png           c2(r) 随密度变化
    figures/gr_<species>.png           g(r) 随密度变化
    figures/trends.png                 峰高/配位/四面体序/氢键 vs rho/rho_c
```

## 6. 在 Volc veMLP 上跑

1. `STAGE=data split` 本地生成数据与 manifest；
2. 用 `--remote-prefix /vepfs-.../workspace-state/shared/bza/symmetrix-polar/bench/maceles_density_scan`
   生成远端路径的 manifest；
3. 任务 entrypoint（与现有 attempt-04 YAML 相同）：

```bash
BZA="$WORKDIR/../../../../.."
sed "s#<remote-prefix>#$BZA#g" manifest.csv > manifest.remote.csv
python3 "$BZA/test/sampling/les_batch.py" --manifest manifest.remote.csv \
  --outroot "$WORKDIR/samples" --run --resume --no-analyze \
  --equil-steps 200000 --steps 2000000 \
  --lmp "$BZA/external-repos/symmetrix-xl/symmetrix/build-lammps-99ca96e1225f5a30/lmp" \
  --model "$BZA/symmetrix-polar/models/maceles_off_chno.json" \
  --cubin "$BZA/symmetrix-polar/bench/q4_maceles_off/attempt-03/maceles_off.cubin" \
  --ld-library-path "$LD_LIBRARY_PATH"
```

单卡单任务；完事后本地跑 `STAGE=post`（轨迹已 gzip，`scan_analysis` 直接读 `.gz`）。

## 7. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `Error writing dump 1: Disk quota exceeded` | 工作区/身份配额满（worker 以 `nobody` 写，额度小） | 逐条 gzip（默认开）、调大 `DUMP_EVERY`，或把 `OUT` 指到大配额盘 |
| `PermissionError` on `samples/` | 任务切到了别的 uid | 不要加 `--drop-priv`；目录属主需与运行用户一致 |
| `Lost atoms` | 初始结构有近接触/键被拉伸 | 用 `make_density_scan.py` 的刚性缩放；保持 `neigh_modify every 10 ... check yes` |
| `CUDA error` / 无 GPU | 未在 GPU 节点或 `-k on g 1` 未生效 | `nvidia-smi` 自查；`les_c1c2.run_engine` 默认 `-k on g 1` |
| `ModuleNotFoundError: ase` | 环境没装 | `scripts/setup_env.sh --install` |
| `cannot find -lblas` 链接失败 | 缺 OpenBLAS | `export LIBRARY_PATH=<conda>/lib:$LIBRARY_PATH`（build_engine.sh 已设） |

## 8. 文件地图

| 文件 | 作用 |
|---|---|
| `sampling/make_density_scan.py` | 6 物种 × N 密度的输入盒（刚性分子缩放） |
| `sampling/split_scan.py` | 清单切 shard、远端路径重写 |
| `sampling/les_c1c2.py` | LAMMPS 输入、运行、c1/c2 提取 |
| `sampling/les_batch.py` | 批量驱动：`--run/--analyze-only/--no-analyze`、gzip、meta.json |
| `sampling/scan_analysis.py` | 统一后处理 + 汇总 + 图表（本文件上游） |
| `sampling/make_smoke_box.py` | 冒烟测试用 64 水分子盒 |
| `sampling/cdft_dataset.py` | PRL/PNAS 格式的 c1/c2 计算（被上层调用） |
| `scripts/setup_env.sh` | Python 依赖检查/安装 |
| `scripts/build_engine.sh` | LAMMPS + Symmetrix/LES 编译 |
| `scripts/smoke_test.sh` | 端到端安装验证 |
| `scripts/run_density_scan.sh` | 一条命令跑完整流程 |
