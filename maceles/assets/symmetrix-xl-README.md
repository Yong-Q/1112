# symmetrix-xl source snapshot (LES, pinned)

> 在线版本：`git clone --recurse-submodules -b polar-les \
> github.com/Yong-Q/symmetrix-xl_polar`（公开）. 这个 tar 是同一棵树的离线快照。

`symmetrix-xl-6c5f73b-les-src.tar.gz` — complete buildable source tree of the
MACELES/LES engine, so the wheel can be built without network access and
without patching upstream:

- superproject at `ccdde6b` (= upstream base `6c5f73b` + the LES commit),
  i.e. exactly the tree the speed/force validation was run on;
- vendored submodules at their pinned commits: Kokkos, Kokkos-Kernels,
  nlohmann/json, sphericart, cblas-prototypes;
- no `.git`, no `build*/` directories, no `.bak` files.

sha256: `7b3ee1e998a0084985d2674f95c4af76cb4e681b290b960c76bef57f5b148585`

## Build from it

```bash
mkdir -p ~/engines && tar xzf assets/symmetrix-xl-6c5f73b-les-src.tar.gz -C ~/engines/symmetrix-xl --strip-components=0
SYMMETRIX_DIR=~/engines/symmetrix-xl \
SYMMPOLAR_CUDA_ARCH="80;90" \
  packaging/symmpolar-engine/build_wheel.sh
```

or point LAMMPS at the same tree:

```bash
SYMMETRIX_DIR=~/engines/symmetrix-xl LAMMPS_DIR=~/lammps-stable_30Sep2026 \
  scripts/build_engine.sh
```

The tarball already contains the LES patch, so `build_wheel.sh` skips patching
(it only patches a git checkout).
