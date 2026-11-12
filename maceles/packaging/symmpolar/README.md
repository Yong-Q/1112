# symmpolar

CLI wrapper around the MACELES/LES toolchain:

| command | does |
|---|---|
| `symmpolar check` | verify python deps and, if given, the model/cubin/lmp paths |
| `symmpolar build-lammps --lammps-dir DIR` | install `pair_symmetrix` into a LAMMPS tree and build `lmp` (Kokkos CUDA) |
| `symmpolar extract-model --checkpoint X.pth -o X.json` | official MACELES checkpoint → engine JSON |
| `symmpolar prepare-cubin --model X.json` | build the `jit_device_artifact` .cubin on this GPU |
| `symmpolar smoke --model ... --cubin ... --lmp ...` | 64-water end-to-end test |
| `symmpolar scan --stage data\|split\|run\|post` | the 6-species density scan |

It is a thin front end: the real code stays in the repository
(`scripts/`, `sampling/`), which is why `symmpolar` only depends on numpy /
scipy / ase / matplotlib.

## What can live on PyPI

- this package (pure python) — yes
- the symmetrix python extension (CUDA) — yes, as a per-CUDA/arch wheel
  (large, `linux_x86_64`, built with `scikit-build-core` + Kokkos CUDA)
- a full LAMMPS binary — **no**: GPL-2.0, ~200 MB, MPI-dependent. Ship the
  pair style as a LAMMPS plugin or build locally with `symmpolar build-lammps`.
- model weights (40 MB) — keep on a GitHub Release / HF, not PyPI.

## Two-package install

```
pip install symmpolar                       # CLI (pure python)
pip install symmpolar-engine-0.1.1+cu12.9.sm80-*.whl   # CUDA engine
symmpolar build-lammps --lammps-dir ~/lammps           # or use a release lmp
```

`symmpolar-engine` is built by `packaging/symmpolar-engine/build_wheel.sh`
(local, ~20 min) or by `.github/workflows/wheels.yml` (CUDA 12.9 x sm80/sm90
matrix, publishes on `v*` tags).  It reuses the upstream scikit-build backend
and only changes `SYMMETRIX_DEVICE_BACKEND=CUDA` and the distribution name.
