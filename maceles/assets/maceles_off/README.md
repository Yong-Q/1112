# MACELES-OFF engine assets

`pair_style symmetrix/mace/float32/kk ... streamed_edges direct` needs both
files; without a `jit_device_artifact` the pair style refuses to start.

| file | size | sha256 |
|---|---:|---|
| `MACELES-OFF_small.model` | 11 MB | `343d24a5fa650ecb9cd325ba133c0f156ae877567c394927ac159e17c8176c0a` |
| `maceles_off_chno.json` | 40 MB | `b0e346bbe83306c1b1f73193a54cfaba9ff140788aff71847a10cc3539937431` |
| `maceles_off.cubin` | 221 KB | `acb7f274d5273f2cd072533d56a39d5e9831e4355d9501d57492233ebd8b26e5` |

- **model** — the official **PyTorch** MACELES-OFF checkpoint (TorchScript),
  loadable by `mace-torch`; this is the reference implementation used for the
  engine-vs-PyTorch force/energy comparison.  The export to the engine format
  starts from this file.
- **json** — the same checkpoint exported to the Symmetrix / Kokkos engine
  format (weights, fusion basis, `les_norm_factor=90.0474`).
- **cubin** — the precompiled CUDA R1 direct-execution kernels for that model.
  Compiled for **sm_80 (A100)**; on another architecture it must be rebuilt
  (see below), otherwise the load fails.

## Use

```bash
export MODEL=$PWD/assets/maceles_off/maceles_off_chno.json
export CUBIN=$PWD/assets/maceles_off/maceles_off.cubin
scripts/smoke_test.sh
```

`pair_style` expects `jit_device_blocks_per_compute_unit 8` with this cubin.

## Rebuild

From the official checkpoint:

```bash
python symmetrix/source/symmetrix/extract_maceles.py \
    --model maceles_off.pth --output maceles_off_chno.json
```

From the json, on the target GPU (rebuilds the cubin for the local arch):

```bash
python -m symmetrix.cli.prepare_jit_device_artifact \
    --model maceles_off_chno.json --precision float32 --lammps-arguments
```

Verify with `sha256sum` against the table above before/after copying.
