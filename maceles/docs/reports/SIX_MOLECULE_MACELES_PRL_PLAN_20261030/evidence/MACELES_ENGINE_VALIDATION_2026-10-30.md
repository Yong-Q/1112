# MACELES in the Symmetrix legacy engine — first validation (2026-10-30)

## What was added

Engine (`libsymmetrix/source`):
- `mace_kokkos.hpp`, `mace_kokkos_model.cpp`, `mace_kokkos_second_interaction.cpp`:
  LES charge readout (two layers, `MultilayerPerceptronKokkos` for the second),
  `finalize_les_charges`, LES members.
- `les_ewald.hpp`: header-only periodic triclinic GTO-Ewald, energy + analytic
  forces, verified against `les/module/ewald.py` and finite differences.
- `mace_kokkos_evaluate.cpp`: `apply_les_ewald_energy_forces()` after the
  readouts; geometry from prepared (direct) or all-interactions (generic), or
  from the LAMMPS external geometry; tiled plans rejected.
- `mace_kokkos_runtime.cpp`: LES disables tiled admission.
- `pair_symmetrix/pair_symmetrix_mace_kokkos.cpp`: passes the LAMMPS box and
  node positions, and adds `les_ewald_forces` to the LAMMPS force array after
  the edge scatter.

Converter (`symmetrix/source/symmetrix/extract_maceles.py`):
- Exports `model_type: "MACE_LES"`, flat `les_*` keys, and the structured
  `latent_ewald` block.
- **Fusion basis:** the compact engine H1 basis is
  `weights_to_fuse[0] = interactions[1].linear_up / sqrt(C)`. The linear LES
  readout must be exported as `inv(weights_to_fuse[0]) @ (w/sqrt(C))`; the
  nonlinear readout consumes H2 and stays in the raw basis. Without this the
  charges are wrong (energy off by ~10 eV, force correlation ~0).
- `les_remove_mean` is exported as **False**: `Les.remove_mean` only affects
  the BEC head; the reference Ewald does not subtract the mean charge.

## Validation (1,152-atom water box, PyTorch vs engine)

| quantity | engine | reference | delta |
|---|---:|---:|---:|
| energy | -217.8345 eV | -217.8336 eV | **-0.0008 meV/atom** |
| force MAE | — | — | **0.00082 eV/Å** |
| force max | — | — | 0.0031 eV/Å |
| force corr | — | — | **0.99995** |
| charges (sorted) | — | — | MAE 1.3e-7, corr 1.0 |

float32 engine (`symmetrix/mace/float32/kk`), float64 PyTorch.

## Direct/JIT path — fixed (2026-10-30)

First direct/JIT run produced sign-flipped LES charges (sum -61.2 instead of
-33.7, per-atom corr -0.9997, energy off by +5.6 eV) while the backbone was
identical. Root cause: in `compute_readout_2`, `use_mh0_adjoint_reuse()`
aliases `H2_adj` onto `H2`, and the energy readout's gradient writes the
adjoint in place. The LES MLP readout ran *after* that write, so it consumed
the adjoint instead of the forward H2. Fix: evaluate the LES readout before
the energy readout gradient. Direct/JIT now matches the reference:

| quantity | direct/JIT | reference |
|---|---:|---:|
| energy | -217.83451 eV | -217.83357 eV |
| force MAE | 0.000823 eV/Å | — |
| force corr | 0.99995 | — |
| LES charge sum | -33.73998 | -33.73984 |

## Speed (LAMMPS, one A100, FP32)

| path | atoms | steps | wall | µs/atom/step |
|---|---:|---:|---:|---:|
| generic | 1,152 | 200 | 52.6 s | 228 |
| generic | 5,760 | 100 | 46.8 s | 81.2 |
| **direct/JIT** | **5,760** | **100** | **23.2 s** | **40.3** |
| MACE-OFF23 medium, direct/JIT (reference) | 5,760 | 200 | 43.1 s | 37.4 |
| MACE-OFF23 medium, generic (reference) | 5,760 | 100 | 60.3 s | 104.6 |

MACELES direct is ~8 % slower than the same-class MACE-OFF23 medium direct;
the LES Ewald is a small additive Kokkos kernel.

## Speed (generic streaming path, LAMMPS, one A100)

| atoms | steps | wall | µs/atom/step |
|---:|---:|---:|---:|
| 1,152 | 200 | 52.6 s | **228** |
| 5,760 | 100 | 46.8 s | **81.2** |
| 1,152, backbone only (no LES) | 200 | 54.2 s | 235 |

The LES Ewald is **within noise of the backbone** on the generic path
(<2 %); the generic streaming path itself dominates. This is **not** the
4.47 µs `streamed_edges direct` + JIT path: direct requires a model-specific
JIT device artifact, which needs the native `symmetrix` Python backend
(not built in this environment). Generating the MACELES artifact is the next
step for a production speed number.

## Notes / remaining

- The LAMMPS route supported now is `streamed_edges generic`,
  single-rank, non-tiled. Materialized, multi-rank, and field-coupled are
  rejected with explicit errors.
- Pressure/virial does not yet include the Ewald virial.
- The charge debug dump (`SYMMETRIX_LES_DUMP_CHARGES=1`) writes
  `les_charges.txt`; the node order equals `d_ilist`, so it is a permutation
  of the atom order (the physical pairing with positions is correct, as the
  energy/forces show).
