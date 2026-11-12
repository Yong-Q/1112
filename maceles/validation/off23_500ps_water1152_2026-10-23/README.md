# OFF → POLAR reweighting audit: 1,152 atoms, 500 ps, 501 frames

This directory contains the full reproducibility package for the 500 ps
MACE-OFF23 → MACE-POLAR importance-sampling test at 1,152 atoms.

Everything is on the **current server** under
`/workspace/group/shared/bza/symmetrix-polar/`; nothing is on another machine.

## 1. Requested energy table

`frames_501_total_energies.csv`:

```
frame,step,time_ps,n_atoms,E_OFF_total_eV,E_POLAR_total_eV
```

- 501 data rows, one per dumped frame (0…500).
- `step` is the LAMMPS timestep (0, 1000, …, 500000); `time_ps = step × 0.001`
  because the MD timestep is 1 fs.
- `E_OFF_total_eV` is the **whole-box total** MACE-OFF23 energy (FP32
  calculator, as stored in `reweight_frames.json`).
- `E_POLAR_total_eV` is the **whole-box total** MACE-POLAR-1-M energy (FP64).
- Values are written with `%.17g`; no per-atom conversion and no additional
  rounding. The source values themselves are FP32-precision numbers from the
  reweight run (about 1e-3 eV on a 799 keV total, i.e. ~1e-6 eV/atom); the
  target values are FP64.

## 2. Protocol

### 2.1 MD (source trajectory)

From `in.production_1fs`:

| item | value |
|---|---|
| engine | Symmetrix LAMMPS build, KOKKOS, LAMMPS 30 Sep 2026 |
| pair style | `symmetrix/mace/float32/kk streamed_edges direct profile speed` + JIT cubin |
| model | MACE-OFF23-medium (Symmetrix JSON, FP32) |
| system | 1,152 atoms = 384 H₂O, 23.392142 × 23.392142 × 35.088213 Å |
| seed | last frame of `seed_5ps.lammpstrj` (OFF 5 ps dump, step 10000) |
| timestep | 0.001 ps = 1 fs |
| ensemble | NVT, Nosé–Hoover, 300 K, `Tdamp = 0.1` ps |
| velocities | fresh Maxwell–Boltzmann at 300 K, seed 98765 |
| equilibration | 20,000 steps = 20 ps |
| production | 500,000 steps = 500 ps |
| dump | every 1,000 steps = 1 ps → 501 frames |
| PBC | periodic x/y/z |
| restart | every 25,000 steps |

Wall time: 19:54:04 (from the run log). The trajectory itself was generated
with `pair_style .../float32/kk`, so the **source sampling is FP32**.

### 2.2 Reweighting

`reweight_500ps_water1152.py` re-evaluates every dumped frame in PyTorch:

- source: MACE-OFF23-medium, `default_dtype="float32"` (FP32, matching the MD);
- target: MACE-POLAR-1-M, `default_dtype="float64"` (FP64);
- no frames discarded; stride 1; all 501 frames used;
- temperature 300 K;
- device CUDA;
- PBC taken from the dumped cell (all three directions periodic);
- total charge neutral (384 H₂O); no spin polarization;
- no external electric field (`external_field` is zero; the model's
  `DisplacedGTOExternalFieldBlock` is present but not driven);
- POLAR long-range settings embedded in the model:
  `density_max_l=1` (charge + l=1 dipole multipoles),
  `density_smearing_width=1.5 Å`,
  `kspace_cutoff=1.8467`,
  `include_self_interaction=True`,
  `include_pbc_corrections=True`,
  2 charge-density recursion/SCF steps.
- The MACE-POLAR calculator applies its internal GTO/Ewald long-range kernel;
  no separate Ewald flag was changed.

Reproduction command (paths relative to the symmetrix-polar repository):

```bash
PYTHONPATH=/workspace/group/shared/bza/external-repos/graph_electrostatics-0.4.0:$PWD \
python scripts/reweight_500ps_water1152.py \
  --traj bench/off23_500ps_water1152_2026-10-23/traj_500ps_1fs.lammpstrj \
  --source-model /workspace/group/shared/bza/task3_sampling_database/mlip/models/MACE-OFF23_medium.model \
  --target-model /workspace/group/shared/bza/task3_sampling_database/mlip/models/mace-polar/MACE-POLAR-1-M.model \
  --temperature 300 --source-dtype float32 --target-dtype float64 \
  --outdir bench/off23_500ps_water1152_2026-10-23/reweight
```

MD reproduction command:

```bash
cd bench/off23_500ps_water1152_2026-10-23
<symmetrix-lmp> -k on g 1 -in in.production_1fs
```

The input references the JIT cubin artifact and the Symmetrix JSON
`mace-off23-medium.json`; both hashes are below.

## 3. Model versions and hashes

| artifact | path | SHA-256 |
|---|---|---|
| MACE-OFF23-medium (PyTorch) | `/workspace/group/shared/bza/task3_sampling_database/mlip/models/MACE-OFF23_medium.model` | `4842c52ad210d6e1f84d6cf1ffa70fae25a7e0d755ed55cf223f43913f587db7` |
| MACE-POLAR-1-M (PyTorch) | `/workspace/group/shared/bza/task3_sampling_database/mlip/models/mace-polar/MACE-POLAR-1-M.model` | `fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a` |
| MACE-OFF23-medium (Symmetrix JSON, MD) | `/workspace/group/shared/bza/task3_sampling_database/mlip/models/symmetrix/mace-off23-medium.json` | `ea268ca3830a2f591b60730b65d8804673f805bc66e677540f5ac29c8e1f8d69` |
| trajectory | `traj_500ps_1fs.lammpstrj` | `5ccaa551419deccd4e7b28eb38714fc9a4d402bab52ffa34756ac568b03595c8` |
| data file | `data.lammps` | `cda0683605302b4f0316d978f13fe9984814705b227a252acaa4f25cd51c9d76` |
| MD input | `in.production_1fs` | `8ddc5458990b0c34e291445c54fac0577cdd686176c0acf7af42aadce34695d0` |
| seed dump | `seed_5ps.lammpstrj` | `1e4c254827aea6c961f355fcff5c303aa436edadb731875a5c22353476148bee` |
| energy CSV | `frames_501_total_energies.csv` | `8368a09c7d0e60e590536acd22f3d9742499761a63b7101746d283fb443501e3` |
| ESS JSON | `reweight_summary.json` | `f3b813a865264c62747f6ac992154fc4f71937df96d073e803d23f07acc95a2e` |
| reweight script | `reweight_500ps_water1152.py` | `48a5a60e7c34680b420488ee77702ef10095cd92a994f663c01d961409373fd9` |

The PyTorch model weights themselves are 18 MB (OFF) and 65 MB (POLAR); they
are not duplicated in this repository. They live at the paths above. The
Symmetrix JSON used for the MD is 36 MB and is also at the path above.

## 4. Results (from `reweight_summary.json`)

| quantity | value |
|---|---:|
| frames | 501 |
| atoms | 1,152 |
| T | 300 K |
| ΔE/atom mean (constant offset) | 417.4165 meV/atom |
| ΔE/atom std | **0.308824 meV/atom** |
| residual std after linear fit | 0.285592 meV/atom |
| σ_β (full) | **13.7616** |
| ESS | **2.34867 / 501 = 0.4688 %** |
| linear slope | 0.924770 |
| linear intercept | −59650.777 eV |
| linear R² | 0.962383 |
| verdict | **collapse** |

The per-frame weights and total energies are in `reweight_frames.json` and
`frames_501_total_energies.csv`.

## 5. Potential-energy fluctuations

`potential_fluctuations_501.png` / `.pdf` (two panels),
`_single.png` (all three curves on one axis), `_combined.png` (curves +
fluctuation-magnitude bars) and `_bath.png` (curves with the ±1σ heat-bath
envelope plus the thermostat temperature); script
`validation/plot_501_potential_fluctuations.py`.

![501-frame potential-energy fluctuations](potential_fluctuations_501.png)

![All three curves on one axis](potential_fluctuations_501_single.png)

![Heat-bath envelope and thermostat temperature](potential_fluctuations_501_bath.png)

![Curves and fluctuation magnitudes](potential_fluctuations_501_combined.png)


- left: (E - <E>)/N for MACE-OFF23 and MACE-POLAR over the 501 frames; the
  two curves nearly overlap because the common thermal fluctuation dominates;
- right: ΔU = E_POLAR - E_OFF, mean **480.864 eV**, std **0.356 eV**
  (0.309 meV/atom).  The first ~10 ps show the initial equilibration
  transient; production statistics use all 501 frames unless stated.
- comparison: `std(E_OFF)/N = 1.562`, `std(E_POLAR)/N = 1.473`,
  `std(dU)/N = 0.309 meV/atom`; the thermal fluctuation is ~5x the
  energy-difference fluctuation (the two potentials correlate at 0.981).
  The whole-box `beta*dU` still has sigma = 13.76.

## 5. Logs and other files

- `log.production_1fs` — MD thermo (step, temp, pe) every 1,000 steps.
- `stdout.production_1fs` — MD stdout with the KOKKOS/Symmetrix banner.
- `reweight_500ps.log` — reweighting progress and final JSON.
- `reweight_frames.json` — per-frame `timestep`, `n_atoms`, `e_source_eV`,
  `e_target_eV`, `dE_total_eV` (same numbers as the CSV, plus ΔE).
- `reweight_rdf.json` — O–O RDF for OFF, reweighted, and direct POLAR.
- `traj_500ps_1fs.lammpstrj` — 501-frame positions-only trajectory (17 MB).
- `data.lammps`, `seed_5ps.lammpstrj`, `in.production_1fs` — exact MD inputs.
- `reweight_500ps_water1152.py`, `make_reweight_500ps_figure.py` — analysis
  scripts.

## 6. Caveats

1. **Source trajectory is FP32.** The MD used the FP32 Symmetrix linear path,
   so the source ensemble has FP32-level noise. The reweighting verdict is
   dominated by the POLAR/OFF energy gap (measured with target FP64), but this
   trajectory should not be used as an FP64-accurate reference.
2. **ESS collapse is not a frame-count problem.** With a fixed weight
   distribution, adding frames increases ESS but not the ESS fraction;
   σ_β = 13.76 is extensive and the ESS fraction stays at 0.47 %.
3. The linear fit (R² = 0.962) does **not** rescue the reweighting: the slope
   0.925 mainly absorbs a systematic per-atom reference-energy difference and
   is equivalent to sampling at ~324 K.
4. The RDF from 2.35 effective frames is noise, not a structural shift.

## 7. Source locations on this server

- Original run directory:
  `/workspace/group/shared/bza/symmetrix-polar/bench/off23_500ps_water1152_2026-10-23/`
- Report:
  `/workspace/group/shared/bza/symmetrix-polar/docs/results/REWEIGHT_500PS_1152_2026-10-25.md`
- Reweight log:
  `/workspace/group/shared/bza/symmetrix-polar/logs/reweight_500ps.log`
- This package:
  `/workspace/group/shared/bza/test/validation/off23_500ps_water1152_2026-10-23/`
