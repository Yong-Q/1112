# MACELES engine — repeat verification and real 1,500-atom water speed (2026-10-30)

Two dedicated q4 A100 tasks (queue004, ml.pni2.3xlarge, 1 GPU each):

- `t-20261030163239-2fnb7` — repeatability + numerical validation at 1,002 and
  5,001 atoms, interleaved LES / no-LES arms in one task.
- `t-20261030164741-p898x` — real 1,500-atom water box at 1.0 g/cm³: 10 ps NVT
  equilibration, then interleaved NVT timing and a step-0 force dump.

Both use `streamed_edges direct`, the same model-specific JIT cubin
(`jit-r1-gen11-f32-44c7e72f932e893e`), FP32, and the same `maceles.json`
SHA-256 `29e0b469…dc6ee1` (`model_type=MACE_LES`, 15 `les_*` keys,
sigma=1.0 Å, dl=2.0 Å, remove_self=True, remove_mean=False). The no-LES arm
is the same JSON with the LES module removed (`2035ee52…16fcbd`).

## Repeatability (15 × 20-step NVT samples per arm per size)

| atoms | density (molecules/Å³) | MACELES (min / median) | backbone (min / median) | paired ratio |
|---:|---:|---:|---:|---:|
| 1,002 | 0.020 | 5.09 / 5.42 µs/atom | 3.66 / 4.01 | 1.390 / 1.392 / 1.397 |
| 5,001 | 0.020 | **4.218 / 4.226** | **2.733 / 2.794** | 1.542 / 1.544 / 1.536 |
| 1,500 | 0.0334 (1.0 g/cm³) | **5.372 / 5.710** | **4.462 / 5.395** | 1.204 / 1.212 / 1.196 |

All speeds are µs/atom/step. The 5,001 values are stable to <1 % across all 15
samples. At 1,002/1,500 the loop time is small enough that GPU clock state
(210 MHz idle → 1410 MHz boost) dominates; the minimum is the clean statistic.
GPU at the end of both runs: 1410/1410 MHz, 32–38 °C.

The LES fraction is size- and density-dependent: +54 % at 5,001 low-density
atoms (Ewald reciprocal-space work grows with box volume), +39 % at 1,002,
and +20 % at 1,500 ambient-density atoms where the denser neighbour list
(143 neighbours/atom) dilutes the Ewald term.

## Real 1,500-atom water (500 H₂O, 1.0 g/cm³)

Structure: 8×8×8 molecular grid, 500 molecules, random non-clashing
orientations, L = 24.638 Å, closest initial contact 1.818 Å. Equilibration:
2,000 steps at 0.25 fs ramping 100→300 K (18.98 s) + 18,000 steps at 0.5 fs at
300 K (171.18 s); final T = 299.05 K, PE = −414.48406 eV.

- Continuous-run throughput: **6.34 µs/atom/step** (9 ps in 171.2 s).
- Clean 20-step segments (NVT, `fix nvt 300 300 0.1`): **5.37 µs/atom/step**
  MACELES, 4.46 same backbone.
- 1 ns of 1,500-atom water MD at 0.5 fs: **≈4.5 h** clean, **≈5.3 h** at the
  sustained long-run rate, on one A100. At 1 fs the wall time halves.

These timing runs use the NVT ensemble (`fix nvt`), not `run 0`; the only
`run 0` in the task is the force dump used for the numerical comparison.

## Numerical validation vs PyTorch MACELES (FP64, identical geometries)

| atoms | E engine (eV) | E PyTorch (eV) | ΔE (meV/atom) | F MAE (eV/Å) | F max (eV/Å) | corr |
|---:|---:|---:|---:|---:|---:|---:|
| 1,002 | −174.697540 | −174.696702 | −0.000838 | 0.00333 | 0.0272 | 0.99887 |
| 1,500 | −414.484060 | −414.482860 | −0.000800 | 0.00861 | 0.1352 | 0.99709 |
| 5,001 | −956.652110 | −956.647792 | −0.000864 | 0.00153 | 0.0157 | 0.99986 |

The 1,500-atom reference forces reach 1.12 eV/Å (equilibrated ambient water);
the 1,002/5,001 boxes are low-density and less forced. Relative force error is
~1 % in all three cases.

## Artefacts

- `bench/q4_maceles_verify/` — logs, force dumps, `ref5000_pt.npz`, `RESULTS.md`
- `bench/q4_maceles_water1500/` — builder, inputs, logs, equilibrated data,
  `ref1500_pt.npz`
- Artifacts are workspace-local; `bench/` is git-ignored by design.
