# MACELES engine — missing charge-response force term fixed (2026-10-30)

## Symptom

With the official checkpoint (`MACELES-OFF_small_converted.model`, 192
channels, 2 interactions, r_cut 4.5 Å, L_max=1) the engine reproduced
PyTorch energies but not forces:

- backbone (LES off): engine vs PyTorch corr **1.0000**, MAE 2×10⁻⁵ eV/Å
- LES on: corr 0.42 (1,002 atoms), 0.97 (1,500-atom lattice); the engine LES
  force increment was 19–54 % of the PyTorch increment.

The toy 128-channel checkpoint had passed because its charge readout barely
responds to geometry; the official SPICE model's charge response is large.

## Root cause

The engine's LES force was only the fixed-charge Ewald derivative

    F_engine = -∂E_LR/∂r |_q

and never backpropagated ∂E_LR/∂q through the two LES charge readouts, so

    -Σ_i (∂E_LR/∂q_i)(∂q_i/∂r)

was missing. `les_charges` was used forward-only; there was no charge adjoint.
A second, independent bug: the engine hard-coded `kNormFactor = 90.4756`
while the official checkpoint stores `les_norm_factor = 90.0474` (0.48 %
systematic error in LES energies).

## Fix

1. `les_ewald.hpp`: new `energy_forces_with_charge_gradient()` fills

       dE/dq_i = (2·norm_factor/V) Σ_k w_k (S_re cos(k·r_i) + S_im sin(k·r_i))
                 − 2·norm_factor·self_coeff·q_i  [remove_self]

   with a host reference and a charge finite-difference self-test
   (double rel. error 5×10⁻¹², float 1.9×10⁻⁷).
2. `mace_kokkos_second_interaction.cpp`: the LES readout-2 feature Jacobian is
   cached while H2 still holds forward features (H2_adj may alias H2);
   `apply_les_charge_adjoint()` chains the finalize scaling
   (`q_final = s·(q_raw − m·mean(q_raw))`) and seeds
   `H1_adj(i,0,k) += dq_raw_i·w1_k` and
   `H2_adj(i,k) += dq_raw_i·g2(i,k)` before the reverse pass.
3. `mace_kokkos_evaluate.cpp`: calls the new Ewald API and
   `apply_les_charge_adjoint()` immediately after every Ewald call.
4. `les_norm_factor` is parsed from the model JSON and pushed into the Ewald
   solver (default 90.4756 for older exports).

## Validation (FP32 engine vs FP32/FP64 PyTorch, same official checkpoint)

Forces:

| system | n | MAE (eV/Å) | max | corr |
|---|---:|---:|---:|---:|
| 1,002-atom box, LES | 1,002 | 3.0×10⁻⁵ | 1.6×10⁻⁴ | 0.999999 |
| 1,500-atom lattice, LES | 1,500 | 2.4×10⁻⁵ | 1.3×10⁻⁴ | 1.000000 |
| H₂O gas | 96 | 4.4×10⁻⁵ | 2.1×10⁻⁴ | 1.000000 |
| NH₃ gas | 128 | 3.7×10⁻⁵ | 2.5×10⁻⁴ | 1.000000 |
| CH₄ gas | 160 | 2.3×10⁻⁵ | 2.1×10⁻⁴ | 1.000000 |
| C₂H₄ gas | 192 | 3.0×10⁻⁵ | 2.5×10⁻⁴ | 1.000000 |
| C₂H₆ gas | 256 | 2.2×10⁻⁵ | 1.6×10⁻⁴ | 1.000000 |
| CO₂ gas | 96 | 1.4×10⁻⁴ | 6.9×10⁻⁴ | 1.000000 |

Energies (step 0): LES total matches PyTorch to −3×10⁻⁶ meV/atom (1,002) and
−7×10⁻⁶ meV/atom (1,500); the six gas molecules agree to ≤1.5×10⁻³ meV/atom
(the totals are O(10⁵–10⁶) eV because of the checkpoint's element baselines).

## Reproduce

- Build: `cmake --build symmetrix/build-lammps-99ca96e1225f5a30 -j 64`
  with `LIBRARY_PATH=/opt/matos-tools/conda/envs/matos-compute/lib`
  (the final link needs `-lblas`).
- Force check: `bench/q4_maceles_off/validate/run_check.sh`
  (`REQUIRE_FIX=1` after a rebuild).
- Six gas molecules: `bench/q4_maceles_off/six_molecules/run_six.sh` +
  `compare_six.py`.
- Six 3,000-atom systems: `bench/q4_maceles_off/six_3000/` (data + PyTorch
  references); q4 speed task `t-20261030183607-mjzt7`.

## Remaining

Clean-A100 speed for the six 3,000-atom systems and the 100 ps water c1/c2 run
are in flight; numbers will be appended.
