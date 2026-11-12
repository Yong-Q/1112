# MACE-OFF → MACE-POLAR reweighting: theory and feasibility

> **Superseded interpretation (2026-10-27):** This is a historical note. Its
> categorical ESS saturation, size-failure, entropy equivalence and coverage
> claims are not current conclusions. Read the [revised report](reports/REWEIGHT_VARIANCE_REVISED_2026-10-27.pdf)
> and [primary-PDF formula check](reports/FORMULA_CROSSCHECK_PRIMARY_PDFS_2026-10-27.md).
> The student scores on OFF frames are not a student-ensemble ESS. The actual
> MD/re-evaluation energy consistency remains unresolved; no c1/c2 certification
> follows from the stored diagnostics.

> **Audit notice (2026-10-23):** Section 3 contains an erf expression whose
> monotonicity contradicts the accompanying interpretation. The ESS threshold,
> entropy equivalence and categorical long-run failure claims also require
> revision. The normalized importance-weight identity remains applicable under
> its ensemble/overlap assumptions. See the [audit](AUDIT_2026-10-23.md) and
> [pending correction plan](superpowers/plans/2026-10-23-polar-c1c2-validation.md).

**Date:** 2026-10-23
**Question:** can a long MACE-OFF trajectory be re-evaluated with MACE-POLAR on
a finite frame set and reweighted to recover MACE-POLAR ensemble statistics?
How many frames are needed for 1 ns, and what is the role of a constant
offset / linear `slope, intercept` correction?
**Scope:** theory, a practical protocol, and a short end-to-end feasibility run.
No production 1 ns trajectory was launched for this note.

---

## 1. Exact statement (importance sampling)

Let the source and target Boltzmann distributions be

\[
p_{\rm OFF}(x)=\frac{e^{-\beta V_{\rm OFF}(x)}}{Z_{\rm OFF}},\qquad
p_{\rm POLAR}(x)=\frac{e^{-\beta V_{\rm POLAR}(x)}}{Z_{\rm POLAR}}.
\]

For any observable \(A(x)\), using samples \(x_i\sim p_{\rm OFF}\),

\[
\boxed{
\langle A\rangle_{\rm POLAR}
=
\frac{\sum_i A_i\,e^{-\beta\Delta V_i}}
     {\sum_i e^{-\beta\Delta V_i}},
\qquad
\Delta V_i=V_{\rm POLAR}(x_i)-V_{\rm OFF}(x_i).
}
\]

This is exact in the limit of infinite, equilibrated samples and is the
standard free-energy-perturbation / importance-sampling identity
[1,2]. The same weights give the free-energy difference [1]

\[
\Delta A=-k_BT\ln\left\langle e^{-\beta\Delta V}\right\rangle_{\rm OFF}.
\]

The only question is statistical: do the OFF samples cover the regions that
dominate the POLAR average? The MBAR/BAR estimators [4,5] are the
lower-variance alternatives when several overlapping states are available.

## 2. What the linear slope/intercept can and cannot do

> Naming note: these are the slope \(s\) and intercept \(b\) of a linear
> energy fit.  The PRL direct correlation functions \(c_1, c_2\) are different
> objects; see `DIRECT_CORRELATION.md`.

If one assumes a global linear relation

\[
V_{\rm POLAR}(x)=s\,V_{\rm OFF}(x)+b,
\]

then

\[
e^{-\beta(sV_{\rm OFF}+b-V_{\rm OFF})}
=e^{-\beta(s-1)V_{\rm OFF}}\,e^{-\beta b}.
\]

- **`b` is a constant offset. It cancels in every normalized ensemble
  average.** It shifts the absolute energy and the free-energy zero, but not
  RDFs, densities, or average energies.
- **`s` rescales the source Hamiltonian**, i.e. it acts like an effective
  temperature \(T/s\). A global slope is not a substitute for the
  configuration-dependent residual.
- In practice a linear fit \(V_{\rm POLAR}\approx sV_{\rm OFF}+b\) leaves a
  residual \(\delta V(x)\). The correct weight is
  \(e^{-\beta[(s-1)V_{\rm OFF}(x)+\delta V(x)]}\); dropping
  \(\delta V\) is an approximation, not an identity. The decomposition is
  just the FEP identity [1] rewritten around a fitted linear response; the
  residual term is what the mean-energy-gap approximation of [6] retains in
  a bin-resolved form.

**The physically meaningful quantity is the fluctuation of \(\Delta V\), not
the constant offset.** The absolute MACE-OFF and MACE-POLAR energies can
differ by hundreds of eV because of different per-element reference energies
(the measured 1,296-atom water offset is of order \(10^2\) eV/atom). That
constant is harmless after normalization, but it must be removed before
exponentiating for numerical stability.

## 3. Why direct reweighting collapses at large N

The weights are exponential in an extensive quantity. With
\(\sigma_\beta=\mathrm{std}(\beta\Delta V)\), most samples fall in a window of
width \(\sim\sigma_\beta\) around the mean. The fraction of the weight
distribution that can be sampled is roughly

\[
f\sim \operatorname{erf}\!\left(\frac{3\sigma_\beta}{\sqrt{2}}\right),
\]

so \(f\approx1\) for \(\sigma_\beta\le1\), \(f\approx0.5\) at
\(\sigma_\beta\approx3\), and \(f\to0\) for \(\sigma_\beta>5\)
[6, SI §7–8]. The effective sample size is the standard importance-sampling
diagnostic [1,5,6]

\[
\mathrm{ESS}=\frac{(\sum_i w_i)^2}{\sum_i w_i^2}
=\frac{1}{\sum_i \tilde w_i^2},
\]

where \(\tilde w_i\) are normalized weights. A useful rule of thumb is
**ESS ≥ 10 % of frames**; below that, the reweighted mean is controlled by a
handful of configurations. The reweighting-entropy score of [6] is the same
diagnostic expressed as a normalized Shannon entropy.

For a toy harmonic model where every mode is shifted by \(x_0\), the paper
shows \(\sigma_\beta\propto x_0\sqrt{N}\): even a 1 pm shift becomes
unreliable beyond \(N\approx100\) internal degrees of freedom. This is why
direct exponential reweighting between two MLIPs fails for systems of
practical size (hundreds to thousands of atoms), even when their per-atom
energy errors look small.

## 4. The mean energy-gap / Gaussian approximation

When \(\sigma_\beta\) is too large for direct reweighting, two stable
approximations are used [Majumdar et al. 2026]:

- **Gaussian approximation** for a free-energy profile along a coordinate
  \(z\):
  \[
  A_{\rm POLAR}(z)\approx A_{\rm OFF}(z)
  +\langle\Delta V\rangle_{{\rm OFF},z}
  -\frac{\beta}{2}\sigma^2_{\Delta V,z}.
  \]
- **Energy-only / mean-gap approximation** (neglect the entropic width):
  \[
  A_{\rm POLAR}(z)\approx A_{\rm OFF}(z)
  +\langle\Delta V\rangle_{{\rm OFF},z}.
  \]

These are corrections to a **free-energy profile** (or a bin-resolved
observable), not a license to replace the full weight by a single global
constant. For a global ensemble average, a constant mean gap cancels; the
mean-gap approximation only helps when the gap is resolved along a collective
variable or when the residual fluctuation is genuinely negligible.

## 5. Practical protocol

1. **Source trajectory.** Run MACE-OFF NVT/NPT at the target state point.
   Save every 1 ps (1,000 frames/ns) or every 0.5 ps (2,000 frames/ns).
   For water at 300 K, energy/force correlation times are ~0.05–0.2 ps, so
   1 ps spacing is effectively decorrelated for means; 0.5 ps gives a margin.
2. **Label frames.** Re-evaluate the saved frames with MACE-POLAR. Labeling is
   cheap relative to MD: at 5,184 atoms the exact LAMMPS engine is
   \(128.99\,\mu s/atom/step\), i.e. 0.669 s/frame; 2,000 frames cost about
   22 minutes. The PyTorch `mace_polar` calculator is slower (measured
   ~0.75 s/frame at 1,296 atoms, ~4.6 ms/atom); prefer the engine for large
   frame sets. The 1 ns OFF trajectory itself costs 6.4 h at 4.473
   µs/atom/step in the linear engine, or much longer with the native PyTorch
   MACE-OFF calculator (~0.22 s/step at 1,296 atoms in this container).
3. **Diagnose before trusting.** Compute per-frame \(\Delta V\), its
   per-atom mean and std, \(\sigma_\beta=\mathrm{std}(\beta\Delta V)\),
   ESS, the linear fit \((s,b,R^2)\), and the force MAE. Also compare
   OFF and POLAR RDFs/densities.
4. **Decision gate.**
   - \(\sigma_\beta\le3\) and ESS ≥ 10 % → direct reweighting is usable.
   - \(3<\sigma_\beta\lesssim5\) → direct weights are marginal; use the
     Gaussian/mean-gap correction for free-energy profiles, or BAR/MBAR.
   - \(\sigma_\beta\gg5\) or ESS < 1 % → do not report reweighted ensemble
     averages. Run short MACE-POLAR trajectories or use a distilled model for
     the long-time statistics.
5. **Never use a global constant offset as the correction.** `b` cancels;
   `s` only helps if the residual is negligible and must be validated on a
   held-out set of frames.

### 5.1 Formula provenance

| Equation / statement | Source |
|---|---|
| Importance-sampling identity and normalized weights | [1] Eq. (1)–(2); [2] |
| Free-energy perturbation \(\Delta A=-k_BT\ln\langle e^{-\beta\Delta V}\rangle\) | [1] |
| Constant offset cancels; slope `s` acts as an effective temperature | algebra from [1]; linear-response discussion in [6] |
| \(\sigma_\beta\) coverage and the \(\sigma_\beta\lesssim3\) reliability rule | [6] SI §7–8 |
| ESS \(=(\sum w)^2/\sum w^2\) | standard IS diagnostic, used in [5,6] |
| Gaussian and mean-energy-gap corrections | [6] §4.2.3–4.2.4 |
| MBAR/BAR alternatives | [4,5] |
| Reweighting entropy score | [6] §4.2.2 |

## 6. Feasibility test (this note)

System: 32 H\(_2\)O (96 atoms), 11.696 Å box, density 0.02 molecules/Å³,
MACE-OFF23 medium (source) and MACE-POLAR-1-M (target). A 5 ps NVT trajectory
was generated, the first 1 ps was discarded, and 36 production frames were
re-evaluated. This is a feasibility probe, not a 1 ns production run.

### 6.1 Results

| Quantity | Value |
|---|---:|
| ΔE/atom mean (constant offset) | 416.62 meV/atom |
| ΔE/atom std | 1.166 meV/atom |
| \(\sigma_\beta\) | **3.89** |
| ESS | **3.63 / 36 = 10.1 %** |
| verdict | **marginal** |
| slope \(s\) / intercept \(b\) | 0.9452 / −3,606.84 eV |
| \(R^2\) / residual | 0.9866 / 1.03 meV/atom |
| force MAE | 104.7 meV/Å |

The thermostat setpoint was 300 K; the mean instantaneous temperature after
1 ps was 334 K (weak friction 0.01).  Production runs should use stronger
coupling and a longer equilibration.

### 6.2 Verdict

At 96 atoms the direct exponential average is already marginal: only ~4
effective frames survive, despite the constant offset being removed.  The
linear fit is highly correlated (\(R^2=0.987\)) but its residual (1.03
meV/atom) is almost as large as the total fluctuation (1.17 meV/atom), so a
global slope/intercept correction does not fix the overlap.

Assuming independent local fluctuations, \(\sigma_\beta\propto\sqrt N\) gives
**14.3 at 1,296 atoms** and **28.6 at 5,184 atoms**; both are far beyond the
\(\sigma_\beta\lesssim3\) threshold.  Direct reweighting of a 1 ns MACE-OFF
trajectory to MACE-POLAR is therefore **not feasible at the production sizes**
for ensemble averages.  The mean-gap/Gaussian approximation remains useful for
a free-energy profile along a collective variable, and BAR/MBAR can help when
intermediate overlapping potentials exist; otherwise run short MACE-POLAR
trajectories or a distilled target model.

## 7. Frame-count answer

For 1 ns at a 1 fs timestep:

- save every **1 ps → 1,000 frames**; use this as the default for ensemble
  means and RDFs;
- save every **0.5 ps → 2,000 frames**; use this if you also want reweighting
  diagnostics or a shorter correlation time;
- for reweighting, the limiting number is **ESS**, not the nominal frame
  count. 2,000 frames do not rescue an overlap problem.

## References

1. R. W. Zwanzig, *J. Chem. Phys.* **22**, 1420 (1954).
   DOI: 10.1063/1.1740409.
2. G. M. Torrie, J. P. Valleau, *J. Comput. Phys.* **23**, 187 (1977).
   DOI: 10.1016/0021-9991(77)90121-8.
3. A. M. Ferrenberg, R. H. Swendsen, *Phys. Rev. Lett.* **63**, 1195 (1989).
   DOI: 10.1103/PhysRevLett.63.1195.
4. C. H. Bennett, *J. Comput. Phys.* **22**, 245 (1976).
   DOI: 10.1016/0021-9991(76)90078-4.
5. M. R. Shirts, J. D. Chodera, *J. Chem. Phys.* **129**, 124105 (2008).
   DOI: 10.1063/1.2978177.
6. S. Majumdar, M. Steiner, J. C. B. Dietschreit, S. Roy, D. Willimetz,
   L. Grajciar, R. Gómez-Bombarelli, *Reweighting free energy profiles between
   universal machine learning interatomic potentials for fast consensus
   building*, arXiv:2605.15630 (2026).
