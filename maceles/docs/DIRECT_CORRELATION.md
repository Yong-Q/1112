# Obtaining c1 and c2 (PRL 134, 056103)

> **Audit notice (2026-10-23):** The RDF shell defect was fixed in remote commit
> `865f203`. Widom averaging and the OZ radial-grid convention still require correction.
> O–O scalar OZ output must not be advertised as the full molecular
> position-orientation c2. See the [audit](AUDIT_2026-10-23.md) and
> [pending implementation plan](superpowers/plans/2026-10-23-polar-c1c2-validation.md).
> The historical numerical outputs below are not certified labels.

The PRL by Dijkman et al., *Learning Neural Free-Energy Functionals with
Pair-Correlation Matching*, Phys. Rev. Lett. **134**, 056103 (2025),
uses **c1 and c2 as functional derivatives of the excess free-energy
functional**, not as linear energy-fit coefficients.

\[
c_1(\mathbf r;[\rho])
=-\beta\frac{\delta F_{\rm exc}[\rho]}{\delta \rho(\mathbf r)}
\]

\[
c_2(\mathbf r,\mathbf r';[\rho])
=\frac{\delta c_1(\mathbf r;[\rho])}{\delta \rho(\mathbf r')}
=-\beta\frac{\delta^2 F_{\rm exc}[\rho]}
{\delta\rho(\mathbf r)\,\delta\rho(\mathbf r')}
\]

- **c1** is the one-body direct correlation function.  A model trained with
  c1 data needs inhomogeneous density profiles; in the PRL this is the
  \(F_\theta^{(1)}\) variant.
- **c2** is the two-body direct correlation function.  For a homogeneous
  fluid it follows from the bulk radial distribution function through the
  Ornstein–Zernike relation; the PRL's pair-correlation matching is
  \(c_2\)-matching and needs only bulk \(g(r)\).

## c2 from a trajectory

For a homogeneous isotropic fluid:

\[
h(r)=g(r)-1,
\]

\[
h(k)=4\pi\int r^2 h(r)\frac{\sin kr}{kr}\,dr,
\qquad
c_2(k)=\frac{h(k)}{1+\rho h(k)},
\]

\[
c_2(r)=\frac{1}{2\pi^2}\int k^2 c_2(k)\frac{\sin kr}{kr}\,dk.
\]

The module implements this:

```python
from sampling.off_polar import OffPolarReweighter
from sampling.direct_correlation import c2_from_g

# Keep rmax <= L/2: beyond that the minimum-image convention folds g(r).
rmax = min(8.0, 0.5 * min(frames[0].cell.cellpar()[:3]))
r, g = OffPolarReweighter.rdf(frames, pair=("O", "O"), rmax=rmax, bins=80)
rho = n_O / volume
r, c2r, k, c2k = c2_from_g(r, g, rho)
```

`sampling/direct_correlation.py` also provides
`direct_correlation_report(frames, rho, outdir)`, which writes `c2.json` and
`c2.png`.  For a 96-atom box (\(L=11.7\) Å) this means `rmax=5.75` Å, not 8 Å.
The RDF shell volume was corrected on 2026-10-23; see
[`EXPERIMENTS.md`](EXPERIMENTS.md) §3.5.

## c1 from a trajectory

For a homogeneous fluid c1 is a constant fixed by the excess chemical
potential:

\[
c_1=-\beta\mu_{\rm ex}
=\ln\left\langle e^{-\beta\Delta U}\right\rangle
\]

in the excess convention.  The full one-body function including the ideal
term is \(c_{1,\rm full}=-\ln\rho+c_{1,\rm ex}\).  Widom insertion is
implemented as `c1_from_widom(frames, calculator, probe, temperature_K, ...)`.

Dense water is a hard case for Widom insertion: most random insertions
overlap, so the average is dominated by rare cavities.  For production c1 in
dense fluids use a soft-core/repulsive-core insertion, thermodynamic
integration, or a free-energy functional derivative.

## What is easy and what is not

| Quantity | Cost | Notes |
|---|---|---|
| c2(r) from bulk g(r) | cheap | one RDF + OZ inversion; needs a well-converged g(r) |
| c1 (constant) via Widom | moderate to expensive | many insertions in dense fluids |
| c1(r) in an inhomogeneous system | expensive | needs density profiles under external potentials and a functional derivative, or a trained functional |

A 96-atom, 5 ps trajectory is **not** enough for a quantitative c2: the
small box and short sampling make \(S(0)\) and the low-k limit noisy.  Use
5–10 ns and a box large enough that \(L\gtrsim2r_{\rm cut}\) for production
c2/c1.

## Relation to the reweighting workflow

The reweighting manifest's `linear.slope` and `linear.intercept_eV` are the
coefficients of a local linear energy fit.  They are **not** the PRL c1/c2.
The PRL objects are functions/functionals of density; they cannot be reduced
to a single per-trajectory offset.

## References

1. J. Dijkman, M. Dijkstra, R. van Roij, M. Welling, J.-W. van de Meent,
   B. Ensing, *Learning Neural Free-Energy Functionals with Pair-Correlation
   Matching*, Phys. Rev. Lett. **134**, 056103 (2025).
   DOI: 10.1103/PhysRevLett.134.056103; arXiv:2403.15007.
2. R. Evans, *Adv. Phys.* **28**, 143 (1979) (classical DFT and direct
   correlation functions).
3. J.-P. Hansen, I. R. McDonald, *Theory of Simple Liquids* (Ornstein–Zernike
   relation and c2).
