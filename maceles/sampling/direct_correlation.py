"""One- and two-body direct correlation functions (PRL 134, 056103).

In the PRL (Dijkman et al., Phys. Rev. Lett. 134, 056103 (2025)) the objects
called c1 and c2 are functional derivatives of the excess free-energy
functional:

    c1(r; [rho]) = -beta * delta F_exc[rho] / delta rho(r)
    c2(r, r'; [rho]) = delta c1(r) / delta rho(r')
                     = -beta * delta^2 F_exc[rho] / delta rho(r) delta rho(r')

These are NOT the slope/intercept of a linear energy fit.  For a homogeneous
isotropic fluid c2(r) follows from the Ornstein-Zernike relation

    h(r) = g(r) - 1
    h(k) = 4 pi int r^2 h(r) sin(kr)/(kr) dr
    c2(k) = h(k) / (1 + rho h(k))
    c2(r) = 1/(2 pi^2) int k^2 c2(k) sin(kr)/(kr) dk

and c1 is a constant fixed by the excess chemical potential,

    c1 = -beta mu_ex = ln <exp(-beta Delta U)>   (excess convention).

The full one-body direct correlation (including the ideal term) is
c1_full = -ln rho + c1_ex.  See docs/DIRECT_CORRELATION.md.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def c2_from_g(r: Sequence[float], g: Sequence[float], rho: float,
              kmax: float | None = None) -> tuple[np.ndarray, np.ndarray,
                                                  np.ndarray, np.ndarray]:
    """Return (r, c2(r), k, c2(k)) from a radial distribution function."""
    r = np.asarray(r, dtype=float)
    g = np.asarray(g, dtype=float)
    if r.ndim != 1 or g.shape != r.shape or len(r) < 8:
        raise ValueError("r and g must be 1-D arrays of equal length")
    dr = float(r[1] - r[0])
    h = g - 1.0
    # Transform on a padded radial grid; the tail of h is zero by construction.
    rpad = np.arange(0.0, 2.0 * r[-1] + dr, dr)
    hpad = np.zeros_like(rpad)
    hpad[:len(r)] = h
    L = 2.0 * (r[-1] + dr)
    if kmax is None:
        kmax = np.pi / dr
    dk = 2.0 * np.pi / L
    k = np.arange(0.0, kmax + 0.5 * dk, dk)
    kr = np.outer(k, rpad)
    sinc = np.ones_like(kr)
    np.divide(np.sin(kr), kr, out=sinc, where=kr != 0.0)
    hk = 4.0 * np.pi * dr * ((rpad ** 2 * hpad) @ sinc.T)
    ck = hk / (1.0 + rho * hk)
    cr = (dk / (2.0 * np.pi ** 2)) * ((k ** 2 * ck) @ sinc)
    return r, cr[:len(r)], k, ck


def c1_from_widom(frames: Sequence[Any], calculator: Any,
                  probe: Any, temperature_K: float,
                  n_insertions: int = 1000, seed: int = 20261023,
                  overlap_cutoff_A: float = 1.2) -> dict[str, float]:
    """Estimate the excess c1 of a homogeneous fluid by Widom insertion.

    `probe` is an ASE Atoms object of the inserted molecule.  Insertions with
    a host-probe distance below `overlap_cutoff_A` are skipped (their
    Boltzmann factor is effectively zero).  This is a utility for dilute or
    moderately dense fluids; dense-water c1 needs a soft-core/TI treatment.
    """
    from ase import units
    from scipy.spatial.transform import Rotation

    beta = 1.0 / (units.kB * temperature_K)
    rng = np.random.default_rng(seed)
    host = list(frames)
    if not host:
        raise ValueError("no frames")
    host_e = []
    for atoms in host:
        atoms = atoms.copy()
        atoms.calc = calculator
        host_e.append(float(atoms.get_potential_energy()))

    log_w = []
    tried = 0
    for i in range(n_insertions):
        atoms = host[i % len(host)].copy()
        cell = np.asarray(atoms.cell)
        pos = rng.random(3) @ cell
        rot = Rotation.random(random_state=rng).as_matrix()
        mol = probe.copy()
        mol.set_positions(mol.get_positions() @ rot.T)
        mol.translate(pos)
        combined = atoms + mol
        combined.set_cell(cell)
        combined.set_pbc(True)
        # cheap overlap precheck
        d = combined.get_all_distances(mic=True)[:len(atoms), len(atoms):]
        if d.size and d.min() < overlap_cutoff_A:
            continue
        combined.calc = calculator
        e = float(combined.get_potential_energy())
        if not np.isfinite(e):
            continue
        dU = e - host_e[i % len(host)]
        log_w.append(-beta * dU)
        tried += 1
    if not log_w:
        return {"c1_ex": float("nan"), "mu_ex_eV": float("nan"),
                "n_used": 0, "n_insertions": n_insertions,
                "temperature_K": temperature_K}
    log_w = np.asarray(log_w)
    m = log_w.max()
    c1 = float(m + np.log(np.mean(np.exp(log_w - m))))
    return {"c1_ex": c1, "mu_ex_eV": float(-c1 / beta),
            "n_used": int(tried), "n_insertions": int(n_insertions),
            "temperature_K": temperature_K}


def direct_correlation_report(frames: Sequence[Any], rho: float,
                              outdir: str | Path,
                              rmax: float = 8.0, bins: int = 80) -> dict:
    """Compute c2(r) from the O-O RDF of a frame set and write JSON/plot."""
    from sampling.off_polar import OffPolarReweighter
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r, g = OffPolarReweighter.rdf(frames, pair=("O", "O"), rmax=rmax,
                                  bins=bins)
    r, c2r, k, c2k = c2_from_g(r, g, rho)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "kind": "direct_correlation_c2",
        "rho_A3": rho,
        "n_frames": len(frames),
        "r_A": r.tolist(),
        "g_OO": g.tolist(),
        "c2_r": c2r.tolist(),
        "k_A_inv": k.tolist(),
        "c2_k": c2k.tolist(),
    }
    (out / "c2.json").write_text(json.dumps(payload, indent=1))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(r, g)
    ax[0].set_xlabel("r (Å)")
    ax[0].set_ylabel("g$_{OO}$(r)")
    ax[0].set_title("O–O RDF")
    ax[1].plot(r, c2r)
    ax[1].set_xlabel("r (Å)")
    ax[1].set_ylabel("c$_2$(r)")
    ax[1].set_title(f"direct correlation, rho={rho:.4g} Å$^{{-3}}$")
    fig.tight_layout()
    fig.savefig(out / "c2.png", dpi=150)
    plt.close(fig)
    return {"r": r, "g": g, "c2_r": c2r, "k": k, "c2_k": c2k}
