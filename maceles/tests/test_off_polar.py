import math
import numpy as np
from ase import Atoms

from sampling.off_polar import (FrameRecord, OffPolarReweighter, ReweightConfig,
                                ReweightResult)


def make_result(deltas_meV_per_atom, n_atoms=96):
    cfg = ReweightConfig(temperature_K=300.0)
    r = ReweightResult(config=cfg)
    beta = cfg.beta_eV_inv
    es = np.linspace(-1000.0, -900.0, len(deltas_meV_per_atom))
    for i, d in enumerate(deltas_meV_per_atom):
        dtot = d * n_atoms / 1000.0
        r.frames.append(FrameRecord(
            index=i, n_atoms=n_atoms, e_source_eV=float(es[i]),
            e_target_eV=float(es[i] + dtot), dE_total_eV=float(dtot),
            dE_meV_per_atom=float(d), beta_dE=float(beta * dtot),
            weight=0.0, force_mae_meV_per_A=1.0))
    w = np.exp(-(np.array([f.beta_dE for f in r.frames]) -
                 min(f.beta_dE for f in r.frames)))
    w /= w.sum()
    for f, wi in zip(r.frames, w):
        f.weight = float(wi)
    return r


def test_weights_sum_and_ess():
    r = make_result([0.0, 0.01, 0.02, 0.03])
    assert math.isclose(float(r.weights.sum()), 1.0, rel_tol=1e-12)
    assert r.ess >= 1.0
    assert r.ess <= r.n_frames
    # tiny deltas in beta*dE give ESS close to N
    assert r.ess_fraction > 0.95


def test_collapse_verdict():
    r = make_result([-1000.0, 0.0, 1000.0, 2000.0])
    assert r.sigma_beta > 5.0
    assert r.verdict == "collapse"


def test_linear_fit_recovers_slope():
    cfg = ReweightConfig()
    r = ReweightResult(config=cfg)
    es = np.linspace(-1000.0, -900.0, 20)
    for i, e in enumerate(es):
        r.frames.append(FrameRecord(
            index=i, n_atoms=96, e_source_eV=float(e),
            e_target_eV=float(2.0 * e + 5.0), dE_total_eV=float(e + 5.0),
            dE_meV_per_atom=float((e + 5.0) / 96 * 1000),
            beta_dE=float(cfg.beta_eV_inv * (e + 5.0)), weight=1.0 / 20))
    assert abs(r.linear_slope - 2.0) < 1e-10
    assert abs(r.linear_r2 - 1.0) < 1e-10


def test_observable_reweighting():
    r = make_result([0.0, 0.0, 0.0, 0.0])
    assert abs(r.reweight_observable([1.0, 2.0, 3.0, 4.0]) - 2.5) < 1e-12


def random_gas(n=500, box=20.0, seed=7):
    rng = np.random.default_rng(seed)
    return Atoms("Ar" * n, positions=rng.random((n, 3)) * box,
                 cell=[box, box, box], pbc=True)


def test_rdf_ideal_gas_normalizes_to_one():
    atoms = random_gas()
    r, g = OffPolarReweighter.rdf([atoms], pair=("Ar", "Ar"),
                                  rmax=9.0, bins=90)
    m = (r > 1.5) & (r < 8.0)
    # A random configuration has no structure: g(r) must average to 1.
    assert abs(float(np.mean(g[m])) - 1.0) < 0.05


def test_rdf_integral_matches_pair_count():
    atoms = random_gas(n=300, box=18.0, seed=11)
    r, g = OffPolarReweighter.rdf([atoms], pair=("Ar", "Ar"),
                                  rmax=8.0, bins=80)
    rho = 300.0 / 18.0 ** 3
    dr = r[1] - r[0]
    m = (r > 2.0) & (r < 4.0)
    # 4*pi*rho*int r^2 g(r) dr equals the number of neighbours per atom.
    integral = 4.0 * np.pi * rho * np.sum(r[m] ** 2 * g[m]) * dr
    d = atoms.get_all_distances(mic=True)
    iu = np.triu_indices(len(atoms), k=1)
    raw = 2.0 * np.sum((d[iu] > 2.0) & (d[iu] < 4.0)) / len(atoms)
    assert abs(integral - raw) / raw < 0.05
