#!/usr/bin/env python3
"""MBAR pilot: OFF23 <-> POLAR with one intermediate lambda state.

Builds 32 H2O (96 atoms), samples lambda = 0, 0.5, 1 with mixed forces,
then runs MBAR over the three states and compares the POLAR ESS with the
one-step reweighting from lambda=0 samples.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import molecule as ase_molecule

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/results/mbar_pilot_20261028"
OUT.mkdir(parents=True, exist_ok=True)
GRAPH = "/workspace/group/shared/bza/external-repos/graph_electrostatics-0.4.0"
if GRAPH not in sys.path:
    sys.path.insert(0, GRAPH)
KB = 8.617333262e-5
T = 300.0
BETA = 1.0 / (KB * T)


def build_box(n_mol=32, rho=0.02, seed=20261028):
    rng = np.random.default_rng(seed)
    mol = ase_molecule("H2O")
    base = mol.get_positions() - mol.get_positions().mean(axis=0)
    numbers = mol.numbers
    L = (n_mol / rho) ** (1.0 / 3.0)
    n_side = int(math.ceil(n_mol ** (1.0 / 3.0)))
    spacing = L / n_side
    sites = [((i + 0.5) * spacing, (j + 0.5) * spacing, (k + 0.5) * spacing)
             for i in range(n_side) for j in range(n_side) for k in range(n_side)]
    rng.shuffle(sites)
    pos = []
    for c in sites[:n_mol]:
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        pos.append(base @ R.T + np.asarray(c) + rng.normal(scale=0.03,
                                                          size=base.shape))
    return Atoms(numbers=np.tile(numbers, n_mol),
                 positions=np.vstack(pos), cell=[L, L, L], pbc=True)


def mbar(u, nk, iters=20000, tol=1e-10):
    """u: (K, N) reduced potentials; nk: samples per state."""
    K, N = u.shape
    f = np.zeros(K)
    ln_nk = np.log(nk)
    for _ in range(iters):
        log_den = np.logaddexp.reduce(ln_nk[:, None] + f[:, None] - u, axis=0)
        f_new = -np.logaddexp.reduce(-u - log_den[None, :], axis=1)
        f_new -= f_new[0]
        if np.max(np.abs(f_new - f)) < tol:
            f = f_new
            break
        f = f_new
    log_den = np.logaddexp.reduce(ln_nk[:, None] + f[:, None] - u, axis=0)
    ess = []
    for k in range(K):
        lw = f[k] - u[k] - log_den
        w = np.exp(lw - lw.max())
        w /= w.sum()
        ess.append(1.0 / np.sum(w * w))
    return f, np.array(ess)


def main():
    from ase import units
    from mace.calculators import MACECalculator, mace_polar
    off = MACECalculator(
        model_paths="/workspace/group/shared/bza/task3_sampling_database/"
                    "mlip/models/MACE-OFF23_medium.model",
        device="cuda", default_dtype="float32")
    pol = mace_polar(
        model="/workspace/group/shared/bza/task3_sampling_database/mlip/"
              "models/mace-polar/MACE-POLAR-1-M.model",
        device="cuda", default_dtype="float64")
    atoms = build_box()
    atoms.calc = off
    from ase.optimize import FIRE
    FIRE(atoms, logfile=None).run(fmax=0.2, steps=300)

    from ase.calculators.calculator import Calculator
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    class Mixed(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self, off, pol, lam):
            super().__init__()
            self.off, self.pol, self.lam = off, pol, lam

        def calculate(self, atoms, properties, system_changes):
            super().calculate(atoms, properties, system_changes)
            e0 = float(self.off.get_potential_energy(atoms))
            f0 = self.off.get_forces(atoms)
            e1 = float(self.pol.get_potential_energy(atoms))
            f1 = self.pol.get_forces(atoms)
            self.results = {
                "energy": (1.0 - self.lam) * e0 + self.lam * e1,
                "forces": (1.0 - self.lam) * f0 + self.lam * f1,
            }

    lams = [0.0, 0.5, 1.0]
    n_eq, n_prod, dump = 2000, 4000, 100
    frames = {lam: [] for lam in lams}
    energies = {lam: [] for lam in lams}
    t0 = time.time()
    for lam in lams:
        mixed = Mixed(off, pol, lam)
        atoms.calc = mixed
        MaxwellBoltzmannDistribution(atoms, temperature_K=T)
        dyn = Langevin(atoms, timestep=0.5 * units.fs, temperature_K=T,
                       friction=0.01)
        dyn.run(n_eq)

        def record():
            e0 = float(off.get_potential_energy(atoms))
            e1 = float(pol.get_potential_energy(atoms))
            frames[lam].append(atoms.get_positions().copy())
            energies[lam].append((e0, e1))

        dyn.attach(record, interval=dump)
        dyn.run(n_prod)
        print(f"[mbar] lam {lam} done n={len(frames[lam])} "
              f"{time.time()-t0:.0f}s", flush=True)

    # assemble MBAR arrays
    all_u0, all_u1, nk = [], [], []
    for lam in lams:
        e = np.array(energies[lam])
        all_u0.append(BETA * e[:, 0])
        all_u1.append(BETA * e[:, 1])
        nk.append(len(e))
    u = np.vstack([np.concatenate(all_u0), np.concatenate(all_u1)])
    nk = np.array(nk)
    # states for MBAR are lambda=0,0.5,1: U(lam) = (1-lam) U0 + lam U1
    u_states = np.array([(1.0 - lam) * u[0] + lam * u[1] for lam in lams])
    f, ess = mbar(u_states, nk)
    # naive one-step from lambda=0 samples to lambda=1
    u0 = u_states[0][:nk[0]]
    u1 = u_states[2][:nk[0]]
    w = np.exp(-(u1 - u1.min()))
    w /= w.sum()
    ess_naive = 1.0 / np.sum(w * w)

    res = {
        "lams": lams, "nk": nk.tolist(),
        "mbar_f": f.tolist(),
        "mbar_ess": ess.tolist(),
        "mbar_ess_fraction": (ess / nk.sum()).tolist(),
        "mbar_ess_polar_within_lambda1": float(ess[2] / nk[2]),
        "naive_ess": float(ess_naive),
        "naive_ess_fraction": float(ess_naive / nk[0]),
        "n_total": int(nk.sum()),
    }
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
