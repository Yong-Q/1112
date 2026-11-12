#!/usr/bin/env python3
"""PRL-style density scan for the six molecules.

For each species: temperature T = 1.55 Tc (supercritical), 10 densities
log-spaced between 0.2 and 2.2 rho_c, starting from the 3000-atom dense box
isotropically rescaled to each target density.  Writes LAMMPS data files and
a manifest for sampling/les_batch.py.

Usage:
    python sampling/make_density_scan.py --outroot maceles_density_scan
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from ase.io import read

SIX_ROOT = Path(
    "/workspace/group/shared/bza/symmetrix-polar/bench/q4_maceles_off/six_3000")
# Tc (K), rho_c (molecules/A^3), c2 pair
PRL = {
    "H2O": (647.1, 0.01076, "O O"),
    "NH3": (405.4, 0.00831, "N N"),
    "CH4": (190.6, 0.00608, "C C"),
    "C2H4": (282.3, 0.00459, "C C"),
    "C2H6": (305.3, 0.00407, "C C"),
    "CO2": (304.1, 0.00640, "C O"),
}
NMOL = {"H2O": 667, "NH3": 500, "CH4": 500, "C2H4": 500,
        "C2H6": 500, "CO2": 667}
# Atoms per molecule; every source box stores molecules as contiguous
# atom blocks whose first atom is the heavy centre (O/N/C).
NATS = {"H2O": 3, "NH3": 4, "CH4": 5, "C2H4": 6, "C2H6": 8, "CO2": 3}
COLUMNS = ["sim_id", "data", "traj", "species", "temperature", "timestep",
           "steps", "dump_every", "density", "c2_pair", "seed", "elements"]


def rescaled_data(src: Path, dst: Path, rho_target: float, n_mol: int,
                  natoms_per_mol: int) -> float:
    atoms = read(src, format="lammps-data", atom_style="atomic")
    atoms.wrap()
    vol0 = atoms.get_volume()
    rho0 = n_mol / vol0
    scale = (rho0 / rho_target) ** (1.0 / 3.0)
    # Rigid-molecule rescale: scale only the molecular anchor (first atom of
    # each contiguous block) and keep every intramolecular offset fixed.
    # `atoms.set_cell(..., scale_atoms=True)` would stretch bonds by `scale`
    # (e.g. N-H 1.01 -> 1.58 A), which makes the ML potential explode.
    pos = atoms.get_positions().reshape(n_mol, natoms_per_mol, 3)
    sym = np.array(atoms.get_chemical_symbols()).reshape(n_mol, natoms_per_mol)
    L0 = np.array(atoms.cell.lengths())
    rel = pos - pos[:, 0:1, :]
    rel -= L0 * np.round(rel / L0)          # minimum image to the anchor
    anchor = pos[:, 0, :] * scale
    new_pos = (anchor[:, None, :] + rel).reshape(-1, 3)
    atoms.set_cell(atoms.cell * scale)
    atoms.set_positions(new_pos)
    atoms.wrap()
    # Bond lengths must survive the rescale exactly.
    r_before = np.linalg.norm(rel, axis=2)
    p_after = atoms.get_positions().reshape(n_mol, natoms_per_mol, 3)
    L1 = np.array(atoms.cell.lengths())
    d_after = p_after - p_after[:, 0:1, :]
    d_after -= L1 * np.round(d_after / L1)
    r_after = np.linalg.norm(d_after, axis=2)
    assert np.allclose(r_before, r_after, atol=1e-2), "bonds changed"
    dst.parent.mkdir(parents=True, exist_ok=True)
    masses = {1: 12.011, 2: 1.008, 3: 14.007, 4: 15.999}
    L = atoms.cell.lengths()
    assert np.allclose(L, L[0], atol=1e-6)
    with open(dst, "w") as fh:
        fh.write(f"{n_mol}-molecule density-scan frame rho={rho_target:.6g}\n\n")
        fh.write(f"{len(atoms)} atoms\n4 atom types\n\n")
        fh.write(f"0.0 {L[0]:.10f} xlo xhi\n")
        fh.write(f"0.0 {L[0]:.10f} ylo yhi\n")
        fh.write(f"0.0 {L[0]:.10f} zlo zhi\n\nMasses\n\n")
        for t in range(1, 5):
            fh.write(f"{t} {masses[t]:.3f}\n")
        fh.write("\nAtoms # molecular\n\n")
        for i, (s, p) in enumerate(zip(atoms.get_chemical_symbols(),
                                       atoms.get_positions()), 1):
            typ = {"C": 1, "H": 2, "N": 3, "O": 4}[s]
            mol = (i - 1) // natoms_per_mol + 1
            fh.write(f"{i} {mol} {typ} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n")
    return rho0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outroot", default="maceles_density_scan")
    ap.add_argument("--src-root", default=str(SIX_ROOT),
                    help="source box root (six_3000), overrides the remote "
                         "default for local generation")
    ap.add_argument("--n-densities", type=int, default=10)
    ap.add_argument("--lo", type=float, default=0.02)  # ρ* = ρσ³ (PRL)
    ap.add_argument("--hi", type=float, default=0.67)
    ap.add_argument("--steps", type=int, default=2000000)
    ap.add_argument("--dump-every", type=int, default=1000)
    args = ap.parse_args(argv)
    outroot = Path(args.outroot)
    src_root = Path(args.src_root)
    data_root = outroot / "data"
    rows = []
    for species, (tc, rhoc, pair) in PRL.items():
        temperature = round(2.0 / 1.312 * tc, 1)  # PRL T*=2, full-LJ T*c=1.312
        src = src_root / species / "data.lammps"
        rho_stars = np.linspace(args.lo, args.hi, args.n_densities)
        rhos = rho_stars * rhoc / 0.316  # ρ = ρ* ρc / ρ*c, full-LJ ρ*c=0.316
        for i, rho in enumerate(rhos):
            rho = float(rho)
            dst = data_root / species / f"r{i:02d}_{rho:.6f}.lammps"
            rescaled_data(src, dst, rho, NMOL[species], NATS[species])
            rows.append({
                "sim_id": f"{species}-PRL-T{temperature:g}-r{i:02d}-{rho:.6f}",
                "data": str(dst.resolve()),
                "traj": "", "species": species,
                "temperature": temperature, "timestep": 0.0005,
                "steps": args.steps, "dump_every": args.dump_every,
                "density": f"{rho:.6f}", "c2_pair": pair, "seed": 0,
                "elements": "C H N O",
            })
        print(f"{species}: T={temperature:g} K, rho_c={rhoc:g}, "
              f"{len(rhos)} densities {rhos[0]:.6g}..{rhos[-1]:.6g}")
    manifest = outroot / "density_scan.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
