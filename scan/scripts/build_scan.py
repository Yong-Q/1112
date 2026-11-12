#!/usr/bin/env python3
"""Generate the CPU MACELES-LES density scan for the six PRL molecules.

Systems (molecule counts per the six-molecule plan):
  H2O  667 molecules (2001 atoms)   NH3  500 molecules (2000 atoms)
  CO2  667 molecules (2001 atoms)   CH4  500 molecules (2500 atoms)
                                    C2H4 500 molecules (3000 atoms)
                                    C2H6 500 molecules (4000 atoms)

10 densities rho* = 0.02 .. 0.67 (uniform), ordered HIGH -> LOW, at the PRL
supercritical temperature T = 2*Tc/1.312:
  rho = rho* * rho_c / 0.316        (full-LJ rho*_c = 0.316)
  rho_c = 0.316 / sigma^3           (molecules/A^3)

Writes molecular-style LAMMPS data files (types C=1 H=2 N=3 O=4, molecules in
contiguous atom blocks whose first atom is the heavy centre) plus a
``density_scan.csv`` manifest ready for ``sampling.les_batch.py``.

Usage:
    python scripts/build_scan.py --outroot /path/to/cdft/scan
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import molecule as ase_molecule
from scipy.spatial.transform import Rotation

# species -> (Tc K, sigma A, molecules, atoms/molecule, c2 pair)
# rho_c (molecules/A^3) = RHO_STAR_C / sigma^3 = 0.316 / sigma^3.
PRL = {
    "H2O":  (647.1,  3.0848947060835306, 667, 3, "O O"),
    "CO2":  (304.1,  3.668909707550405,  667, 3, "C O"),
    "NH3":  (405.56, 3.3711712223208106, 500, 4, "N N"),
    "CH4":  (190.56, 3.7265920929791556, 500, 5, "C C"),
    "C2H4": (282.35, 4.095803722936752,  500, 6, "C C"),
    "C2H6": (305.32, 4.245573878585463,  500, 8, "C C"),
}
TYPE_OF_Z = {6: 1, 1: 2, 7: 3, 8: 4}   # C=1 H=2 N=3 O=4
MASSES = {1: 12.011, 2: 1.008, 3: 14.007, 4: 15.999}
COLUMNS = ["sim_id", "data", "traj", "species", "temperature", "timestep",
           "steps", "dump_every", "density", "c2_pair", "seed", "elements"]
RHO_STAR_LO = 0.02
RHO_STAR_HI = 0.67
RHO_STAR_C = 0.316          # full-LJ reduced critical density


def molecule_template(species: str) -> Atoms:
    mol = ase_molecule(species)
    pos = mol.get_positions() - mol.get_positions().mean(axis=0)
    return Atoms(numbers=mol.numbers, positions=pos)


def build_box(species: str, n_mol: int, rho: float, seed: int) -> Atoms:
    """Place ``n_mol`` randomly-oriented molecules on a cubic grid at ``rho``."""
    rng = np.random.default_rng(seed)
    base = molecule_template(species)
    numbers = base.numbers
    n_atoms_per_mol = len(base)
    L = (n_mol / rho) ** (1.0 / 3.0)
    n_side = int(np.ceil(n_mol ** (1.0 / 3.0)))
    spacing = L / n_side
    grid = np.array([[(i + 0.5) * spacing, (j + 0.5) * spacing,
                      (k + 0.5) * spacing]
                     for i in range(n_side) for j in range(n_side)
                     for k in range(n_side)])
    rng.shuffle(grid)
    pos = []
    for c in grid[:n_mol]:
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        rot = Rotation.from_quat(q).as_matrix()
        pos.append(base.get_positions() @ rot.T + c)
    pos = np.vstack(pos) % L
    symbols = np.tile(base.get_chemical_symbols(), n_mol)
    return Atoms(symbols=symbols, positions=pos, cell=[L, L, L], pbc=True)


def write_lammps_data(atoms: Atoms, path: Path, n_mol: int,
                      n_atoms_per_mol: int) -> None:
    syms = atoms.get_chemical_symbols()
    types = [TYPE_OF_Z[z] for z in atoms.numbers]
    L = atoms.cell.lengths()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(f"{n_mol}-molecule density-scan box; L={L[0]:.6f} A\n\n")
        fh.write(f"{len(atoms)} atoms\n4 atom types\n\n")
        fh.write(f"0.0 {L[0]:.10f} xlo xhi\n0.0 {L[1]:.10f} ylo yhi\n"
                 f"0.0 {L[2]:.10f} zlo zhi\n\nMasses\n\n")
        for t in range(1, 5):
            fh.write(f"{t} {MASSES[t]:.3f}\n")
        fh.write("\nAtoms # molecular\n\n")
        for i, (t, p) in enumerate(zip(types, atoms.get_positions()), 1):
            mol = (i - 1) // n_atoms_per_mol + 1
            fh.write(f"{i} {mol} {t} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outroot", required=True,
                    help="scan root (writes <root>/data and density_scan.csv)")
    ap.add_argument("--n-densities", type=int, default=10)
    ap.add_argument("--lo", type=float, default=RHO_STAR_LO)
    ap.add_argument("--hi", type=float, default=RHO_STAR_HI)
    ap.add_argument("--timestep", type=float, default=0.0005)
    ap.add_argument("--equil-steps", type=int, default=200000,
                    help="100 ps equilibration at 0.5 fs")
    ap.add_argument("--steps", type=int, default=2000000,
                    help="1 ns production at 0.5 fs")
    ap.add_argument("--dump-every", type=int, default=1000,
                    help="dump/thermo cadence (0.5 ps at 0.5 fs)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--species", nargs="*", default=None,
                    help="restrict to these species (default: all in PRL)")
    ap.add_argument("--manifest-name", default="density_scan.csv",
                    help="manifest filename written under outroot")
    args = ap.parse_args(argv)

    outroot = Path(args.outroot)
    data_root = outroot / "data"
    rows = []
    for species, (tc, sigma, n_mol, n_atoms_per_mol, pair) in PRL.items():
        if args.species and species not in args.species:
            continue
        temperature = round(2.0 / 1.312 * tc, 1)
        rhoc = RHO_STAR_C / sigma ** 3
        rho_stars = np.linspace(args.lo, args.hi, args.n_densities)
        # HIGH -> LOW concentration order (the scan owner's requirement).
        rho_stars = rho_stars[::-1]
        rhos = rho_stars * rhoc / RHO_STAR_C
        for i, rho in enumerate(rhos):
            rho = float(rho)
            rho_star = float(rho_stars[i])
            box = build_box(species, n_mol, rho, args.seed + i)
            dst = data_root / species / f"r{i:02d}_rho{rho_star:.4f}.lammps"
            write_lammps_data(box, dst, n_mol, n_atoms_per_mol)
            rows.append({
                "sim_id": f"{species}-T{temperature:g}-r{i:02d}-rho{rho_star:.4f}",
                "data": str(dst.resolve()),
                "traj": "",
                "species": species,
                "temperature": temperature,
                "timestep": args.timestep,
                "steps": args.steps,
                "dump_every": args.dump_every,
                "density": f"{rho:.6f}",
                "c2_pair": pair,
                "seed": args.seed,
                "elements": "C H N O",
            })
            print(f"{species}: rho*={rho_star:.4f} rho={rho:.6f} mol/A^3 "
                  f"L={box.cell.lengths()[0]:.4f} A natoms={len(box)}")
    manifest = outroot / args.manifest_name
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows (HIGH->LOW) to {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
