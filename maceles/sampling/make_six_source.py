#!/usr/bin/env python3
"""Generate the six dense source boxes (six_3000) locally.

Each box: NMOL molecules at the SIX reference density, molecules in contiguous
atom blocks whose first atom is the heavy centre (O/N/C), LAMMPS types
C=1 H=2 N=3 O=4, written in atomic style (the density scan rescales these).
"""
from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import molecule as ase_molecule
from scipy.spatial.transform import Rotation

SIX = {
    "H2O": (0.0334, 667),
    "NH3": (0.0241, 500),
    "CH4": (0.0158, 500),
    "C2H4": (0.0123, 500),
    "C2H6": (0.0105, 500),
    "CO2": (0.0137, 667),
}
TYPE_OF_Z = {6: 1, 1: 2, 7: 3, 8: 4}
MASSES = {1: 12.011, 2: 1.008, 3: 14.007, 4: 15.999}


def build_box(species: str, n_mol: int, rho: float, seed: int) -> Atoms:
    rng = np.random.default_rng(seed)
    mol = ase_molecule(species)
    base = mol.get_positions() - mol.get_positions().mean(axis=0)
    numbers = mol.numbers
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
        pos.append(base @ rot.T + c)
    return Atoms(numbers=np.tile(numbers, n_mol),
                 positions=np.vstack(pos) % L,
                 cell=[L, L, L], pbc=True)


def write_lammps_data(atoms: Atoms, path, n_mol: int) -> None:
    syms = atoms.get_chemical_symbols()
    types = [TYPE_OF_Z[z] for z in atoms.numbers]
    L = atoms.cell.lengths()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(f"{n_mol}-molecule source box; L={L[0]:.6f} A\n\n")
        fh.write(f"{len(atoms)} atoms\n4 atom types\n\n")
        fh.write(f"0.0 {L[0]:.10f} xlo xhi\n0.0 {L[1]:.10f} ylo yhi\n"
                 f"0.0 {L[2]:.10f} zlo zhi\n\nMasses\n\n")
        for t in range(1, 5):
            fh.write(f"{t} {MASSES[t]:.3f}\n")
        fh.write("\nAtoms # atomic\n\n")
        for i, (t, p) in enumerate(zip(types, atoms.get_positions()), 1):
            fh.write(f"{i} {t} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--outroot", default="six_3000")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    from pathlib import Path
    out = Path(args.outroot)
    for species, (rho, nmol) in SIX.items():
        box = build_box(species, nmol, rho, args.seed)
        write_lammps_data(box, out / species / "data.lammps", nmol)
        print(f"{species}: {len(box)} atoms, L={box.cell.lengths()[0]:.3f} A, "
              f"rho={rho}")


if __name__ == "__main__":
    main()
