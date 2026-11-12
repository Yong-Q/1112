#!/usr/bin/env python3
"""Build a tiny water box + one-row manifest for an installation smoke test.

    python -m sampling.make_smoke_box --out smoke

Creates ``smoke/data/H2O_64.lammps`` (64 molecules, 0.0334 A^-3) and
``smoke/manifest.csv`` ready for ``sampling.les_batch --run``.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

N_MOL = 64
RHO = 0.0334          # molecules / A^3, ambient water
R_OH = 0.9572
ANGLE = 104.52
MASSES = {1: 12.011, 2: 1.008, 3: 14.007, 4: 15.999}


def water_molecule(rng):
    half = np.radians(ANGLE) / 2
    o = np.zeros(3)
    h1 = R_OH * np.array([np.sin(half), 0.0, np.cos(half)])
    h2 = R_OH * np.array([-np.sin(half), 0.0, np.cos(half)])
    mol = np.vstack([o, h1, h2])
    # random orientation
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return mol @ rot.T


def build(out: Path, seed=7):
    rng = np.random.default_rng(seed)
    L = (N_MOL / RHO) ** (1.0 / 3.0)
    n_side = int(np.ceil(N_MOL ** (1 / 3)))
    grid = (np.arange(n_side) + 0.5) * L / n_side
    centers = np.array([[x, y, z] for x in grid for y in grid for z in grid])
    centers = centers[:N_MOL]
    pos, typ = [], []
    for c in centers:
        mol = water_molecule(rng) + c
        pos.extend(mol)
        typ.extend([4, 2, 2])
    pos = np.array(pos) % L
    data = out / "data" / "H2O_64.lammps"
    data.parent.mkdir(parents=True, exist_ok=True)
    with open(data, "w") as fh:
        fh.write(f"{N_MOL}-molecule smoke box; L={L:.4f} A\n\n")
        fh.write(f"{len(pos)} atoms\n4 atom types\n\n")
        fh.write(f"0.0 {L:.10f} xlo xhi\n0.0 {L:.10f} ylo yhi\n"
                 f"0.0 {L:.10f} zlo zhi\n\nMasses\n\n")
        for t in range(1, 5):
            fh.write(f"{t} {MASSES[t]:.3f}\n")
        fh.write("\nAtoms # molecular\n\n")
        for i, (t, p) in enumerate(zip(typ, pos), 1):
            mol = (i - 1) // 3 + 1
            fh.write(f"{i} {mol} {t} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n")
    man = out / "manifest.csv"
    with open(man, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sim_id", "data", "traj", "species", "temperature",
                    "timestep", "steps", "dump_every", "density", "c2_pair",
                    "seed", "elements"])
        w.writerow(["H2O-SMOKE-T300-r00-0.0334", str(data.resolve()), "",
                    "H2O", 300.0, 0.0005, 500, 100, f"{RHO:.4f}", "O O",
                    12345, "C H N O"])
    print(f"wrote {data} ({len(pos)} atoms, L={L:.3f} A) and {man}")
    return man


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="smoke")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    build(Path(args.out), args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
