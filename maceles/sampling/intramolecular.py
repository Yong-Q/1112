#!/usr/bin/env python3
"""Intramolecular geometry and molecular reorientation statistics.

For every molecule in a trajectory this collects

* all bond lengths, bond angles and dihedrals (labelled by element triplets /
  quadruplets, connectivity from covalent radii),
* the molecular orientation: unit principal axis per molecule per frame
  (linear molecules use the bond axis), its cos(theta_z) distribution and the
  rank-2 rotational correlation C(t) = <P2(u(0).u(t))>.

Works for any of the six species; the labels come out of the element symbols,
so H2O gives O-H and H-O-H, CO2 gives C-O and O-C-O (single angle, linear),
C2H6 additionally gives the H-C-C-H dihedral, and so on.
"""
from __future__ import annotations

import numpy as np

COVALENT = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66}
BOND_SCALE = 1.3
MAX_FRAMES_AXES = 400


def _labels(sym):
    """Unique bond / angle / dihedral element patterns of one molecule."""
    from itertools import combinations, permutations
    n = len(sym)
    bonds = [(i, j) for i, j in combinations(range(n), 2)]
    angles = [(j, i, k) for i, j, k in permutations(range(n), 3)]
    dihedrals = [(i, j, k, l) for i, j, k, l in permutations(range(n), 4)]
    return bonds, angles, dihedrals


def _topology(sym):
    """Connectivity of one molecule from covalent radii (returns index pairs)."""
    from itertools import combinations
    out = []
    for i, j in combinations(range(len(sym)), 2):
        cut = BOND_SCALE * (COVALENT.get(sym[i], 0.8) + COVALENT.get(sym[j], 0.8))
        out.append((i, j, cut))
    return out


def _angle(a, b, c):
    """Angle at b for vectors a-b and c-b (arrays (..., 3))."""
    v1, v2 = a - b, c - b
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    cos = (v1 * v2).sum(-1) / np.maximum(n1 * n2, 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _dihedral(p0, p1, p2, p3):
    """Signed dihedral angle (deg) with the standard praxeolitic formula."""
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / np.maximum(np.linalg.norm(b1, axis=-1)[..., None], 1e-12)
    v = b0 - (b0 * b1n).sum(-1)[..., None] * b1n
    w = b2 - (b2 * b1n).sum(-1)[..., None] * b1n
    x = (v * w).sum(-1)
    y = (np.cross(b1n, v) * w).sum(-1)
    return np.degrees(np.arctan2(y, x))


def intramolecular_stats(frames, natoms_per_mol, max_frames=200,
                         axes_frames=MAX_FRAMES_AXES):
    """Return {'bonds':..., 'angles':..., 'dihedrals':...} + axes/corr."""
    bonds, angles, dihedrals = {}, {}, {}
    axes_all = []
    sym0 = None
    n_seen = 0
    for atoms in frames:
        if n_seen >= max_frames:
            break
        sym = np.array(atoms.get_chemical_symbols())
        if sym0 is None:
            sym0 = sym[:natoms_per_mol]
        pos = atoms.get_positions().reshape(-1, natoms_per_mol, 3)
        cell = np.array(atoms.cell.lengths())
        rel = pos - pos[:, 0:1, :]
        rel -= cell * np.round(rel / cell)
        d = np.linalg.norm(rel[:, :, None, :] - rel[:, None, :, :], axis=-1)
        topo = _topology([str(s) for s in sym0])
        bonded = []
        for i, j, cut in topo:
            dij = d[:, i, j]
            if float(dij.mean()) >= cut:          # not bonded (e.g. H...H)
                continue
            bonded.append((i, j))
            lab = "-".join(sorted([str(sym0[i]), str(sym0[j])]))
            bonds.setdefault(lab, []).append(dij)
        # angles and dihedrals over the bonded graph
        nbr = {k: [] for k in range(natoms_per_mol)}
        for i, j in bonded:
            nbr[i].append(j)
            nbr[j].append(i)
        for c, ns in nbr.items():
            for a in range(len(ns)):
                for b in range(a + 1, len(ns)):
                    i, k = ns[a], ns[b]
                    lab = f"{sym0[i]}-{sym0[c]}-{sym0[k]}"
                    angles.setdefault(lab, []).append(
                        _angle(rel[:, i, :], rel[:, c, :], rel[:, k, :]))
        for j in range(natoms_per_mol):
            for k in nbr[j]:
                for i in nbr[j]:
                    if i == k:
                        continue
                    for l in nbr[k]:
                        if l == j or l <= i:          # unique direction
                            continue
                        lab = f"{sym0[i]}-{sym0[j]}-{sym0[k]}-{sym0[l]}"
                        dihedrals.setdefault(lab, []).append(
                            _dihedral(rel[:, i, :], rel[:, j, :],
                                      rel[:, k, :], rel[:, l, :]))
        # principal axis (linear molecules fall back to the first bond)
        cen = pos - pos.mean(axis=1, keepdims=True)
        cen -= cell * np.round(cen / cell)
        cov = np.einsum("mki,mkj->mij", cen, cen)
        _, vec = np.linalg.eigh(cov)
        axis = vec[:, :, 2]
        if natoms_per_mol == 3:
            v1 = rel[:, 1, :] - rel[:, 0, :]
            v2 = rel[:, 2, :] - rel[:, 0, :]
            v1 = v1 / np.maximum(np.linalg.norm(v1, axis=1)[:, None], 1e-12)
            v2 = v2 / np.maximum(np.linalg.norm(v2, axis=1)[:, None], 1e-12)
            if float(np.abs((v1 * v2).sum(-1).mean())) > 0.99:   # linear (CO2)
                axis = v2
        axes_all.append(axis.astype(np.float32))
        n_seen += 1

    out = {"nframes": n_seen}
    for name, store in (("bond", bonds), ("angle", angles),
                        ("dihedral", dihedrals)):
        for lab, vals in store.items():
            v = np.concatenate(vals)
            out[f"{name}_{lab}"] = v.astype(np.float32)
    axes = np.stack(axes_all) if axes_all else np.zeros((0, 0, 3), np.float32)
    out["axes"] = axes
    if len(axes):
        cosz = axes[:, :, 2].ravel()
        out["p2_mean"] = np.float32(np.mean(0.5 * (3 * cosz ** 2 - 1)))
        nf = min(len(axes), axes_frames)
        lags = np.unique(np.linspace(0, nf - 1, min(nf, 30)).astype(int))
        corr = []
        for lag in lags:
            u0 = axes[:nf - lag]
            u1 = axes[lag:nf]
            dot = (u0 * u1).sum(-1)
            corr.append(float(np.mean(0.5 * (3 * dot ** 2 - 1))))
        out["rot_lag_frames"] = lags
        out["rot_corr"] = np.asarray(corr, dtype=np.float32)
    return out
