#!/usr/bin/env python3
"""All-pair radial + angular pair correlations for one trajectory.

For every element pair present (e.g. H-H, H-O, O-O in water) this returns

* the radial distribution g_ab(r),
* the spherical-harmonic anisotropy of the pair-vector directions,
  a_l(r) = 4pi/(2l+1) * sum_m |<Y_lm>|^2  (0 = isotropic, ->1 = ordered),
  for l = 1, 2,
* the cos(theta_z) distribution inside the first coordination shell.

Everything is accumulated from neighbour lists, so the full 3D pair-vector
information is reduced to compact profiles that fit next to the c1/c2 output.
"""
from __future__ import annotations

import numpy as np

SQ3_4PI = np.sqrt(3.0 / (4.0 * np.pi))
SQ3_8PI = np.sqrt(3.0 / (8.0 * np.pi))
SQ5_16PI = np.sqrt(5.0 / (16.0 * np.pi))
SQ15_8PI = np.sqrt(15.0 / (8.0 * np.pi))
SQ15_32PI = np.sqrt(15.0 / (32.0 * np.pi))


def _ylm(u):
    """Real/complex spherical harmonics l<=2 for unit vectors u (n,3)."""
    x, y, z = u[:, 0], u[:, 1], u[:, 2]
    ct = z
    st = np.sqrt(np.maximum(0.0, 1.0 - ct * ct))
    phi = np.arctan2(y, x)
    e1 = np.exp(1j * phi)
    e2 = np.exp(2j * phi)
    y10 = SQ3_4PI * ct
    y11 = -SQ3_8PI * st * e1
    y20 = SQ5_16PI * (3.0 * ct * ct - 1.0)
    y21 = -SQ15_8PI * st * ct * e1
    y22 = SQ15_32PI * st * st * e2
    return (np.stack([y11, y10, np.conj(y11)], axis=1),
            np.stack([y22, y21, y20, np.conj(y21), np.conj(y22)], axis=1))


def pair_correlations(frames, rmax=8.0, bins=100, lmax=2, shell_width=0.4):
    """Accumulate g_ab(r), anisotropy a_l(r) and first-shell cos(theta).

    ``frames`` may be ``ase.Atoms`` or dicts from ``sampling.dump_mol`` (which
    carry molecule ids).  With molecule ids the pair histogram is split into
    intramolecular and intermolecular contributions — essential for species
    with repeated elements (H2O has two H, C2H6 six H), where the H-H and O-H
    intramolecular peaks otherwise dominate the fluid structure.
    """
    from ase import Atoms
    from ase.neighborlist import neighbor_list

    edges = np.linspace(0.0, rmax, bins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    shell_vol = 4.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3) / 3.0

    acc = {}          # pair -> dict(hist, sh1, sh2, n, cos, rho_pair)
    nframes = 0
    for fr in frames:
        if isinstance(fr, dict):
            sym = np.array([str(x) for x in fr["symbols"]])
            atoms = Atoms(symbols=sym.tolist(), positions=fr["positions"],
                          cell=fr["cell"], pbc=True)
            mol = fr.get("mol")
        else:
            atoms = fr
            sym = np.array(atoms.get_chemical_symbols())
            mol = None
        vol = atoms.get_volume()
        ii, jj, D = neighbor_list("ijD", atoms, rmax)
        r = np.linalg.norm(D, axis=1)
        keep = r > 1e-8
        ii, jj, D, r = ii[keep], jj[keep], D[keep], r[keep]
        if mol is not None:
            mol = np.asarray(mol)
            inter = mol[ii] != mol[jj]
        else:
            inter = np.ones(len(ii), dtype=bool)
        u = D / r[:, None]
        Y1, Y2 = _ylm(u)
        idx = np.clip(np.digitize(r, edges) - 1, 0, bins - 1)
        uniq = np.unique(sym)
        for ia, a in enumerate(uniq):
            for b in uniq[ia:]:
                m = (sym[ii] == a) & (sym[jj] == b)
                if not m.any():
                    continue
                key = (str(a), str(b))
                d = acc.setdefault(key, {
                    "hist": np.zeros(bins), "hist_inter": np.zeros(bins),
                    "sh1": np.zeros((bins, 3), complex),
                    "sh2": np.zeros((bins, 5), complex), "n": np.zeros(bins),
                    "cos": np.zeros(36), "nc": 0, "shell_r": None})
                rm, idxm = r[m], idx[m]
                d["hist"] += np.bincount(idxm, minlength=bins)
                mi = m & inter
                if mi.any():
                    d["hist_inter"] += np.bincount(idx[mi], minlength=bins)
                d["n"] += np.bincount(idxm, minlength=bins)
                np.add.at(d["sh1"], idxm, Y1[m])
                np.add.at(d["sh2"], idxm, Y2[m])
                peak = int(np.argmax(d["hist"])) if d["hist"].max() > 0 else 0
                rpk = centers[peak]
                cs = (rm > rpk - shell_width) & (rm < rpk + shell_width)
                if cs.any():
                    d["cos"] += np.histogram(
                        np.clip(u[m][cs, 2], -1, 1),
                        bins=np.linspace(-1, 1, 37))[0]
                    d["nc"] += int(cs.sum())
                    d["shell_r"] = rpk
        nframes += 1

    out = {"r": centers, "nframes": nframes}
    na = {}
    # pair counts from the last frame (same composition every frame)
    for (a, b), d in acc.items():
        na[(a, b)] = (int((sym == a).sum()), int((sym == b).sum()))
    for (a, b), d in acc.items():
        n_a, n_b = na[(a, b)]
        npair = (0.5 * n_a * (n_a - 1)) if a == b else n_a * n_b
        rho = npair / (vol * (1 if a == b else 1))   # pairs per volume
        g = (d["hist"] / nframes) / np.maximum(rho * shell_vol, 1e-30)
        a1 = 4.0 * np.pi / 3.0 * (
            np.abs(d["sh1"] / np.maximum(d["n"], 1)[:, None]) ** 2).sum(1)
        a2 = 4.0 * np.pi / 5.0 * (
            np.abs(d["sh2"] / np.maximum(d["n"], 1)[:, None]) ** 2).sum(1)
        out[f"g_{a}{b}"] = g
        out[f"ginter_{a}{b}"] = (d["hist_inter"] / nframes) / np.maximum(
            rho * shell_vol, 1e-30)
        out[f"a1_{a}{b}"] = np.where(d["n"] > 0, a1, 0.0)
        out[f"a2_{a}{b}"] = np.where(d["n"] > 0, a2, 0.0)
        out[f"cos_{a}{b}"] = (d["cos"] / d["nc"]) if d["nc"] else d["cos"]
        out[f"shellr_{a}{b}"] = np.float64(d["shell_r"] or 0.0)
    return out
