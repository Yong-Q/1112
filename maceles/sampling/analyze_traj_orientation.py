#!/usr/bin/env python3
"""Extract structural/orientational information from a water LAMMPS dump.

Assumes the dump was written with `id type x y z` and atom ids ordered as
molecule blocks (O, H, H) — true for the six_3000 boxes and their dumps.
Outputs: H-O-H angle and O-H length distributions, dipole orientation
distribution, dipole autocorrelation, tetrahedral order parameter,
hydrogen-bond counts, and O-O/O-H/H-H g(r).

Usage:
    python sampling/analyze_traj_orientation.py --traj traj.dump --outdir analysis
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import ase.io
from ase.neighborlist import neighbor_list
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ELEMENTS = ["C", "H", "N", "O"]
TYPE_O, TYPE_H = 4, 2


def load_frames(path, stride=1):
    frames = []
    for i, atoms in enumerate(ase.io.iread(str(path), format="lammps-dump-text",
                                           specorder=ELEMENTS)):
        if i % stride == 0:
            frames.append(atoms)
    return frames


def rdf_from_dist(dist, mask_a, mask_b, rmax, bins, volume):
    ia = np.where(mask_a)[0]
    ib = np.where(mask_b)[0]
    block = dist[np.ix_(ia, ib)]
    if mask_a is mask_b:
        iu = np.triu_indices(len(ia), k=1)
        rr = block[iu]
        expected = 0.5 * len(ia) * (len(ia) - 1)
    else:
        rr = block.reshape(-1)
        expected = len(ia) * len(ib)
    edges = np.linspace(0, rmax, bins + 1)
    h, _ = np.histogram(rr, bins=edges)
    shell = 4 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3) / 3
    g = h / np.maximum(expected * shell / volume, 1e-30)
    return 0.5 * (edges[1:] + edges[:-1]), g


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--rmax", type=float, default=8.0)
    ap.add_argument("--rdf-bins", type=int, default=100)
    ap.add_argument("--rdf-stride", type=int, default=5)
    args = ap.parse_args(argv)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    frames = load_frames(args.traj, stride=1)
    n_atoms = len(frames[0])
    n_mol = n_atoms // 3
    mono = frames[0].get_chemical_symbols()
    assert mono[0] == "O" and mono[1] == "H" and mono[2] == "H"

    angles, oh_lengths, dipoles = [], [], []
    q_tet = []
    hb_count = []
    for atoms in frames:
        pos = atoms.get_positions()
        pos = pos.reshape(n_mol, 3, 3)
        o = pos[:, 0, :]
        h1 = pos[:, 1, :]
        h2 = pos[:, 2, :]
        v1 = h1 - o
        v2 = h2 - o
        # minimum image for molecules straddling the periodic boundary
        L = atoms.cell.lengths()
        v1 -= L * np.round(v1 / L)
        v2 -= L * np.round(v2 / L)
        l1 = np.linalg.norm(v1, axis=1)
        l2 = np.linalg.norm(v2, axis=1)
        oh_lengths.append(np.concatenate([l1, l2]))
        cosang = np.sum(v1 * v2, axis=1) / (l1 * l2)
        angles.append(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
        d = v1 + v2
        d /= np.linalg.norm(d, axis=1)[:, None]
        dipoles.append(d)
        # tetrahedral order from the 3 nearest O neighbours (plus the central
        # atom's own 4th-shell omission): use 4 nearest O including PBC.
        diff = o[None, :, :] - o[:, None, :]
        diff -= atoms.cell.lengths() * np.round(diff / atoms.cell.lengths())
        dist = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(dist, 1e9)
        nn = np.argsort(dist, axis=1)[:, :4]
        q = np.zeros(n_mol)
        for a in range(n_mol):
            vecs = diff[a, nn[a]]
            vecs /= np.linalg.norm(vecs, axis=1)[:, None]
            s = 0.0
            cnt = 0
            for i in range(4):
                for j in range(i + 1, 4):
                    s += (np.dot(vecs[i], vecs[j]) + 1 / 3) ** 2
                    cnt += 1
            q[a] = 1 - (3 / 8) * s
        q_tet.append(q)
        # hydrogen bonds via a neighbor list (O...H < 2.5 A + O-H...O > 150)
        sym = np.array(atoms.get_chemical_symbols())
        ii, jj, dd, dvec = neighbor_list("ijdD", atoms, 2.5)
        hb = 0
        for hi in np.where(sym == "H")[0]:
            sel = (ii == hi) & (sym[jj] == "O")
            if not sel.any():
                continue
            o_list = jj[sel]
            vecs = dvec[sel]              # H -> O (minimum image)
            dho = dd[sel]
            ip = np.argmin(dho)
            parent = o_list[ip]
            vh = vecs[ip]                 # H -> parent O
            for k, ao in enumerate(o_list):
                if ao == parent:
                    continue
                va = vecs[k]              # H -> acceptor O
                cosv = np.dot(vh, va) / (np.linalg.norm(vh) * np.linalg.norm(va))
                # O-H...O angle at H: H->parent vs H->acceptor; true angle >150
                if np.degrees(np.arccos(np.clip(cosv, -1, 1))) > 150:
                    hb += 1
        hb_count.append(hb / n_mol)

    angles = np.concatenate(angles)
    oh_lengths = np.concatenate(oh_lengths)
    dipoles = np.stack(dipoles)            # (nframes, nmol, 3)
    q_tet = np.concatenate(q_tet)
    hb_count = np.array(hb_count)

    # dipole autocorrelation C(tau) = <d(t).d(t+tau)>
    nf = len(dipoles)
    c_t = []
    for lag in range(nf):
        prod = np.sum(dipoles[:nf - lag] * dipoles[lag:], axis=2).mean()
        c_t.append(float(prod))
    c_t = np.array(c_t)

    cos_theta = dipoles[..., 2].ravel()
    p2 = float(np.mean(0.5 * (3 * cos_theta ** 2 - 1)))

    # partial RDFs on a subset of frames
    sub = load_frames(args.traj, stride=args.rdf_stride)
    rmax = args.rmax
    nb = args.rdf_bins
    r_oo = r_oh = r_hh = None
    g_oo = g_oh = g_hh = 0.0
    for atoms in sub:
        dist = atoms.get_all_distances(mic=True)
        sym = np.array(atoms.get_chemical_symbols())
        mO, mH = sym == "O", sym == "H"
        vol = atoms.get_volume()
        if r_oo is None:
            r_oo, g_oo = rdf_from_dist(dist, mO, mO, rmax, nb, vol)
            r_oh, g_oh = rdf_from_dist(dist, mO, mH, rmax, nb, vol)
            r_hh, g_hh = rdf_from_dist(dist, mH, mH, rmax, nb, vol)
        else:
            _, x = rdf_from_dist(dist, mO, mO, rmax, nb, vol); g_oo = g_oo + x
            _, x = rdf_from_dist(dist, mO, mH, rmax, nb, vol); g_oh = g_oh + x
            _, x = rdf_from_dist(dist, mH, mH, rmax, nb, vol); g_hh = g_hh + x
    k = len(sub)
    g_oo, g_oh, g_hh = g_oo / k, g_oh / k, g_hh / k

    np.savez(out / "orientation.npz", angles=angles, oh_lengths=oh_lengths,
             dipoles=dipoles, q_tet=q_tet, hb_per_mol=hb_count,
             c_dipole=c_t, cos_theta=cos_theta,
             r_oo=r_oo, g_oo=g_oo, r_oh=r_oh, g_oh=g_oh,
             r_hh=r_hh, g_hh=g_hh)

    summary = {
        "n_frames": len(frames), "n_molecules": int(n_mol),
        "angle_mean_deg": float(angles.mean()),
        "angle_std_deg": float(angles.std()),
        "oh_mean_A": float(oh_lengths.mean()),
        "oh_std_A": float(oh_lengths.std()),
        "q_tet_mean": float(q_tet.mean()),
        "q_tet_std": float(q_tet.std()),
        "hbonds_per_molecule": float(hb_count.mean()),
        "dipole_P2": p2,
        "dipole_corr_tau0": float(c_t[0]),
        "dipole_corr_tau_0p5ps": float(c_t[1]) if len(c_t) > 1 else None,
        "frame_interval_ps": 0.5,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].hist(angles, bins=80, color="#0072B2")
    axes[0, 0].set_xlabel("H-O-H angle (deg)"); axes[0, 0].set_ylabel("count")
    axes[0, 1].hist(oh_lengths, bins=80, color="#D55E00")
    axes[0, 1].set_xlabel("O-H length (A)"); axes[0, 1].set_ylabel("count")
    axes[1, 0].hist(cos_theta, bins=60, color="#009E73")
    axes[1, 0].set_xlabel("dipole cos(theta_z)"); axes[1, 0].set_ylabel("count")
    axes[1, 1].plot(np.arange(len(c_t)) * 0.5, c_t, "o-", color="#CC79A7")
    axes[1, 1].set_xlabel("lag (ps)"); axes[1, 1].set_ylabel("dipole autocorrelation")
    fig.tight_layout(); fig.savefig(out / "orientation.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r_oo, g_oo, label="O-O"); ax.plot(r_oh, g_oh, label="O-H")
    ax.plot(r_hh, g_hh, label="H-H")
    ax.set_xlabel("r (A)"); ax.set_ylabel("g(r)"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "rdf_partial.png", dpi=150)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"[orientation] wrote {out}/orientation.png, rdf_partial.png, orientation.npz")


if __name__ == "__main__":
    raise SystemExit(main())
