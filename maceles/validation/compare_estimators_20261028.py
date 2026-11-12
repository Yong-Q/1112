#!/usr/bin/env python3
"""Compare estimators for g(r), density and the reweighting ratio.

Uses the stored 501-frame OFF23 trajectory and the POLAR energies:
  - g_OO: histogram vs a smooth Gaussian-basis (spectral-like) estimator;
  - density profile: counting vs force-assisted integration (POLAR forces);
  - reweighting: naive sparse ratio vs control-variate multirate.

Outputs JSON/CSV/PNG under docs/results/estimator_comparison_20261028/.
"""
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "validation/off23_500ps_water1152_2026-10-23"
OUT = ROOT / "docs/results/estimator_comparison_20261028"
OUT.mkdir(parents=True, exist_ok=True)
GRAPH = "/workspace/group/shared/bza/external-repos/graph_electrostatics-0.4.0"
if GRAPH not in sys.path:
    sys.path.insert(0, GRAPH)
KB = 8.617333262e-5
T = 300.0
BETA = 1.0 / (KB * T)


def read_dump(path):
    text = Path(path).read_text()
    for chunk in text.split("ITEM: TIMESTEP")[1:]:
        body = "ITEM: TIMESTEP" + chunk
        if "ITEM: ATOMS" not in body:
            continue
        head, _, _ = body.partition("ITEM: ATOMS")
        try:
            n_atoms = int(head.split("ITEM: NUMBER OF ATOMS")[1].split()[0])
        except Exception:
            continue
        ts = int(chunk.strip().splitlines()[0])
        lines = body.split("ITEM: ATOMS", 1)[1].strip().splitlines()[1:]
        if len(lines) < n_atoms:
            continue
        arr = np.array([ln.split() for ln in lines[:n_atoms]], dtype=float)
        yield ts, arr[:, 1].astype(int), arr[:, 2:5]


def hist_g(pos, L, n_o, bins, rmax):
    dx = pos[:, None, :] - pos[None, :, :]
    dx -= L * np.round(dx / L)
    r = np.sqrt((dx ** 2).sum(-1))
    iu = np.triu_indices(len(pos), 1)
    h, _ = np.histogram(r[iu], bins=bins, range=(0.0, rmax))
    shell = 4.0 * np.pi / 3.0 * (bins[1:] ** 3 - bins[:-1] ** 3)
    norm = n_o * (n_o - 1) / 2.0 * shell / (L[0] * L[1] * L[2])
    return h / norm


def spectral_sine(pos, L, n_o, rmax, nmax):
    """Orthonormal sine basis under the 4 pi r^2 dr measure.

    psi_n(r) = sqrt(2/rmax) sin(n pi r/rmax) / (2 r sqrt(pi))
    c_n = V/(N(N-1)) <sum_{i<j} psi_n(r_ij)>
    g(r) = sum_n c_n psi_n(r)
    """
    dx = pos[:, None, :] - pos[None, :, :]
    dx -= L * np.round(dx / L)
    r = np.sqrt((dx ** 2).sum(-1))
    iu = np.triu_indices(len(pos), 1)
    rij = r[iu]
    n = np.arange(1, nmax + 1)
    k = n * np.pi / rmax
    psi = (np.sqrt(2.0 / rmax) * np.sin(rij[:, None] * k[None, :])
           / (2.0 * rij[:, None] * np.sqrt(np.pi)))
    coef = 2.0 * (L[0] * L[1] * L[2]) / (n_o * (n_o - 1)) * psi.sum(0)
    rr = np.linspace(0.05, rmax, 800)
    basis = (np.sqrt(2.0 / rmax) * np.sin(rr[:, None] * k[None, :])
             / (2.0 * rr[:, None] * np.sqrt(np.pi)))
    return rr, basis @ coef


def spectral_g(pos, L, n_o, centers, sigma):
    dx = pos[:, None, :] - pos[None, :, :]
    dx -= L * np.round(dx / L)
    r = np.sqrt((dx ** 2).sum(-1))
    iu = np.triu_indices(len(pos), 1)
    rij = r[iu]
    phi = np.exp(-0.5 * ((rij[:, None] - centers[None, :]) / sigma) ** 2)
    # normalization: integral of phi over 4 pi r^2 dr
    rr = np.linspace(0.0, 15.0, 3000)
    integ = np.array([
        np.trapezoid(np.exp(-0.5 * ((rr - c) / sigma) ** 2) * 4 * np.pi * rr ** 2, rr)
        for c in centers])
    coef = (L[0] * L[1] * L[2]) / (n_o * (n_o - 1)) * phi.sum(0) / integ
    g = np.exp(-0.5 * ((rr[:, None] - centers[None, :]) / sigma) ** 2) @ coef
    return rr, g


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="also compute the force-assisted density")
    ap.add_argument("--force-device", default="cuda",
                    choices=["cuda", "cpu"])
    args = ap.parse_args()
    energies = []
    with open(PKG / "frames_501_total_energies.csv") as fh:
        next(fh)
        for line in fh:
            parts = line.strip().split(",")
            energies.append(float(parts[5]) - float(parts[4]))
    energies = np.array(energies)
    q = np.exp(-BETA * (energies - energies.min()))
    w = q / q.sum()
    frames = list(read_dump(PKG / "traj_500ps_1fs.lammpstrj"))
    print("frames", len(frames), flush=True)

    rmax, nbins = 8.0, 80
    bins = np.linspace(0.0, rmax, nbins + 1)
    centers = np.arange(0.3, 8.01, 0.2)
    sigma = 0.2
    hist = []
    spec = []
    rhist = 0.5 * (bins[1:] + bins[:-1])
    for k, (ts, types, pos) in enumerate(frames):
        L = np.array([23.3921419057029247, 23.3921419057029247,
                      35.0882128585543853])
        o = pos[types == 1]
        hist.append(hist_g(o, L, len(o), bins, rmax))
        rr, g = spectral_sine(o, L, len(o), rmax, 60)
        spec.append(np.interp(rhist, rr, g))
        if (k + 1) % 100 == 0:
            print("g", k + 1, flush=True)
    hist = np.array(hist)
    spec = np.array(spec)

    def block_mean(arr, wts, nblk=5):
        n = len(arr)
        idx = np.array_split(np.arange(n), nblk)
        est = np.array([np.average(arr[i], axis=0, weights=wts[i]) for i in idx])
        return est.mean(0), est.std(0) / np.sqrt(nblk)

    g_hist, e_hist = block_mean(hist, w)
    g_spec, e_spec = block_mean(spec, w)

    # density profile, counting
    zedges = np.arange(0.0, 35.0882128585543853 + 0.5, 0.5)
    zc = 0.5 * (zedges[1:] + zedges[:-1])
    Lx = Ly = 23.3921419057029247
    rho_cnt = []
    for ts, types, pos in frames:
        h, _ = np.histogram(pos[types == 1, 2], bins=zedges)
        rho_cnt.append(h / (Lx * Ly * (zedges[1:] - zedges[:-1])))
    rho_cnt = np.array(rho_cnt)
    rho_c, e_rho_c = block_mean(rho_cnt, w)

    # multirate comparison (CPU only)
    qA = (q[:, None] * hist)
    sparse = np.arange(0, len(q), 5)
    dense = np.arange(len(q))
    Y = qA[sparse].mean(0)
    Xs = q[sparse].mean()
    Xd = q[dense].mean()
    naive = Y / Xs
    cov = ((qA[sparse] - Y) * (q[sparse, None] - Xs)).mean(0)
    beta_cv = cov / np.var(q[sparse])
    cv = (Y + beta_cv * (Xd - Xs)) / Xd
    offs = []
    for s0 in range(5):
        idx = np.arange(s0, len(q), 5)
        offs.append(qA[idx].mean(0) / q[idx].mean())
    offs = np.array(offs)

    summary = {
        "n_frames": len(frames),
        "g_rms_hist_vs_spec": float(np.sqrt(np.mean((g_hist - g_spec) ** 2))),
        "g_block_err_hist_mean": float(np.mean(e_hist)),
        "g_block_err_spec_mean": float(np.mean(e_spec)),
        "rho_block_err_count_mean": float(np.mean(e_rho_c)),
        "multirate": {
            "naive_vs_cv_rms": float(np.sqrt(np.mean((naive - cv) ** 2))),
            "offset_spread_rms": float(np.sqrt(np.mean(np.var(offs, axis=0)))),
            "n_dense": len(dense), "n_sparse": len(sparse),
        },
    }

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(force=None):
        (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
        np.savetxt(OUT / "g_hist.csv",
                   np.column_stack([rhist, g_hist, e_hist, g_spec]),
                   delimiter=",", header="r,g_hist,g_hist_err,g_spec",
                   comments="")
        cols = [zc, rho_c, e_rho_c] + ([force[0], force[1]] if force else [])
        header = "z,rho_count,rho_count_err" + (
            ",rho_force,rho_force_err" if force else "")
        np.savetxt(OUT / "density.csv", np.column_stack(cols), delimiter=",",
                   header=header, comments="")
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        ax[0].errorbar(rhist, g_hist, yerr=e_hist, fmt="o", ms=2.5, lw=0.8,
                       label="histogram", color="#0072B2")
        ax[0].plot(rhist, g_spec, lw=1.2, label="Gaussian basis",
                   color="#D55E00")
        ax[0].set_xlabel("r (Å)"); ax[0].set_ylabel("g$_{OO}$(r)")
        ax[0].legend()
        ax[1].errorbar(zc, rho_c, yerr=e_rho_c, fmt="o", ms=2.5, lw=0.8,
                       label="counting", color="#0072B2")
        if force:
            ax[1].plot(zc, force[0], lw=1.2, label="force-assisted",
                       color="#009E73")
        ax[1].set_xlabel("z (Å)"); ax[1].set_ylabel("ρ$_O$(z)")
        ax[1].legend()
        for a in ax:
            a.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(OUT / "estimator_comparison.png", dpi=180)

    if not args.force:
        save()
        print(json.dumps(summary, indent=1))
        print("wrote (count only)", OUT)
        return

    # force-assisted density on a subset
    step = 25
    sel = np.arange(0, len(frames), step)
    import subprocess as sp
    import time as _t
    t0 = _t.time()
    while args.force_device == "cuda" and _t.time() - t0 < 7200:
        out = sp.run(["nvidia-smi", "--query-gpu=memory.free",
                      "--format=csv,noheader,nounits"],
                     capture_output=True, text=True)
        free = min(int(x) for x in out.stdout.split())
        if free >= 12000:
            break
        print(f"[gpu] free {free} MiB; waiting for the force pass", flush=True)
        _t.sleep(60)
    from mace.calculators import mace_polar
    model = ("/workspace/group/shared/bza/task3_sampling_database/mlip/models/"
             "mace-polar/MACE-POLAR-1-M.model")
    from ase import Atoms
    dtype = "float64" if args.force_device == "cuda" else "float32"
    calc = mace_polar(model=model, device=args.force_device,
                      default_dtype=dtype)
    rho_force = []
    rho_cnt_sub = []
    for n, k in enumerate(sel):
        ts, types, pos = frames[k]
        atoms = Atoms(numbers=np.where(types == 1, 8, 1), positions=pos,
                      cell=[Lx, Ly, 35.0882128585543853], pbc=True)
        atoms.calc = calc
        f = atoms.get_forces()
        fz = f[types == 1, 2]
        h, _ = np.histogram(pos[types == 1, 2], bins=zedges)
        fh, _ = np.histogram(pos[types == 1, 2], bins=zedges, weights=fz)
        area = Lx * Ly
        dz = zedges[1:] - zedges[:-1]
        fdens = fh / (area * dz)
        rho = np.cumsum(BETA * fdens * dz)
        rho += rho_c[0] - rho[0]
        rho_force.append(rho)
        rho_cnt_sub.append(h / (area * dz))
        if (n + 1) % 10 == 0:
            print("force", n + 1, len(sel), flush=True)
    rho_force = np.array(rho_force)
    rho_cnt_sub = np.array(rho_cnt_sub)
    wsub = w[sel] / w[sel].sum()
    rf, e_rf = block_mean(rho_force, wsub, nblk=5)
    rc, e_rc = block_mean(rho_cnt_sub, wsub, nblk=5)
    summary["rho_rms_count_vs_force"] = float(
        np.sqrt(np.mean((rho_c - rf) ** 2)))
    summary["rho_block_err_force_mean"] = float(np.mean(e_rf))
    save((rf, e_rf))
    print(json.dumps(summary, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
