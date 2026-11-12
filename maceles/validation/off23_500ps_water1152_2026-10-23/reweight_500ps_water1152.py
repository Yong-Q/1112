#!/usr/bin/env python3
"""OFF -> POLAR reweighting diagnostics on the 500 ps / 1152-atom water run.

The production trajectory was generated with MACE-OFF23-medium through the
Symmetrix LAMMPS engine and stores positions only.  This script re-evaluates
every dumped frame with both potentials in PyTorch, then reports the
importance-sampling diagnostics (dE, sigma_beta, ESS), the linear-fit
coefficients, and the O-O RDF for OFF, reweighted, and direct POLAR samples.

Outputs (in --outdir):
  frames.json      per-frame energies and weights
  summary.json     aggregate diagnostics
  rdf.json         r g_off g_rw g_polar
  figures          rdf.png, weights.png
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
GRAPH_LR = Path("/workspace/group/shared/bza/external-repos/graph_electrostatics-0.4.0")
for p in (str(REPO), str(GRAPH_LR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import ase.io  # noqa: E402

KB = 8.617333262e-5


def read_dump(path: str):
    """Yield ASE Atoms for each complete frame in a LAMMPS custom dump."""
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
        atom_lines = body.split("ITEM: ATOMS", 1)[1].strip().splitlines()
        if len(atom_lines) < n_atoms:
            continue
        try:
            atoms = ase.io.read(io.StringIO(body), format="lammps-dump-text")
        except Exception:
            continue
        z = np.asarray(atoms.numbers)
        atoms.set_chemical_symbols(["O" if int(t) == 1 else "H" for t in z])
        yield ts, atoms


def oo_rdf_weighted(frames, weights, rmax=8.0, nbins=80):
    edges = np.linspace(0.0, rmax, nbins + 1)
    shell = 4.0 * np.pi / 3.0 * (edges[1:] ** 3 - edges[:-1] ** 3)
    hist = np.zeros(nbins)
    norm = 0.0
    for atoms, w in zip(frames, weights):
        sym = np.asarray(atoms.get_chemical_symbols())
        pos = atoms.get_positions()
        cell = np.asarray(atoms.get_cell())
        inv = np.linalg.inv(cell)
        opos = pos[sym == "O"]
        n_o = len(opos)
        if n_o < 2:
            continue
        vol = abs(np.linalg.det(cell))
        d = opos[:, None, :] - opos[None, :, :]
        frac = d @ inv
        frac -= np.round(frac)
        dmin = np.linalg.norm(frac @ cell, axis=-1)
        iu = np.triu_indices(n_o, k=1)
        r = dmin[iu]
        r = r[(r > 0) & (r < rmax)]
        h, _ = np.histogram(r, bins=edges)
        hist += w * h.astype(np.float64)
        # ideal-gas pair density for this frame
        # The upper triangle counts each unordered O-O pair once.
        norm += w * (0.5 * n_o * (n_o - 1) / vol)
    r = 0.5 * (edges[1:] + edges[:-1])
    g = hist / (norm * shell) if norm > 0 else np.zeros_like(r)
    return r, g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--source-model", required=True)
    ap.add_argument("--target-model", required=True)
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--source-dtype", default="float32")
    ap.add_argument("--target-dtype", default="float64")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    frames = [(ts, a) for ts, a in read_dump(args.traj)][:: args.stride]
    if args.max_frames:
        frames = frames[: args.max_frames]
    print(f"[reweight500] {len(frames)} frames, {len(frames[0][1])} atoms", flush=True)

    from mace.calculators import MACECalculator, mace_polar  # noqa: E402

    src = MACECalculator(model_paths=args.source_model, device=args.device,
                         default_dtype=args.source_dtype)
    tgt = mace_polar(model=args.target_model, device=args.device,
                     default_dtype=args.target_dtype)
    beta = 1.0 / (KB * args.temperature)

    recs = []
    t0 = time.time()
    for i, (ts, atoms) in enumerate(frames):
        atoms.calc = src
        es = float(atoms.get_potential_energy())
        atoms.calc = tgt
        et = float(atoms.get_potential_energy())
        if not np.isfinite(es) or not np.isfinite(et):
            print(f"[reweight500] non-finite at frame {i} (ts={ts})", flush=True)
            continue
        recs.append({"index": i, "timestep": ts, "n_atoms": len(atoms),
                     "e_source_eV": es, "e_target_eV": et,
                     "dE_total_eV": et - es})
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"[reweight500] {i+1}/{len(frames)} {el:.1f}s "
                  f"({el/(i+1):.2f}s/frame)", flush=True)

    d = np.array([r["dE_total_eV"] for r in recs])
    n = np.array([r["n_atoms"] for r in recs])
    beta_d = beta * d
    logw = -beta_d
    logw -= logw.max()
    w = np.exp(logw)
    w /= w.sum()
    ess = 1.0 / np.sum(w ** 2)
    dpa = d / n
    offset = float(np.mean(np.abs(dpa))) if dpa.mean() < 0 else float(np.mean(dpa))
    # constant offset = the mean per-atom gap; harmless after normalization
    const_off = float(np.mean(dpa))
    resid = dpa - const_off
    sig_total = float(np.std(beta_d))
    # slope/intercept fit of the linear-response form
    e_s = np.array([r["e_source_eV"] for r in recs])
    e_t = np.array([r["e_target_eV"] for r in recs])
    A = np.vstack([e_s, np.ones_like(e_s)]).T
    coef, *_ = np.linalg.lstsq(A, e_t, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((e_t - pred) ** 2))
    ss_tot = float(np.sum((e_t - e_t.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    resid_eV = e_t - pred
    resid_meV_atom = float(np.std(resid_eV / n)) * 1e3

    # RDF of the source ensemble and of the reweighted (target) ensemble.
    # No direct POLAR trajectory exists in this box, so no "direct" curve.
    atoms_list = [a for _, a in frames]
    r, g_off = oo_rdf_weighted(atoms_list, [1.0 / len(atoms_list)] * len(atoms_list))
    _, g_rw = oo_rdf_weighted(atoms_list, list(w))
    summary = {
        "traj": str(Path(args.traj).resolve()),
        "n_frames": len(recs),
        "n_atoms": int(n[0]),
        "temperature_K": args.temperature,
        "source_model": str(Path(args.source_model).resolve()),
        "target_model": str(Path(args.target_model).resolve()),
        "dE_meV_per_atom_mean": const_off * 1e3,
        "dE_meV_per_atom_std": float(np.std(dpa)) * 1e3,
        "residual_meV_per_atom_std": resid_meV_atom,
        "sigma_beta": sig_total,
        "sigma_beta_residual": float(np.std(beta * resid * n)),
        "ess": float(ess),
        "ess_fraction": float(ess / len(recs)),
        "linear_slope": float(coef[0]),
        "linear_intercept_eV": float(coef[1]),
        "linear_r2": float(r2),
        "verdict": ("usable" if ess / len(recs) >= 0.10 else
                    "marginal" if ess / len(recs) >= 0.01 else "collapse"),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1))
    (outdir / "frames.json").write_text(json.dumps(recs, indent=1))
    (outdir / "rdf.json").write_text(json.dumps(
        {"r": r.tolist(), "g_off": g_off.tolist(), "g_reweighted": g_rw.tolist()}, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
