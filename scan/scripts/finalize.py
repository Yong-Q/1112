#!/usr/bin/env python3
"""Aggregate post-processed states into a density-scan summary.

Reads scan/post/<sim_id>/les_c1c2.json for every state, builds
scan/analysis/scan_summary.csv (rho*, rho, T, c2 metadata) and one combined
scan/analysis/c2_grid.npz holding r grid + per-state c2(r).

Usage:  python scripts/finalize.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

SCAN = Path("/publicfs10/fs10-m9/home/m9s005258/qy/cdft/scan")
RHO_STAR_C = 0.316
RHO_C = {"H2O": 0.01076, "CO2": 0.00640}


def main() -> int:
    post = SCAN / "post"
    markers = sorted(post.glob("*/les_c1c2.json"))
    if not markers:
        print("no post-processed states yet")
        return 1
    out = SCAN / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    c2 = {}
    r_common = None
    for mk in markers:
        sim_id = mk.parent.name
        payload = json.loads(mk.read_text())
        rec = payload.get("c1c2", {})
        species = rec.get("species") or sim_id.split("-")[0]
        rho = float(rec.get("density_mol_A3", 0.0))
        rhostar = rho * RHO_STAR_C / RHO_C.get(species, 1.0)
        f2 = mk.parent / "f2"
        sim = rec.get("sim_id", sim_id)
        r_path = f2 / f"r_values_{sim}.npy"
        c2_path = f2 / f"c2_{sim}.npy"
        if r_path.exists() and c2_path.exists():
            r = np.load(r_path)
            c2r = np.load(c2_path)
            if r_common is None:
                r_common = r
            c2[sim_id] = c2r
        rows.append({
            "sim_id": sim_id, "species": species,
            "rho_star": f"{rhostar:.4f}", "rho_mol_A3": f"{rho:.6f}",
            "T_K": rec.get("T_K", ""), "n_frames": rec.get("n_frames", ""),
            "c2_pair": " ".join(rec.get("c2_pair", [])),
            "c2_rmax_A": rec.get("c2_rmax_A", ""),
            "c2_bins": rec.get("c2_bins", ""),
            "rho_pair_A3": rec.get("rho_pair_A3", ""),
        })
    fields = ["sim_id", "species", "rho_star", "rho_mol_A3", "T_K",
              "n_frames", "c2_pair", "c2_rmax_A", "c2_bins", "rho_pair_A3"]
    with open(out / "scan_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    if r_common is not None:
        np.savez(out / "c2_grid.npz", r=r_common,
                 **{k: v for k, v in c2.items()})
    print(f"wrote {len(rows)} states -> {out/'scan_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
